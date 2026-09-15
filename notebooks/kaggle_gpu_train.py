"""Kaggle GPU Training Script: Contrastive Peak Transformer (P0-1).

Designed to run on Kaggle GPU (T4 or P100) within quota limits.
Features:
- PyTorch AMP (Automatic Mixed Precision: fp16) for 3x training speedup and low VRAM footprint.
- Structure-disjoint holdout on enveda-np-examples (Bruker timsTOF benchmark).
- Dynamic batch collation with variable peak lists (d_model=256, 4 Transformer layers, 8 heads).
- InfoNCE contrastive alignment between spectra and COCONUT/training molecules.
- Checkpoints saved to best_peak_transformer.pt and best_molecule_encoder.pt.
"""

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, Dataset

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")

# -------------------------------------------------------------
# 1. Config & Kaggle Path Auto-Detection
# -------------------------------------------------------------
TRAIN_PATHS = [
    "/kaggle/input/enveda-casmi26-molecule-id-mass-spectra/train.parquet",
    "data/train.parquet",
    "train.parquet",
]
COCONUT_PATHS = [
    "/kaggle/input/enveda-coconut/coconut_indexed.parquet",
    "/kaggle/input/coconut-indexed/coconut_indexed.parquet",
    "data/external/coconut_indexed.parquet",
    "coconut_indexed.parquet",
]

TRAIN_PARQUET = next((p for p in TRAIN_PATHS if os.path.exists(p)), None)
COCONUT_PARQUET = next((p for p in COCONUT_PATHS if os.path.exists(p)), None)

print(f"Detected Train Parquet: {TRAIN_PARQUET}")
print(f"Detected COCONUT Parquet: {COCONUT_PARQUET}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compute Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# -------------------------------------------------------------
# 2. Sinusoidal & Transformer Architectures
# -------------------------------------------------------------
KNOWN_INSTRUMENTS = [
    "timsTOF", "Orbitrap", "QTOF", "LC-ESI-QTOF", "ESI-QFT", "qtof",
    "LC-ESI-QQ", "orbitrap", "Linear Ion Trap", "Hybrid FT", "FT-ICR", "Triple Quad", "unknown"
]
INSTRUMENT_TO_IDX = {inst: i for i, inst in enumerate(KNOWN_INSTRUMENTS)}

def get_instrument_idx(inst_str: Optional[str]) -> int:
    if not inst_str:
        return INSTRUMENT_TO_IDX["unknown"]
    for k, idx in INSTRUMENT_TO_IDX.items():
        if k.lower() in inst_str.lower():
            return idx
    return INSTRUMENT_TO_IDX["unknown"]

class SinusoidalEncoding(nn.Module):
    def __init__(self, dim: int = 64, max_period: float = 2000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = x.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class PeakTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, d_ff=512, dropout=0.1, d_latent=512):
        super().__init__()
        self.d_model = d_model
        self.mz_sinusoidal = SinusoidalEncoding(dim=64, max_period=2000.0)
        self.loss_sinusoidal = SinusoidalEncoding(dim=64, max_period=2000.0)
        self.peak_proj = nn.Sequential(
            nn.Linear(64 + 64 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prec_sinusoidal = SinusoidalEncoding(dim=64, max_period=2000.0)
        self.prec_proj = nn.Linear(64, d_model)
        self.ce_proj = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.inst_emb = nn.Embedding(len(KNOWN_INSTRUMENTS), d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.latent_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LayerNorm(d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_latent),
        )
        self.fp_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LayerNorm(d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 2048),
            nn.Sigmoid(),
        )

    def forward(self, peak_mzs, peak_ints, peak_mask, prec_mzs, ces, insts):
        B, P = peak_mzs.shape
        prec_exp = prec_mzs.unsqueeze(1).expand_as(peak_mzs)
        neutral_losses = torch.clamp(prec_exp - peak_mzs, min=0.0)

        mz_enc = self.mz_sinusoidal(peak_mzs)
        loss_enc = self.loss_sinusoidal(neutral_losses)
        sqrt_ints = torch.sqrt(torch.clamp(peak_ints, min=0.0)).unsqueeze(-1).float()

        raw_feat = torch.cat([mz_enc, loss_enc, sqrt_ints], dim=-1)
        peak_tokens = self.peak_proj(raw_feat)

        cls_base = self.cls_token.expand(B, 1, -1)
        prec_emb = self.prec_proj(self.prec_sinusoidal(prec_mzs)).unsqueeze(1)
        ce_emb = self.ce_proj(ces).unsqueeze(1)
        inst_emb = self.inst_emb(insts).unsqueeze(1)
        cls_token = cls_base + prec_emb + ce_emb + inst_emb

        tokens = torch.cat([cls_token, peak_tokens], dim=1)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=peak_mask.device)
        padding_mask = torch.cat([cls_mask, ~peak_mask], dim=1)

        out = self.transformer(tokens, src_key_padding_mask=padding_mask)
        out = self.norm(out)
        cls_out = out[:, 0, :]

        latent = F.normalize(self.latent_head(cls_out), p=2, dim=-1)
        pred_fp = self.fp_head(cls_out)
        return latent, pred_fp

class MoleculeEncoder(nn.Module):
    def __init__(self, in_features=2048, d_ff=512, d_latent=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_ff),
            nn.LayerNorm(d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.LayerNorm(d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_latent),
        )

    def forward(self, fps):
        return F.normalize(self.net(fps), p=2, dim=-1)

class ContrastiveInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, s_latent, m_latent):
        sim = torch.matmul(s_latent, m_latent.T) / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device, dtype=torch.long)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0

# -------------------------------------------------------------
# 3. Main Execution Function
# -------------------------------------------------------------
if __name__ == "__main__":
    from src.models.train_contrastive import train_contrastive_alignment

    if TRAIN_PARQUET is None:
        raise FileNotFoundError("Could not find train.parquet!")

    train_contrastive_alignment(
        train_parquet_path=TRAIN_PARQUET,
        coconut_parquet_path=COCONUT_PARQUET or "data/external/coconut_indexed.parquet",
        output_dir="/kaggle/working" if os.path.exists("/kaggle/working") else "models",
        max_train_samples=500000,
        epochs=30,
        batch_size=256 if DEVICE == "cuda" else 16,
        lr=3e-4,
        smoke_test=False,
    )
