"""Contrastive Peak Transformer for Spectrum-Molecule Alignment (P0-1).

Architectural Features:
1. Sinusoidal continuous m/z encoding (preserves sub-Da high-resolution mass precision).
2. Continuous neutral loss (precursor_mz - peak_mz) sinusoidal encoding.
3. Stepped collision energy (CE in eV: mean, min, max, has_ce) continuous conditioning.
4. Categorical instrument type embedding (timsTOF, Orbitrap, QTOF, etc.).
5. Multi-head self-attention over variable-length peak lists with attention masking.
6. Contrastive latent projection head (shared 512-dim unit sphere) + auxiliary Morgan fingerprint head.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Common instrument types in training & test data
KNOWN_INSTRUMENTS = [
    "timsTOF",
    "Orbitrap",
    "QTOF",
    "LC-ESI-QTOF",
    "ESI-QFT",
    "qtof",
    "LC-ESI-QQ",
    "orbitrap",
    "Linear Ion Trap",
    "Hybrid FT",
    "FT-ICR",
    "Triple Quad",
    "unknown",
]
INSTRUMENT_TO_IDX: Dict[str, int] = {
    inst: idx for idx, inst in enumerate(KNOWN_INSTRUMENTS)
}


def get_instrument_idx(instrument_str: Optional[str]) -> int:
    """Map raw instrument string to integer ID."""
    if not instrument_str:
        return INSTRUMENT_TO_IDX["unknown"]
    for inst, idx in INSTRUMENT_TO_IDX.items():
        if inst.lower() in instrument_str.lower():
            return idx
    return INSTRUMENT_TO_IDX["unknown"]


class SinusoidalEncoding(nn.Module):
    """Continuous sinusoidal positional encoding for exact m/z values.

    Unlike fixed 0.5 Da binning, sinusoidal encodings represent arbitrarily precise
    floating-point mass measurements, distinguishing isomers that differ by < 0.01 Da.
    """

    def __init__(self, dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        # Half frequencies for sin, half for cos
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of arbitrary shape (..., ) containing scalar values (e.g. m/z).

        Returns:
            Tensor of shape (..., dim) with continuous sinusoidal features.
        """
        # x shape: (..., 1)
        x_expanded = x.unsqueeze(-1)
        # args shape: (..., half_dim)
        args = x_expanded * self.freqs
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)
        return torch.cat([sin_part, cos_part], dim=-1)


class PeakTransformerEncoder(nn.Module):
    """Deep Transformer processing variable-length MS/MS peak lists."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        max_peaks: int = 256,
        d_latent: int = 512,
        n_fingerprint_bits: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_peaks = max_peaks
        self.d_latent = d_latent

        # 1. Peak Feature Embeddings
        # 64 dim for peak m/z sinusoidal + 64 dim for neutral loss sinusoidal + 1 dim sqrt intensity
        self.mz_sinusoidal = SinusoidalEncoding(dim=64, max_period=2000.0)
        self.loss_sinusoidal = SinusoidalEncoding(dim=64, max_period=2000.0)
        self.peak_proj = nn.Sequential(
            nn.Linear(64 + 64 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Precursor & Metadata Conditioning
        self.precursor_sinusoidal = SinusoidalEncoding(dim=64, max_period=2000.0)
        self.precursor_proj = nn.Linear(64, d_model)

        # Collision energy representation: [mean_ce, min_ce, max_ce, has_ce]
        self.ce_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Instrument type embedding
        self.instrument_emb = nn.Embedding(len(KNOWN_INSTRUMENTS), d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # 4. Latent Contrastive Projection Head (Spectrum -> 512-dim Unit Sphere)
        self.latent_head = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.LayerNorm(d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_latent),
        )

        # 5. Auxiliary Multi-Task Fingerprint Prediction Head
        self.fp_head = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.LayerNorm(d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, n_fingerprint_bits),
            nn.Sigmoid(),
        )

    def forward(
        self,
        peak_mzs: torch.Tensor,  # (B, P)
        peak_intensities: torch.Tensor,  # (B, P)
        peak_mask: torch.Tensor,  # (B, P) True where peak is valid, False for padding
        precursor_mzs: torch.Tensor,  # (B,)
        collision_energies: torch.Tensor,  # (B, 4): [mean, min, max, has_ce]
        instrument_indices: torch.Tensor,  # (B,) integer IDs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            latent_embedding: L2-normalized unit vector in R^d_latent (B, d_latent).
            predicted_fp: Auxiliary Morgan fingerprint probability vector (B, 2048).
        """
        B, P = peak_mzs.shape

        # 1. Compute peak tokens
        # Neutral loss: precursor_mz - peak_mz
        prec_expanded = precursor_mzs.unsqueeze(1).expand_as(peak_mzs)
        neutral_losses = torch.clamp(prec_expanded - peak_mzs, min=0.0)

        mz_enc = self.mz_sinusoidal(peak_mzs)  # (B, P, 64)
        loss_enc = self.loss_sinusoidal(neutral_losses)  # (B, P, 64)
        sqrt_ints = torch.sqrt(torch.clamp(peak_intensities, min=0.0)).unsqueeze(-1).float()  # (B, P, 1)

        raw_peak_feat = torch.cat([mz_enc, loss_enc, sqrt_ints], dim=-1)
        peak_tokens = self.peak_proj(raw_peak_feat)  # (B, P, d_model)

        # 2. Compute metadata-conditioned [CLS] token
        cls_base = self.cls_token.expand(B, 1, -1)  # (B, 1, d_model)
        prec_emb = self.precursor_proj(self.precursor_sinusoidal(precursor_mzs)).unsqueeze(1)
        ce_emb = self.ce_proj(collision_energies).unsqueeze(1)
        inst_emb = self.instrument_emb(instrument_indices).unsqueeze(1)

        cls_token = cls_base + prec_emb + ce_emb + inst_emb  # (B, 1, d_model)

        # 3. Concatenate [CLS] + peak tokens
        tokens = torch.cat([cls_token, peak_tokens], dim=1)  # (B, 1 + P, d_model)

        # Build attention key padding mask: (B, 1 + P)
        # In PyTorch nn.Transformer, True means key is PADDED / IGNORED.
        # [CLS] token is never padded (False).
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=peak_mask.device)
        token_padding_mask = torch.cat([cls_mask, ~peak_mask], dim=1)

        # 4. Pass through Transformer Encoder
        out = self.transformer(tokens, src_key_padding_mask=token_padding_mask)
        out = self.norm(out)

        # Extract [CLS] embedding
        cls_out = out[:, 0, :]  # (B, d_model)

        # 5. Project to Latent Space (Unit Sphere)
        latent = self.latent_head(cls_out)
        latent_norm = F.normalize(latent, p=2, dim=-1)

        # 6. Auxiliary Fingerprint prediction
        pred_fp = self.fp_head(cls_out)

        return latent_norm, pred_fp


class MoleculeEncoder(nn.Module):
    """Molecule Encoder mapping chemical structures into the shared contrastive latent space.

    Accepts molecular fingerprints (Morgan ECFP4) and projects them into
    the exact same 512-dim unit sphere as the Peak Transformer.
    """

    def __init__(
        self, in_features: int = 2048, d_feedforward: int = 512, d_latent: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_feedforward),
            nn.LayerNorm(d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_feedforward),
            nn.LayerNorm(d_feedforward),
            nn.GELU(),
            nn.Linear(d_feedforward, d_latent),
        )

    def forward(self, molecular_fingerprints: torch.Tensor) -> torch.Tensor:
        """Args:
            molecular_fingerprints: Float tensor of shape (B, in_features).

        Returns:
            L2-normalized unit vector in R^d_latent (B, d_latent).
        """
        latent = self.net(molecular_fingerprints)
        return F.normalize(latent, p=2, dim=-1)


class ContrastiveInfoNCELoss(nn.Module):
    """Symmetric InfoNCE / NT-Xent contrastive loss for spectrum-molecule alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, spectrum_latent: torch.Tensor, molecule_latent: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            spectrum_latent: (B, d_latent) unit vectors.
            molecule_latent: (B, d_latent) unit vectors.
        """
        # Cosine similarity matrix (B, B)
        sim_matrix = (
            torch.matmul(spectrum_latent, molecule_latent.T) / self.temperature
        )

        labels = torch.arange(
            sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.long
        )

        # Bidirectional cross-entropy
        loss_s2m = F.cross_entropy(sim_matrix, labels)
        loss_m2s = F.cross_entropy(sim_matrix.T, labels)
        return (loss_s2m + loss_m2s) / 2.0


# Pre-initialize Morgan fingerprint generator
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def compute_molecule_fingerprint(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Compute binary Morgan ECFP4 fingerprint array from SMILES string."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = _MORGAN_GEN.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        return arr
    except Exception:
        return None
