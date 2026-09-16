import json

nb = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text + '\n']
    })

def add_code(text):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.split('\n')]
    })

# Cell 0: Header
add_md('''# CASMI 2026: Spectrum-to-Fingerprint Transformer (GPU Training)
### Phase 2 Dense Model: Predicting 2,048-bit Morgan Fingerprints from MS2 Spectra

- **Input:** Top 128 MS2 peaks with Continuous Fourier $m/z$ Positional Encoding & $\\sqrt{I}$ normalized intensity.
- **Backbone:** 4-layer Bidirectional Peak Transformer with Multi-Head Self-Attention.
- **Output:** 2,048-dimensional logits with Focal/BCE loss against RDKit Morgan ECFP4 fingerprints.
- **Inference use:** Dense Tanimoto re-ranking of same-mass candidate isomers to resolve Class 2 queries and push MRR towards 0.30+.
''')

# Cell 1: Environment & Setup
add_code('''# Cell 1: Setup & Offline RDKit Installation
import os
import sys
import glob
import time
import math
import subprocess
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Install offline RDKit
whl_files = glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
if whl_files:
    print(f"Installing offline RDKit: {whl_files[0]}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", whl_files[0]], check=False)

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
print("RDKit ready for Morgan fingerprint generation!")
''')

# Cell 2: Data Extraction & Preprocessing
add_code('''# Cell 2: Data Loading & Fingerprint Computation
def find_file(name: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found")
    return sorted(hits, key=len)[0]

TRAIN_PATH = find_file("train.parquet")
print(f"Loading spectra from: {TRAIN_PATH}")

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smi: str) -> np.ndarray:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            return np.array(fp_gen.GetFingerprint(m), dtype=np.float32)
    except Exception:
        pass
    return None

# Stream distinct molecules from train.parquet to build high-diversity training set
pq_file = pq.ParquetFile(TRAIN_PATH)
print(f"Total row groups in train: {pq_file.num_row_groups}")

TARGET_SAMPLES = 100000
collected_peaks = []
collected_fps = []
seen_smiles = set()

t0 = time.time()
for rg_idx in range(min(45, pq_file.num_row_groups)):
    table = pq_file.read_row_group(
        rg_idx,
        columns=["normalized_smiles", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    )
    df = table.to_pandas()
    for _, row in df.iterrows():
        smi = row["normalized_smiles"]
        if not smi or smi in seen_smiles:
            continue
        fp = smiles_to_fp(smi)
        if fp is None:
            continue
        
        mzs = np.asarray(row["ms2_mzs"], dtype=np.float32)
        ints = np.asarray(row["ms2_normalized_intensities"], dtype=np.float32)
        if len(mzs) == 0 or len(ints) == 0:
            continue
        
        # Keep top 128 peaks
        if len(ints) > 128:
            top_idx = np.argpartition(ints, -128)[-128:]
            mzs = mzs[top_idx]
            ints = ints[top_idx]
            
        # sqrt intensity normalization
        w = np.sqrt(ints)
        norm = np.linalg.norm(w)
        if norm > 1e-8:
            w /= norm
            
        order = np.argsort(mzs)
        mzs = mzs[order]
        w = w[order]
        
        collected_peaks.append((mzs, w))
        collected_fps.append(fp)
        seen_smiles.add(smi)
        
        if len(collected_fps) >= TARGET_SAMPLES:
            break
    if len(collected_fps) >= TARGET_SAMPLES:
        break

print(f"Collected {len(collected_fps):,} unique molecule-spectrum pairs in {time.time() - t0:.1f}s!")
''')

# Cell 3: PyTorch Dataset & Continuous Fourier Peak Transformer
add_code('''# Cell 3: Continuous Fourier Peak Transformer Architecture
MAX_PEAKS = 128
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
FP_SIZE = 2048

class FourierPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_mz: float = 1500.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        # Frequencies spanning fine (0.01 Da) to coarse (100 Da)
        freqs = torch.exp(torch.linspace(math.log(1.0 / max_mz), math.log(100.0), half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        # mz: [B, K] -> [B, K, half_dim]
        angles = mz.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class PeakTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=4, fp_dim=2048):
        super().__init__()
        self.pos_enc = FourierPositionalEncoding(d_model)
        self.intensity_proj = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, fp_dim)
        )

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mz: [B, K], intensity: [B, K], mask: [B, K] (True for valid peaks)
        pe = self.pos_enc(mz)
        ie = self.intensity_proj(intensity.unsqueeze(-1))
        x = pe + ie
        
        # Invert mask for PyTorch transformer (True = ignored/padded)
        padding_mask = ~mask
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Weighted mean pooling using normalized intensities
        w = (intensity * mask.float()).unsqueeze(-1)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (out * w).sum(dim=1) / w_sum.squeeze(1)
        
        logits = self.head(pooled)
        return logits

class SpectrumDataset(Dataset):
    def __init__(self, peaks_list, fps_list, max_peaks=128):
        self.peaks_list = peaks_list
        self.fps_list = fps_list
        self.max_peaks = max_peaks

    def __len__(self):
        return len(self.peaks_list)

    def __getitem__(self, idx):
        mzs, ints = self.peaks_list[idx]
        fp = self.fps_list[idx]
        
        n = min(len(mzs), self.max_peaks)
        mz_pad = np.zeros(self.max_peaks, dtype=np.float32)
        int_pad = np.zeros(self.max_peaks, dtype=np.float32)
        mask = np.zeros(self.max_peaks, dtype=bool)
        
        mz_pad[:n] = mzs[:n]
        int_pad[:n] = ints[:n]
        mask[:n] = True
        
        return (
            torch.from_numpy(mz_pad),
            torch.from_numpy(int_pad),
            torch.from_numpy(mask),
            torch.from_numpy(fp)
        )
''')

# Cell 4: Training Loop with Mixed Precision
add_code('''# Cell 4: Model Training with FP16 & Cosine LR
split_idx = int(len(collected_fps) * 0.90)
train_ds = SpectrumDataset(collected_peaks[:split_idx], collected_fps[:split_idx])
val_ds = SpectrumDataset(collected_peaks[split_idx:], collected_fps[split_idx:])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

model = PeakTransformer(d_model=256, n_heads=8, num_layers=4, fp_dim=2048).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 12
total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

print(f"Starting training: {len(train_ds):,} train samples, {len(val_ds):,} val samples, {EPOCHS} epochs...")
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for mz, it, mask, target_fp in train_loader:
        mz, it, mask, target_fp = mz.to(device), it.to(device), mask.to(device), target_fp.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                logits = model(mz, it, mask)
                loss = criterion(logits, target_fp)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mz, it, mask)
            loss = criterion(logits, target_fp)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        train_loss += loss.item() * len(mz)
        
    train_loss /= len(train_ds)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_tanimoto = []
    with torch.no_grad():
        for mz, it, mask, target_fp in val_loader:
            mz, it, mask, target_fp = mz.to(device), it.to(device), mask.to(device), target_fp.to(device)
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(mz, it, mask)
                    loss = criterion(logits, target_fp)
            else:
                logits = model(mz, it, mask)
                loss = criterion(logits, target_fp)
            val_loss += loss.item() * len(mz)
            
            # Approximate Tanimoto correlation
            preds = torch.sigmoid(logits) > 0.5
            targets = target_fp > 0.5
            intersection = (preds & targets).float().sum(dim=1)
            union = (preds | targets).float().sum(dim=1).clamp(min=1.0)
            val_tanimoto.extend((intersection / union).cpu().tolist())
            
    val_loss /= len(val_ds)
    mean_tanimoto = np.mean(val_tanimoto)
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | Val Tanimoto: {mean_tanimoto:.3f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/kaggle/working/fingerprint_transformer.pt")
        print(f"  -> Saved best model checkpoint (Val Loss: {best_val_loss:.4f})")

print("Training Complete! Checkpoint saved as fingerprint_transformer.pt")
''')

# Cell 5: Evaluation & Manifest
add_code('''# Cell 5: Model Artifact Verification
assert os.path.exists("/kaggle/working/fingerprint_transformer.pt"), "Model checkpoint not found!"
size_mb = os.path.getsize("/kaggle/working/fingerprint_transformer.pt") / (1024 * 1024)
print(f"Checkpoint size: {size_mb:.2f} MB")
print("Model ready for candidate re-ranking pipeline!")
''')

with open('/home/rythamo/some/kaggle_train_fp/train_fp.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Generated /home/rythamo/some/kaggle_train_fp/train_fp.ipynb successfully!")
