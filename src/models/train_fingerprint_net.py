"""Train the Spectrum-to-Fingerprint neural network on diverse natural product libraries."""

import os
import time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from src.models.fingerprint_net import featurize_spectrum, SpectrumFingerprintNet, smiles_to_morgan_fingerprint


class SpectrumDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def prepare_training_data(
    train_parquet_path: str,
    max_samples: int = 30000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample diverse spectra and compute (feature, fingerprint) pairs."""
    print(f"Sampling training data from {train_parquet_path}...")
    t0 = time.time()

    # Priority to natural product libraries
    np_libs = ["enveda-np-examples", "gnps", "riken", "massbank", "mona"]

    df = pl.read_parquet(
        train_parquet_path,
        columns=[
            "ingest_lib",
            "normalized_smiles",
            "precursor_mz",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ],
    )

    # Filter to natural product libraries first
    np_df = df.filter(pl.col("ingest_lib").is_in(np_libs))
    print(f"Found {len(np_df)} natural product spectra in preferred libraries.")

    if len(np_df) < max_samples:
        other_df = df.filter(~pl.col("ingest_lib").is_in(np_libs)).sample(
            n=max_samples - len(np_df), seed=42
        )
        sample_df = pl.concat([np_df, other_df])
    else:
        sample_df = np_df.sample(n=max_samples, seed=42)

    print(f"Featurizing {len(sample_df)} spectra and generating Morgan fingerprints...")
    features_list = []
    targets_list = []

    for row in sample_df.iter_rows(named=True):
        smi = row["normalized_smiles"]
        fp = smiles_to_morgan_fingerprint(smi, n_bits=2048, radius=2)
        if fp is None or fp.sum() < 3:
            continue

        prec_mz = float(row["precursor_mz"])
        mzs = np.array(row["ms2_mzs"], dtype=np.float32)
        ints = np.array(row["ms2_normalized_intensities"], dtype=np.float32)

        feat = featurize_spectrum(mzs, ints, precursor_mz=prec_mz)
        features_list.append(feat)
        targets_list.append(fp)

    features = np.array(features_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    print(f"Data preparation complete in {time.time()-t0:.2f}s! Valid dataset size: {len(features)}")
    return features, targets


def train_model(
    train_parquet_path: str = "data/train.parquet",
    output_model_path: str = "models/fingerprint_net.pt",
    max_samples: int = 30000,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    features, targets = prepare_training_data(train_parquet_path, max_samples=max_samples)

    # Split into train and validation
    val_size = int(len(features) * 0.1)
    train_feat, val_feat = features[val_size:], features[:val_size]
    train_targ, val_targ = targets[val_size:], targets[:val_size]

    train_ds = SpectrumDataset(train_feat, train_targ)
    val_ds = SpectrumDataset(val_feat, val_targ)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SpectrumFingerprintNet(in_features=4001, out_features=2048, hidden_dim=1024)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Training SpectrumFingerprintNet for {epochs} epochs on {len(train_ds)} samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                v_preds = model(val_x)
                val_loss += criterion(v_preds, val_y).item() * len(val_x)
        val_loss /= len(val_ds)

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-t0:.1f}s"
        )

    torch.save(model.state_dict(), output_model_path)
    print(f"Trained model weights saved to {output_model_path}!")


if __name__ == "__main__":
    train_model(
        train_parquet_path="data/train.parquet",
        output_model_path="models/fingerprint_net.pt",
        max_samples=25000,
        epochs=5,
        batch_size=128,
    )
