"""Train Contrastive Peak Transformer (P0-1) for Spectrum-Molecule Alignment."""

import argparse
import os
import time
from typing import Optional
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.peak_transformer import (
    PeakTransformerEncoder,
    MoleculeEncoder,
    ContrastiveInfoNCELoss,
)
from src.models.spectrum_dataset import (
    create_structure_disjoint_datasets,
    collate_spectrum_batch,
)
from src.evaluation.validate_mrr import evaluate_mrr_on_holdout


def train_contrastive_alignment(
    train_parquet_path: str = "data/train.parquet",
    val_parquet_path: Optional[str] = None,
    coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
    output_dir: str = "models",
    max_train_samples: Optional[int] = 500000,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    val_check_interval: int = 1,
    val_eval_limit: int = 100,
    smoke_test: bool = False,
):
    """Train the Peak Transformer spectrum encoder and Molecule encoder jointly with InfoNCE."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device.upper()}")

    if smoke_test:
        print(">>> RUNNING IN SMOKE TEST MODE (Fast local verification) <<<")
        max_train_samples = 150
        epochs = 2
        batch_size = 16
        val_eval_limit = 20

    # 1. Prepare Datasets
    t0 = time.time()
    train_ds, val_ds = create_structure_disjoint_datasets(
        train_parquet_path=train_parquet_path,
        val_lib="enveda-np-examples",
        max_train_samples=max_train_samples,
        max_peaks=128,
    )
    print(f"Dataset preparation complete in {time.time()-t0:.1f}s!")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_spectrum_batch,
        drop_last=True,
        num_workers=0,
    )

    # 2. Instantiate Models
    d_model = 128 if smoke_test else 256
    n_heads = 4 if smoke_test else 8
    n_layers = 2 if smoke_test else 4
    d_feedforward = 256 if smoke_test else 512
    d_latent = 256 if smoke_test else 512

    spec_encoder = PeakTransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_feedforward=d_feedforward,
        d_latent=d_latent,
        dropout=0.1,
    ).to(device)

    mol_encoder = MoleculeEncoder(
        in_features=2048,
        d_feedforward=d_feedforward,
        d_latent=d_latent,
        dropout=0.1,
    ).to(device)

    # 3. Loss Functions & Optimizer
    contrastive_criterion = ContrastiveInfoNCELoss(temperature=0.07)
    fp_criterion = nn.BCELoss()

    parameters = list(spec_encoder.parameters()) + list(mol_encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
    )

    # Build validation DataFrame for holdout evaluation
    val_df_for_eval = pl.read_parquet(
        train_parquet_path,
        columns=[
            "ingest_lib",
            "normalized_smiles",
            "inchikey14",
            "precursor_mz",
            "adduct",
            "instrument_type",
            "collision_energy_ev",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ],
    ).filter(pl.col("ingest_lib") == "enveda-np-examples")

    best_mrr = -1.0
    best_spec_path = os.path.join(output_dir, "best_peak_transformer.pt")
    best_mol_path = os.path.join(output_dir, "best_molecule_encoder.pt")

    print(f"\nBeginning training for {epochs} epochs ({len(train_loader)} batches/epoch)...")

    for epoch in range(1, epochs + 1):
        spec_encoder.train()
        mol_encoder.train()

        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_fp_loss = 0.0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            peak_mzs = batch["peak_mzs"].to(device)
            peak_ints = batch["peak_intensities"].to(device)
            peak_mask = batch["peak_mask"].to(device)
            prec_mzs = batch["precursor_mzs"].to(device)
            ces = batch["collision_energies"].to(device)
            insts = batch["instrument_indices"].to(device)
            target_fps = batch["fingerprints"].to(device)

            spec_latent, pred_fp = spec_encoder(
                peak_mzs, peak_ints, peak_mask, prec_mzs, ces, insts
            )
            mol_latent = mol_encoder(target_fps)

            # Losses
            loss_contra = contrastive_criterion(spec_latent, mol_latent)
            loss_fp = fp_criterion(pred_fp, target_fps)
            loss = loss_contra + 0.3 * loss_fp

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_contrastive_loss += loss_contra.item()
            total_fp_loss += loss_fp.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg_l = total_loss / (batch_idx + 1)
                avg_c = total_contrastive_loss / (batch_idx + 1)
                avg_f = total_fp_loss / (batch_idx + 1)
                print(
                    f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {avg_l:.4f} (InfoNCE: {avg_c:.4f}, FP: {avg_f:.4f}) | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        print(f"Epoch {epoch} finished in {time.time()-t_epoch:.1f}s.")

        # 4. Periodic Rigorous Holdout MRR@25 Evaluation
        if epoch % val_check_interval == 0:
            print(f"\n--- Running Structure-Disjoint Holdout Validation (Epoch {epoch}) ---")
            val_results = evaluate_mrr_on_holdout(
                spec_encoder=spec_encoder,
                mol_encoder=mol_encoder,
                val_df=val_df_for_eval,
                coconut_parquet_path=coconut_parquet_path,
                device=device,
                sample_limit=val_eval_limit,
            )
            val_mrr = val_results["mrr_at_25"]

            if val_mrr > best_mrr:
                best_mrr = val_mrr
                torch.save(spec_encoder.state_dict(), best_spec_path)
                torch.save(mol_encoder.state_dict(), best_mol_path)
                print(f">>> New Best Holdout MRR@25: {best_mrr:.4f}! Saved checkpoints to {output_dir}/")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Holdout MRR@25: {best_mrr:.4f}")
    print(f"Weights saved at: {best_spec_path} and {best_mol_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Peak Transformer Contrastive Alignment.")
    parser.add_argument("--train_parquet", type=str, default="data/train.parquet")
    parser.add_argument("--coconut_parquet", type=str, default="data/external/coconut_indexed.parquet")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--max_samples", type=int, default=500000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()
    train_contrastive_alignment(
        train_parquet_path=args.train_parquet,
        coconut_parquet_path=args.coconut_parquet,
        output_dir=args.output_dir,
        max_train_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
    )
