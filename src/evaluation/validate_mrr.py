"""Honest, rigorous MRR@25 validation on structure-disjoint holdouts."""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from src.evaluation.metrics import smiles_to_inchikey14, calculate_mrr_at_k
from src.preprocessing.adducts import calculate_neutral_mass
from src.models.peak_transformer import (
    PeakTransformerEncoder,
    MoleculeEncoder,
    compute_molecule_fingerprint,
    get_instrument_idx,
)
from src.models.spectrum_dataset import parse_collision_energy


def evaluate_mrr_on_holdout(
    spec_encoder: PeakTransformerEncoder,
    mol_encoder: MoleculeEncoder,
    val_df: pl.DataFrame,
    coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
    device: str = "cpu",
    ppm_tol: float = 15.0,
    max_cands: int = 25,
    sample_limit: Optional[int] = None,
) -> Dict[str, float]:
    """Run full candidate retrieval on holdout spectra against COCONUT and compute honest MRR@25."""
    print("=" * 70)
    print("STARTING RIGOROUS MRR@25 EVALUATION")
    print("=" * 70)
    t0 = time.time()

    spec_encoder.to(device).eval()
    mol_encoder.to(device).eval()

    # Load COCONUT natural products database
    print(f"Loading COCONUT index from {coconut_parquet_path}...")
    coco_df = pl.read_parquet(coconut_parquet_path)
    coco_masses = coco_df["exact_mass"].to_numpy()
    coco_smiles = coco_df["clean_smiles"].to_list()
    coco_keys = coco_df["inchikey14"].to_list()
    print(f"Loaded {len(coco_masses)} COCONUT candidate molecules.")

    # Pre-compute or cache candidate fingerprints as needed
    fp_cache: Dict[str, np.ndarray] = {}

    if sample_limit is not None and len(val_df) > sample_limit:
        val_df = val_df.sample(n=sample_limit, seed=42)
        print(f"Subsampled validation set to {len(val_df)} spectra for evaluation.")

    reciprocal_ranks: List[float] = []
    hits_at_1: List[float] = []
    hits_at_5: List[float] = []
    hits_at_10: List[float] = []
    hits_at_25: List[float] = []
    candidate_counts: List[int] = []

    print(f"Evaluating {len(val_df)} holdout spectra...")

    for row_idx, row in enumerate(val_df.iter_rows(named=True)):
        true_smi = row["normalized_smiles"]
        true_k14 = row["inchikey14"]
        if not true_k14:
            true_k14 = smiles_to_inchikey14(true_smi)
        if not true_k14:
            continue

        prec_mz = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        neutral_mass = calculate_neutral_mass(prec_mz, adduct)
        if neutral_mass is None or neutral_mass <= 0:
            neutral_mass = prec_mz  # fallback

        # 1. Search COCONUT within mass tolerance
        delta = neutral_mass * ppm_tol * 1e-6
        left = np.searchsorted(coco_masses, neutral_mass - delta, side="left")
        right = np.searchsorted(coco_masses, neutral_mass + delta, side="right")

        num_cands = right - left
        candidate_counts.append(num_cands)

        if num_cands == 0:
            reciprocal_ranks.append(0.0)
            hits_at_1.append(0.0)
            hits_at_5.append(0.0)
            hits_at_10.append(0.0)
            hits_at_25.append(0.0)
            continue

        # 2. Extract and embed candidate fingerprints
        cand_smiles_slice = coco_smiles[left:right]
        cand_keys_slice = coco_keys[left:right]

        cand_fps_list = []
        valid_cands = []

        for c_smi, c_k14 in zip(cand_smiles_slice, cand_keys_slice):
            if c_smi not in fp_cache:
                c_fp = compute_molecule_fingerprint(c_smi, n_bits=2048)
                fp_cache[c_smi] = c_fp if c_fp is not None else np.zeros(2048, dtype=np.float32)
            fp_arr = fp_cache[c_smi]
            if fp_arr.sum() > 0:
                cand_fps_list.append(fp_arr)
                valid_cands.append((c_smi, c_k14))

        if not valid_cands:
            reciprocal_ranks.append(0.0)
            hits_at_1.append(0.0)
            hits_at_5.append(0.0)
            hits_at_10.append(0.0)
            hits_at_25.append(0.0)
            continue

        cand_fps_tensor = torch.from_numpy(np.array(cand_fps_list, dtype=np.float32)).to(device)

        # 3. Embed spectrum with PeakTransformer
        mzs = np.array(row["ms2_mzs"] if row["ms2_mzs"] is not None else [], dtype=np.float32)
        ints = np.array(row["ms2_normalized_intensities"] if row["ms2_normalized_intensities"] is not None else [], dtype=np.float32)

        if len(mzs) > 128:
            top_idx = np.argpartition(ints, -128)[-128:]
            order = np.argsort(mzs[top_idx])
            mzs = mzs[top_idx][order]
            ints = ints[top_idx][order]
        elif len(mzs) > 0:
            order = np.argsort(mzs)
            mzs = mzs[order]
            ints = ints[order]

        n_p = max(len(mzs), 1)
        padded_mzs = np.zeros((1, n_p), dtype=np.float32)
        padded_ints = np.zeros((1, n_p), dtype=np.float32)
        mask = np.zeros((1, n_p), dtype=bool)

        if len(mzs) > 0:
            padded_mzs[0, :len(mzs)] = mzs
            padded_ints[0, :len(ints)] = ints
            mask[0, :len(mzs)] = True

        ce_feat = np.array([parse_collision_energy(row.get("collision_energy_ev"))], dtype=np.float32)
        inst_idx = np.array([get_instrument_idx(row.get("instrument_type"))], dtype=np.int64)

        with torch.no_grad():
            spec_latent, _ = spec_encoder(
                torch.from_numpy(padded_mzs).to(device),
                torch.from_numpy(padded_ints).to(device),
                torch.from_numpy(mask).to(device),
                torch.tensor([prec_mz], dtype=torch.float32).to(device),
                torch.from_numpy(ce_feat).to(device),
                torch.from_numpy(inst_idx).to(device),
            )
            mol_latents = mol_encoder(cand_fps_tensor)

            # Cosine similarity between spectrum and all candidates
            # (1, d) @ (N, d).T -> (1, N)
            scores = torch.matmul(spec_latent, mol_latents.T).squeeze(0).cpu().numpy()

        # 4. Rank candidates descending by score
        order = np.argsort(-scores)
        seen_keys = set()
        hit_rank = None
        current_rank = 1

        for idx in order:
            c_smi, c_k14 = valid_cands[idx]
            # Strict canonical InChIKey14 check
            if not c_k14:
                c_k14 = smiles_to_inchikey14(c_smi)
            if not c_k14 or c_k14 in seen_keys:
                continue
            seen_keys.add(c_k14)

            if c_k14 == true_k14:
                hit_rank = current_rank
                break

            current_rank += 1
            if current_rank > max_cands:
                break

        rr = 1.0 / hit_rank if (hit_rank is not None and hit_rank <= max_cands) else 0.0
        reciprocal_ranks.append(rr)
        hits_at_1.append(1.0 if hit_rank == 1 else 0.0)
        hits_at_5.append(1.0 if (hit_rank is not None and hit_rank <= 5) else 0.0)
        hits_at_10.append(1.0 if (hit_rank is not None and hit_rank <= 10) else 0.0)
        hits_at_25.append(1.0 if (hit_rank is not None and hit_rank <= 25) else 0.0)

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == len(val_df):
            curr_mrr = np.mean(reciprocal_ranks)
            print(f"[{row_idx+1}/{len(val_df)}] Current MRR@25: {curr_mrr:.4f} | Hit@1: {np.mean(hits_at_1)*100:.1f}% | Hit@25: {np.mean(hits_at_25)*100:.1f}%")

    mean_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    hit1 = float(np.mean(hits_at_1)) if hits_at_1 else 0.0
    hit5 = float(np.mean(hits_at_5)) if hits_at_5 else 0.0
    hit10 = float(np.mean(hits_at_10)) if hits_at_10 else 0.0
    hit25 = float(np.mean(hits_at_25)) if hits_at_25 else 0.0
    mean_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0

    print("\n" + "=" * 70)
    print(f"FINAL BENCHMARK EVALUATION RESULTS ({len(reciprocal_ranks)} Spectra):")
    print(f"  MRR@25:                 {mean_mrr:.4f}")
    print(f"  Hit@1 Rate:             {hit1*100:.2f}%")
    print(f"  Hit@5 Rate:             {hit5*100:.2f}%")
    print(f"  Hit@10 Rate:            {hit10*100:.2f}%")
    print(f"  Hit@25 Rate:            {hit25*100:.2f}%")
    print(f"  Mean Mass Candidates:   {mean_cands:.1f}")
    print(f"  Evaluation Time:        {time.time()-t0:.1f}s")
    print("=" * 70 + "\n")

    return {
        "mrr_at_25": mean_mrr,
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "hit_at_25": hit25,
        "mean_candidates": mean_cands,
    }
