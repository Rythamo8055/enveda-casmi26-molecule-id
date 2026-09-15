"""Tier-1 + Tier-2 Hybrid Pipeline: Library Matcher + Deep Neural Retrieval on COCONUT."""

import os
import time
import numpy as np
import polars as pl
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.preprocessing.adducts import calculate_neutral_mass
from src.retrieval.spectral_matcher import SpectralLibraryIndex as FastSpectralIndex
from src.models.fingerprint_net import (
    SpectrumFingerprintNet,
    featurize_spectrum,
    smiles_to_morgan_fingerprint,
    continuous_tanimoto_similarity,
)
from src.retrieval.substructure_scorer import score_candidate_by_fragmentation
from src.evaluation.metrics import smiles_to_inchikey14


class HybridRetrievalPipeline:
    def __init__(
        self,
        spectral_index: FastSpectralIndex,
        coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
        model_weights_path: str = "models/fingerprint_net.pt",
    ):
        self.spectral_index = spectral_index

        # Load COCONUT natural products database
        print(f"Loading COCONUT database from {coconut_parquet_path}...")
        self.coco_df = pl.read_parquet(coconut_parquet_path)
        self.coco_masses = self.coco_df["exact_mass"].to_numpy()
        self.coco_smiles = self.coco_df["clean_smiles"].to_list()
        self.coco_keys = self.coco_df["inchikey14"].to_list()

        # Load SpectrumFingerprintNet
        self.model = SpectrumFingerprintNet(in_features=4001, out_features=2048, hidden_dim=1024)
        if os.path.exists(model_weights_path):
            print(f"Loading neural fingerprint predictor from {model_weights_path}...")
            self.model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
        self.model.eval()

    def predict_molecule(
        self,
        spectra: List[Tuple[float, str, np.ndarray, np.ndarray]],
        ppm_tol: float = 15.0,
        mz_tol: float = 0.02,
        max_cands: int = 25,
    ) -> str:
        """Predict top 25 candidate SMILES for a single test molecule."""
        candidate_scores: Dict[str, Tuple[str, float]] = {}  # InChIKey14 -> (SMILES, score)

        # 1. Tier 1: Library Search against known spectra (train.parquet)
        for prec, adduct, mzs, ints in spectra:
            hits = self.spectral_index.query(prec, adduct, mzs, ints, ppm_tol=ppm_tol, mz_tol=mz_tol)
            for smi, raw_k14, sim in hits:
                if sim < 0.30:
                    continue  # Ignore noisy library hits
                k14 = smiles_to_inchikey14(smi)
                if not k14:
                    continue
                # Give library hits a confidence multiplier
                tier1_score = sim * 1.5
                if k14 not in candidate_scores or tier1_score > candidate_scores[k14][1]:
                    candidate_scores[k14] = (smi, tier1_score)

        # 2. Tier 2: Neural Retrieval against COCONUT Natural Products Database
        # Compute mean precursor neutral mass and average predicted fingerprint across spectra
        neutral_masses = []
        pred_fps = []

        for prec, adduct, mzs, ints in spectra:
            nm = calculate_neutral_mass(prec, adduct)
            if nm is not None and nm > 0:
                neutral_masses.append(nm)

            feat = featurize_spectrum(mzs, ints, precursor_mz=prec)
            with torch.no_grad():
                pred = self.model(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).numpy()
            pred_fps.append(pred)

        if neutral_masses and pred_fps:
            mean_nm = float(np.mean(neutral_masses))
            mean_pred_fp = np.mean(pred_fps, axis=0)

            # Query COCONUT candidates within mass tolerance
            delta = mean_nm * ppm_tol * 1e-6
            left = np.searchsorted(self.coco_masses, mean_nm - delta, side="left")
            right = np.searchsorted(self.coco_masses, mean_nm + delta, side="right")

            # Representative spectrum for in-silico fragment explanation
            rep_prec, rep_add, rep_mzs, rep_ints = spectra[0]

            for idx in range(left, right):
                c_smi = self.coco_smiles[idx]
                c_k14 = self.coco_keys[idx]

                # Check if already captured with high confidence by Tier 1
                if c_k14 in candidate_scores and candidate_scores[c_k14][1] >= 1.0:
                    continue

                c_fp = smiles_to_morgan_fingerprint(c_smi, n_bits=2048, radius=2)
                if c_fp is None:
                    continue

                tanimoto = continuous_tanimoto_similarity(mean_pred_fp, c_fp)
                frag_score = score_candidate_by_fragmentation(
                    c_smi, rep_mzs, rep_ints, precursor_mz=rep_prec
                )

                # Combined Tier 2 score
                tier2_score = 0.7 * tanimoto + 0.3 * frag_score

                # Strictly canonicalize InChIKey14 using RDKit canonical tautomer
                canonical_k14 = smiles_to_inchikey14(c_smi)
                if not canonical_k14:
                    continue

                if canonical_k14 not in candidate_scores or tier2_score > candidate_scores[canonical_k14][1]:
                    candidate_scores[canonical_k14] = (c_smi, tier2_score)

        if not candidate_scores:
            return ""

        # Rank all candidates by score descending
        ranked = sorted(candidate_scores.values(), key=lambda x: x[1], reverse=True)
        top_smiles = [c[0] for c in ranked[:max_cands]]
        return ";".join(top_smiles)


def run_hybrid_pipeline(
    test_parquet_path: str = "data/test.parquet",
    train_parquet_path: str = "data/train.parquet",
    coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
    model_weights_path: str = "models/fingerprint_net.pt",
    output_path: str = "data/submission_hybrid.csv",
) -> pl.DataFrame:
    """Execute the full Tier-1 + Tier-2 hybrid pipeline and produce competition submission."""
    t0 = time.time()
    from src.preprocessing.adducts import calculate_neutral_mass, ADDUCT_OFFSETS
    from src.retrieval.spectral_matcher import SpectralLibraryIndex as FastSpectralIndex
    import pyarrow.parquet as pq
    import pyarrow as pa

    print("Building Tier-1 FastSpectralIndex...")
    test_df = pl.read_parquet(test_parquet_path)

    # Build test neutral mass intervals (25 ppm window)
    test_intervals = []
    for row in test_df.select(["precursor_mz", "adduct"]).iter_rows(named=True):
        nm = calculate_neutral_mass(row["precursor_mz"], row["adduct"])
        if nm is not None and nm > 0:
            delta = nm * 25e-6
            test_intervals.append((nm - delta, nm + delta))

    test_intervals.sort()
    merged = []
    for start, end in test_intervals:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    starts = np.array([m[0] for m in merged], dtype=np.float64)
    ends = np.array([m[1] for m in merged], dtype=np.float64)

    def in_intervals_local(masses: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(ends, masses, side="left")
        valid = idx < len(starts)
        mask = np.zeros(len(masses), dtype=bool)
        mask[valid] = masses[valid] >= starts[idx[valid]]
        return mask

    pq_file = pq.ParquetFile(train_parquet_path)
    matching_tables = []

    for i in range(pq_file.num_row_groups):
        meta_table = pq_file.read_row_group(i, columns=["precursor_mz", "adduct"])
        precursors = meta_table["precursor_mz"].to_numpy()
        adducts = meta_table["adduct"].to_pylist()

        neutral_m = np.zeros(len(precursors), dtype=np.float64)
        has_adduct = np.zeros(len(precursors), dtype=bool)
        for j, (p, a) in enumerate(zip(precursors, adducts)):
            off = ADDUCT_OFFSETS.get(a)
            if off is not None:
                neutral_m[j] = p - off
                has_adduct[j] = True

        hit_mask = has_adduct & in_intervals_local(neutral_m)
        hit_indices = np.where(hit_mask)[0]

        if len(hit_indices) > 0:
            full_rg = pq_file.read_row_group(
                i,
                columns=[
                    "normalized_smiles",
                    "inchikey14",
                    "precursor_mz",
                    "adduct",
                    "ms2_mzs",
                    "ms2_normalized_intensities",
                ],
            )
            matching_tables.append(full_rg.take(hit_indices))

    matched_arrow = pa.concat_tables(matching_tables)
    candidate_train = pl.from_arrow(matched_arrow)

    neutral_masses = [
        calculate_neutral_mass(p, a)
        for p, a in zip(candidate_train["precursor_mz"], candidate_train["adduct"])
    ]

    pruned_mzs = []
    pruned_ints = []
    for mzs_raw, ints_raw in zip(
        candidate_train["ms2_mzs"], candidate_train["ms2_normalized_intensities"]
    ):
        m = np.array(mzs_raw, dtype=np.float32)
        i = np.array(ints_raw, dtype=np.float32)
        if len(m) > 128:
            top_idx = np.argpartition(i, -128)[-128:]
            top_idx = top_idx[np.argsort(m[top_idx])]
            m = m[top_idx]
            i = i[top_idx]
        else:
            order = np.argsort(m)
            m = m[order]
            i = i[order]
        pruned_mzs.append(m)
        pruned_ints.append(i)

    spectral_index = FastSpectralIndex(
        neutral_masses=np.array(neutral_masses, dtype=np.float32),
        smiles_list=candidate_train["normalized_smiles"].to_list(),
        inchikey14_list=candidate_train["inchikey14"].to_list(),
        peak_mzs_list=pruned_mzs,
        peak_intensities_list=pruned_ints,
    )

    # Initialize Hybrid Pipeline
    pipeline = HybridRetrievalPipeline(
        spectral_index=spectral_index,
        coconut_parquet_path=coconut_parquet_path,
        model_weights_path=model_weights_path,
    )

    # Group test spectra by molecule_id
    grouped = defaultdict(list)
    for row in test_df.iter_rows(named=True):
        m_id = row["molecule_id"]
        prec = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        mzs = np.array(row["ms2_mzs"], dtype=np.float32)
        ints = np.array(row["ms2_normalized_intensities"], dtype=np.float32)
        order = np.argsort(mzs)
        grouped[m_id].append((prec, adduct, mzs[order], ints[order]))

    print(f"Executing Hybrid Prediction on {len(grouped)} molecules...")
    results = []
    for m_id, spectra in grouped.items():
        smiles_str = pipeline.predict_molecule(spectra, ppm_tol=15.0)
        results.append({"molecule_id": m_id, "smiles": smiles_str})

    sub_df = pl.DataFrame(results)
    sub_df.write_csv(output_path)
    print(f"Hybrid submission generated at {output_path} in {time.time()-t0:.2f}s!")
    return sub_df


if __name__ == "__main__":
    run_hybrid_pipeline()
