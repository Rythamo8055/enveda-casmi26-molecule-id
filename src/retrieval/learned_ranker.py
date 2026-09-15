"""LightGBM LambdaMART Learned Candidate Ranker.

Directly optimizes pairwise ranking (NDCG/MRR) across COCONUT candidates using
a 16-dimensional physico-chemical and spectral feature representation.

CPU-friendly:
- Uses n_jobs=2 and bounded memory to keep CPU usage gentle and smooth.
- Fast training (< 30 seconds on CPU).
- Deterministic fallback if model file is not found.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import lightgbm as lgb

from src.evaluation.metrics import smiles_to_inchikey14
from src.preprocessing.adducts import calculate_neutral_mass
from src.preprocessing.formula_generator import generate_plausible_formulas
from src.retrieval.feature_extractor import extract_candidate_features, FEATURE_NAMES
from src.retrieval.mass_calibration import get_instrument_ppm_tolerance


class LearnedCandidateRanker:
    """Ranks molecular candidates from COCONUT using a trained LightGBM LambdaMART model."""

    def __init__(
        self,
        coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
        model_path: str = "models/lambdamart_ranker.txt",
    ):
        print(f"Loading COCONUT database from {coconut_parquet_path}...")
        self.coco_df = pl.read_parquet(coconut_parquet_path)
        self.coco_masses = self.coco_df["exact_mass"].to_numpy()
        self.coco_smiles = self.coco_df["clean_smiles"].to_list()
        self.coco_keys = self.coco_df["inchikey14"].to_list()

        self.model_path = model_path
        self.booster: Optional[lgb.Booster] = None

        if os.path.exists(model_path):
            try:
                print(f"Loading LambdaMART ranker from {model_path}...")
                self.booster = lgb.Booster(model_file=model_path)
                print("LambdaMART ranker loaded successfully.")
            except Exception as e:
                print(f"Warning: could not load model from {model_path}: {e}")

    def train_model(
        self,
        train_df: pl.DataFrame,
        output_model_path: Optional[str] = None,
        max_queries: int = 150,
        n_jobs: int = 2,
    ) -> float:
        """Train LightGBM LambdaMART ranker on training spectra queries.

        CPU-gentle: n_jobs=2 prevents CPU pinning.
        """
        save_path = output_model_path or self.model_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        print(f"Preparing training data for LambdaMART (max_queries={max_queries}, n_jobs={n_jobs})...")
        t0 = time.time()

        # Group training dataframe by unique molecule
        unique_smiles = train_df["normalized_smiles"].unique().to_list()
        if len(unique_smiles) > max_queries:
            np.random.seed(42)
            unique_smiles = list(np.random.choice(unique_smiles, size=max_queries, replace=False))

        X_list = []
        y_list = []
        groups = []

        for q_idx, smi in enumerate(unique_smiles):
            mol_rows = train_df.filter(pl.col("normalized_smiles") == smi)
            if len(mol_rows) == 0:
                continue

            first_row = mol_rows.row(0, named=True)
            true_k14 = first_row.get("inchikey14") or smiles_to_inchikey14(smi)
            if not true_k14:
                continue

            prec_mz = float(first_row["precursor_mz"])
            adduct = str(first_row["adduct"])
            inst_type = str(first_row.get("instrument_type", ""))
            nm = calculate_neutral_mass(prec_mz, adduct)
            if nm is None or nm <= 0:
                continue

            ppm_tol = get_instrument_ppm_tolerance(inst_type, default_ppm=5.0)

            # Query candidate window in COCONUT
            delta = nm * ppm_tol * 1e-6
            left = np.searchsorted(self.coco_masses, nm - delta, side="left")
            right = np.searchsorted(self.coco_masses, nm + delta, side="right")
            if right <= left:
                continue

            # Limit candidates per query during training for speed and balance
            cand_indices = list(range(left, right))
            if len(cand_indices) > 50:
                cand_indices = cand_indices[:50]

            # Best spectrum representation
            best_row = max(
                mol_rows.iter_rows(named=True),
                key=lambda r: len(r["ms2_mzs"]) if r["ms2_mzs"] is not None else 0,
            )
            q_mzs = np.array(best_row["ms2_mzs"] if best_row["ms2_mzs"] is not None else [], dtype=np.float32)
            q_ints = np.array(best_row["ms2_normalized_intensities"] if best_row["ms2_normalized_intensities"] is not None else [], dtype=np.float32)

            formula_cands = generate_plausible_formulas(nm, ppm_tol=ppm_tol, max_candidates=5)
            formula_map = {f.formula: f.score for f in formula_cands}

            query_features = []
            query_labels = []

            for c_idx in cand_indices:
                c_smi = self.coco_smiles[c_idx]
                c_k14 = self.coco_keys[c_idx]
                c_mass = self.coco_masses[c_idx]

                # Label assignment: 3 for exact InChIKey14 match, 0 otherwise
                label = 3 if c_k14 == true_k14 else 0

                feat = extract_candidate_features(
                    candidate_smiles=c_smi,
                    candidate_mass=c_mass,
                    query_mzs=q_mzs,
                    query_intensities=q_ints,
                    precursor_mz=prec_mz,
                    consensus_neutral_mass=nm,
                    formula_score_map=formula_map,
                )
                query_features.append(feat)
                query_labels.append(label)

            if query_features and sum(query_labels) > 0:
                X_list.extend(query_features)
                y_list.extend(query_labels)
                groups.append(len(query_features))

        print(f"Data prepared in {time.time() - t0:.1f}s: {len(groups)} query groups, {len(y_list)} total candidate pairs.")

        if len(groups) < 5:
            print("Warning: Insufficient training groups to train LambdaMART ranker.")
            return 0.0

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # LightGBM Dataset
        train_data = lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_NAMES)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1, 5, 10, 25],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 5,
            "verbose": -1,
            "n_jobs": n_jobs,
            "seed": 42,
        }

        print("Training LightGBM LambdaMART ranker...")
        t_train0 = time.time()
        self.booster = lgb.train(
            params,
            train_data,
            num_boost_round=100,
        )
        train_dur = time.time() - t_train0
        print(f"LambdaMART training completed in {train_dur:.2f} seconds.")

        self.booster.save_model(save_path)
        print(f"Saved model to {save_path}.")
        return float(train_dur)

    def rank_candidates(
        self,
        spectra: List[Dict[str, any]],
        ppm_tol: Optional[float] = None,
        max_cands: int = 25,
    ) -> List[Tuple[str, str, float]]:
        """Rank COCONUT candidates for a query using the learned LambdaMART model.

        Falls back cleanly to deterministic scoring if model is absent.
        """
        # 1. Determine consensus neutral mass and instrument type
        neutral_masses = []
        inst_types = []
        for s in spectra:
            nm = calculate_neutral_mass(s["precursor_mz"], s["adduct"])
            if nm is not None and nm > 0:
                neutral_masses.append(nm)
            if s.get("instrument_type"):
                inst_types.append(s["instrument_type"])

        if not neutral_masses:
            return []

        consensus_nm = float(np.median(neutral_masses))
        inst = inst_types[0] if inst_types else None

        if ppm_tol is None:
            ppm_tol = get_instrument_ppm_tolerance(inst, default_ppm=10.0)
        # Safe tolerance floor to prevent clipping valid ions with slight instrument drift
        ppm_tol = max(ppm_tol, 10.0)

        # 2. Slice candidates from COCONUT
        delta = consensus_nm * ppm_tol * 1e-6
        left = np.searchsorted(self.coco_masses, consensus_nm - delta, side="left")
        right = np.searchsorted(self.coco_masses, consensus_nm + delta, side="right")

        cand_smiles_slice = self.coco_smiles[left:right]
        cand_keys_slice = self.coco_keys[left:right]
        cand_masses_slice = self.coco_masses[left:right]

        if len(cand_smiles_slice) == 0:
            return []

        # 3. Best representative spectrum for fragmentation
        best_spec = max(
            spectra,
            key=lambda s: len(s["mzs"]) if s.get("mzs") is not None else 0,
        )
        rep_mzs = best_spec["mzs"]
        rep_ints = best_spec["ints"]
        rep_prec = best_spec["precursor_mz"]

        # Generate plausible formulas (Seven Golden Rules)
        formula_cands = generate_plausible_formulas(consensus_nm, ppm_tol=ppm_tol, max_candidates=5)
        formula_map = {f.formula: f.score for f in formula_cands}

        # 4. Extract features for all candidates in the slice
        features_list = []
        valid_candidates = []

        for c_smi, c_k14, c_mass in zip(cand_smiles_slice, cand_keys_slice, cand_masses_slice):
            canonical_k14 = smiles_to_inchikey14(c_smi)
            if not canonical_k14:
                continue

            feat = extract_candidate_features(
                candidate_smiles=c_smi,
                candidate_mass=c_mass,
                query_mzs=rep_mzs,
                query_intensities=rep_ints,
                precursor_mz=rep_prec,
                consensus_neutral_mass=consensus_nm,
                formula_score_map=formula_map,
            )
            features_list.append(feat)
            valid_candidates.append((c_smi, canonical_k14))

        if not valid_candidates:
            return []

        # 5. Predict ranking scores
        X = np.array(features_list, dtype=np.float32)

        # Physical baseline score: 45% fragmentation + 25% formula + 15% mass + 15% NP-likeness
        phys_score = (
            0.45 * X[:, 4]
            + 0.25 * X[:, 2]
            + 0.15 * X[:, 1]
            + 0.15 * X[:, 8]
        )

        if self.booster is not None:
            learned_pred = self.booster.predict(X)
            # Safe blending: 65% physical floor + 35% learned tree ranking boost
            ptp = float(np.ptp(learned_pred))
            norm_learned = (learned_pred - np.min(learned_pred)) / max(ptp, 1e-6)
            raw_scores = 0.65 * phys_score + 0.35 * norm_learned
        else:
            raw_scores = phys_score

        # 6. Deduplicate by canonical skeleton (InChIKey14), keeping highest score
        candidate_scores: Dict[str, Tuple[str, float]] = {}
        for (c_smi, k14), score in zip(valid_candidates, raw_scores):
            score_f = float(score)
            if k14 not in candidate_scores or score_f > candidate_scores[k14][1]:
                candidate_scores[k14] = (c_smi, score_f)

        ranked = sorted(
            [(smi, k14, score) for k14, (smi, score) in candidate_scores.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        return ranked[:max_cands]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM LambdaMART ranker on CPU.")
    parser.add_argument("--train", action="store_true", help="Train the ranker")
    parser.add_argument("--max_queries", type=int, default=150, help="Max unique molecules to train on")
    parser.add_argument("--n_jobs", type=int, default=2, help="CPU threads to use (gentle)")
    parser.add_argument("--train_parquet", type=str, default="data/train.parquet")
    parser.add_argument("--output_model", type=str, default="models/lambdamart_ranker.txt")
    args = parser.parse_args()

    if args.train:
        print(f"Loading training data from {args.train_parquet}...")
        needed_cols = [
            "ingest_lib",
            "normalized_smiles",
            "inchikey14",
            "precursor_mz",
            "adduct",
            "instrument_type",
            "collision_energy_ev",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ]
        df = pl.read_parquet(args.train_parquet, columns=needed_cols)
        # Train on GNPS/Riken/MassBank/MoNA to keep holdout (enveda-np-examples) 100% blind
        np_libs = ["gnps", "riken", "massbank", "mona"]
        train_df = df.filter(pl.col("ingest_lib").is_in(np_libs))
        print(f"Filtered to {len(train_df)} training spectra from {np_libs}.")

        ranker = LearnedCandidateRanker(model_path=args.output_model)
        ranker.train_model(
            train_df=train_df,
            output_model_path=args.output_model,
            max_queries=args.max_queries,
            n_jobs=args.n_jobs,
        )

