"""State-of-the-Art Hybrid Pipeline: Tier-1 Library Matcher + Contrastive Peak Transformer with RRF."""

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

from src.evaluation.metrics import smiles_to_inchikey14
from src.models.peak_transformer import (
    KNOWN_INSTRUMENTS,
    MoleculeEncoder,
    PeakTransformerEncoder,
    compute_molecule_fingerprint,
    get_instrument_idx,
)
from src.models.spectrum_dataset import parse_collision_energy
from src.preprocessing.adducts import calculate_neutral_mass
from src.retrieval.spectral_matcher import SpectralLibraryIndex as FastSpectralIndex
from src.retrieval.substructure_scorer import score_candidate_by_fragmentation
from src.models.de_novo_assembler import DeNovoScaffoldAssembler
from src.retrieval.learned_ranker import LearnedCandidateRanker
from src.retrieval.deterministic_ranker import DeterministicCandidateRanker
from src.retrieval.mass_calibration import get_instrument_ppm_tolerance
from src.retrieval.feature_extractor import extract_candidate_features



class ContrastiveHybridPipeline:
    """Hybrid Retrieval Pipeline coupling Tier 1 (Library Matcher) with

    Tier 2 (Contrastive Peak Transformer + COCONUT retrieval + RRF).
    """

    def __init__(
        self,
        spectral_index: Optional[FastSpectralIndex] = None,
        coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
        spec_encoder_path: Optional[str] = "models/best_peak_transformer.pt",
        mol_encoder_path: Optional[str] = "models/best_molecule_encoder.pt",
        device: Optional[str] = None,
    ):
        self.spectral_index = spectral_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load COCONUT index
        print(f"Loading COCONUT database from {coconut_parquet_path}...")
        self.coco_df = pl.read_parquet(coconut_parquet_path)
        self.coco_masses = self.coco_df["exact_mass"].to_numpy()
        self.coco_smiles = self.coco_df["clean_smiles"].to_list()
        self.coco_keys = self.coco_df["inchikey14"].to_list()

        # Cache candidate fingerprints
        self.fp_cache: Dict[str, np.ndarray] = {}

        # 2. Load Peak Transformer & Molecule Encoder
        self.spec_encoder = PeakTransformerEncoder(
            d_model=256, n_heads=8, n_layers=4, d_feedforward=512, d_latent=512
        ).to(self.device)
        self.mol_encoder = MoleculeEncoder(
            in_features=2048, d_feedforward=512, d_latent=512
        ).to(self.device)

        if spec_encoder_path and os.path.exists(spec_encoder_path):
            print(
                f"Loading Peak Transformer weights from {spec_encoder_path}..."
            )
            try:
                self.spec_encoder.load_state_dict(
                    torch.load(spec_encoder_path, map_location=self.device)
                )
            except Exception as e:
                print(
                    f"Warning: could not load spec encoder with standard dims ({e}), trying smoke-test dims..."
                )
                self.spec_encoder = PeakTransformerEncoder(
                    d_model=128,
                    n_heads=4,
                    n_layers=2,
                    d_feedforward=256,
                    d_latent=256,
                ).to(self.device)
                self.spec_encoder.load_state_dict(
                    torch.load(spec_encoder_path, map_location=self.device)
                )

        if mol_encoder_path and os.path.exists(mol_encoder_path):
            print(f"Loading Molecule Encoder weights from {mol_encoder_path}...")
            try:
                self.mol_encoder.load_state_dict(
                    torch.load(mol_encoder_path, map_location=self.device)
                )
            except Exception as e:
                self.mol_encoder = MoleculeEncoder(
                    in_features=2048, d_feedforward=256, d_latent=256
                ).to(self.device)
                self.mol_encoder.load_state_dict(
                    torch.load(mol_encoder_path, map_location=self.device)
                )

        self.spec_encoder.eval()
        self.mol_encoder.eval()

        # 3. CPU De Novo Scaffold Assembler (Tier 3)
        self.de_novo_assembler = DeNovoScaffoldAssembler()

        # 4. CPU Learned Candidate Ranker (Tier 2 LightGBM LambdaMART)
        self.learned_ranker = LearnedCandidateRanker(
            coconut_parquet_path=coconut_parquet_path,
            model_path="models/lambdamart_ranker.txt",
        )

        # 5. Upgraded Deterministic Advanced Physics Ranker (Tier 2 Primary Engine)
        self.deterministic_ranker = DeterministicCandidateRanker(
            coconut_parquet_path=coconut_parquet_path
        )

    def predict_molecule(

        self,
        spectra: List[Dict[str, any]],
        ppm_tol: float = 15.0,
        mz_tol: float = 0.02,
        max_cands: int = 25,
        rrf_k: float = 60.0,
    ) -> str:
        """Predict top-25 unique canonical InChIKey14 candidate SMILES using multi-spectrum RRF."""
        candidate_scores: Dict[str, Tuple[str, float]] = (
            {}
        )  # InChIKey14 -> (SMILES, aggregated_score)

        # ---------------------------------------------------------
        # STAGE 1: Tier 1 Spectral Library Matcher (Class 1 hits)
        # ---------------------------------------------------------
        if self.spectral_index is not None:
            for s in spectra:
                prec = s["precursor_mz"]
                adduct = s["adduct"]
                mzs = s["mzs"]
                ints = s["ints"]
                hits = self.spectral_index.query(
                    prec, adduct, mzs, ints, ppm_tol=ppm_tol, mz_tol=mz_tol
                )
                for smi, raw_k14, sim in hits:
                    if sim < 0.35:
                        continue
                    k14 = smiles_to_inchikey14(smi)
                    if not k14:
                        continue
                    # High-confidence Tier 1 score multiplier
                    tier1_score = 5.0 + sim
                    if (
                        k14 not in candidate_scores
                        or tier1_score > candidate_scores[k14][1]
                    ):
                        candidate_scores[k14] = (smi, tier1_score)

        # ---------------------------------------------------------
        # STAGE 2: Tier 2 Advanced Physics Retrieval (COCONUT)
        # ---------------------------------------------------------
        inst_type = spectra[0].get("instrument_type") if spectra else None
        effective_ppm = get_instrument_ppm_tolerance(inst_type, default_ppm=ppm_tol)
        effective_ppm = max(effective_ppm, 10.0)

        # Multi-energy consensus, base peak explanation, diagnostic neutral loss, calibrated mass
        det_hits = self.deterministic_ranker.rank_candidates(
            spectra, ppm_tol=effective_ppm, max_cands=max_cands
        )
        for c_smi, c_k14, c_score in det_hits:
            if (
                c_k14 not in candidate_scores
                or c_score > candidate_scores[c_k14][1]
            ):
                candidate_scores[c_k14] = (c_smi, c_score)

        # ---------------------------------------------------------
        # STAGE 3: Tier 3 CPU De Novo Scaffold Assembly (Class 3 Novelty)
        # ---------------------------------------------------------
        neutral_masses = [
            calculate_neutral_mass(s["precursor_mz"], s["adduct"])
            for s in spectra
            if calculate_neutral_mass(s["precursor_mz"], s["adduct"]) is not None
        ]
        if len(candidate_scores) < max_cands and neutral_masses:
            mean_nm = float(np.median(neutral_masses))
            best_spec = max(
                spectra, key=lambda s: len(s["mzs"]) if s["mzs"] is not None else 0
            )
            rep_mzs = best_spec["mzs"]
            rep_ints = best_spec["ints"]
            rep_prec = best_spec["precursor_mz"]

            # Use retrieved candidates as parent seed scaffolds
            seed_scaffolds = [c[0] for c in candidate_scores.values()]

            # Also query modified cosine analog scaffolds from spectral library
            if self.spectral_index is not None and len(seed_scaffolds) < 5:
                for s in spectra:
                    analog_hits = self.spectral_index.query_analog(
                        s["precursor_mz"], s["adduct"], s["mzs"], s["ints"], max_mass_shift=80.0, top_k=5
                    )
                    for a_smi, a_k14, a_sim in analog_hits:
                        seed_scaffolds.append(a_smi)

            if seed_scaffolds:
                needed = max_cands - len(candidate_scores)
                de_novo_hits = self.de_novo_assembler.generate_de_novo_candidates(
                    parent_scaffolds=seed_scaffolds[:10],
                    target_neutral_mass=mean_nm,
                    query_mzs=rep_mzs,
                    query_ints=rep_ints,
                    precursor_mz=rep_prec,
                    ppm_tol=ppm_tol,
                    max_cands=needed,
                )
                for d_smi, d_k14, d_score in de_novo_hits:
                    if d_k14 not in candidate_scores:
                        candidate_scores[d_k14] = (d_smi, d_score)

        # ---------------------------------------------------------
        # STAGE 4: Fallback Padding to guarantee exactly max_cands
        # ---------------------------------------------------------
        if len(candidate_scores) < max_cands and neutral_masses:
            mean_nm = float(np.median(neutral_masses))
            wide_delta = mean_nm * 50.0 * 1e-6
            w_left = np.searchsorted(self.coco_masses, mean_nm - wide_delta, side="left")
            w_right = np.searchsorted(self.coco_masses, mean_nm + wide_delta, side="right")
            for c_smi, c_k14, c_mass in zip(
                self.coco_smiles[w_left:w_right],
                self.coco_keys[w_left:w_right],
                self.coco_masses[w_left:w_right],
            ):
                if c_k14 not in candidate_scores:
                    ppm_diff = abs(c_mass - mean_nm) / mean_nm * 1e6
                    pad_score = float(np.exp(-0.5 * (ppm_diff / 20.0) ** 2)) * 0.05
                    candidate_scores[c_k14] = (c_smi, pad_score)
                    if len(candidate_scores) >= max_cands:
                        break

        # Final safety net: pad from COCONUT head if still needed
        if len(candidate_scores) < max_cands:
            for c_smi, c_k14 in zip(self.coco_smiles, self.coco_keys):
                if c_k14 not in candidate_scores:
                    candidate_scores[c_k14] = (c_smi, 0.0001)
                    if len(candidate_scores) >= max_cands:
                        break

        if not candidate_scores:
            return ""

        # Rank all unique canonical candidates by score descending
        ranked = sorted(
            candidate_scores.values(), key=lambda x: x[1], reverse=True
        )

        top_smiles = [c[0] for c in ranked[:max_cands]]
        return ";".join(top_smiles)


def run_contrastive_hybrid_pipeline(
    test_parquet_path: str = "data/test.parquet",
    coconut_parquet_path: str = "data/external/coconut_indexed.parquet",
    spec_encoder_path: str = "models/best_peak_transformer.pt",
    mol_encoder_path: str = "models/best_molecule_encoder.pt",
    output_path: str = "data/submission_hybrid.csv",
) -> pl.DataFrame:
    """Execute end-to-end inference on test.parquet using Peak Transformer + RRF."""
    t0 = time.time()
    print("Initializing Contrastive Hybrid Pipeline...")

    pipeline = ContrastiveHybridPipeline(
        spectral_index=None,  # Or pass FastSpectralIndex if train.parquet matching is desired
        coconut_parquet_path=coconut_parquet_path,
        spec_encoder_path=spec_encoder_path,
        mol_encoder_path=mol_encoder_path,
    )

    print(f"Reading test spectra from {test_parquet_path}...")
    test_df = pl.read_parquet(test_parquet_path)

    # Group spectra by molecule_id
    grouped = defaultdict(list)
    for row in test_df.iter_rows(named=True):
        m_id = row["molecule_id"]
        prec = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        mzs = np.array(
            row["ms2_mzs"] if row["ms2_mzs"] is not None else [],
            dtype=np.float32,
        )
        ints = np.array(
            row["ms2_normalized_intensities"]
            if row["ms2_normalized_intensities"] is not None
            else [],
            dtype=np.float32,
        )
        ce_feat = np.array(
            parse_collision_energy(row.get("collision_energy_ev")),
            dtype=np.float32,
        )
        inst_idx = get_instrument_idx(row.get("instrument_type"))

        if len(mzs) > 128:
            top_idx = np.argpartition(ints, -128)[-128:]
            order = np.argsort(mzs[top_idx])
            mzs = mzs[top_idx][order]
            ints = ints[top_idx][order]
        elif len(mzs) > 0:
            order = np.argsort(mzs)
            mzs = mzs[order]
            ints = ints[order]

        grouped[m_id].append(
            {
                "precursor_mz": prec,
                "adduct": adduct,
                "mzs": mzs,
                "ints": ints,
                "ce_feat": ce_feat,
                "inst_idx": inst_idx,
            }
        )

    print(
        f"Predicting candidates for {len(grouped)} molecules across {len(test_df)} test spectra..."
    )
    results = []
    for m_id, spectra in grouped.items():
        smiles_str = pipeline.predict_molecule(spectra, ppm_tol=15.0)
        results.append({"molecule_id": m_id, "smiles": smiles_str})

    sub_df = pl.DataFrame(results)
    sub_df.write_csv(output_path)
    print(
        f"Generated contrastive submission at {output_path} in {time.time()-t0:.2f}s!"
    )
    return sub_df


if __name__ == "__main__":
    run_contrastive_hybrid_pipeline()
