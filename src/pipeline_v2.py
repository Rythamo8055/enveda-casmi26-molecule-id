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
        # STAGE 2: Tier 2 Contrastive Retrieval with RRF
        # ---------------------------------------------------------
        # Find candidate molecules from COCONUT matching precursor mass
        neutral_masses = []
        for s in spectra:
            nm = calculate_neutral_mass(s["precursor_mz"], s["adduct"])
            if nm is not None and nm > 0:
                neutral_masses.append(nm)

        if neutral_masses:
            mean_nm = float(np.median(neutral_masses))
            delta = mean_nm * ppm_tol * 1e-6
            left = np.searchsorted(self.coco_masses, mean_nm - delta, side="left")
            right = np.searchsorted(
                self.coco_masses, mean_nm + delta, side="right"
            )

            cand_smiles_slice = self.coco_smiles[left:right]
            cand_keys_slice = self.coco_keys[left:right]

            # Collect candidate fingerprints
            cand_fps_list = []
            valid_cands = []
            for c_smi, c_k14 in zip(cand_smiles_slice, cand_keys_slice):
                if c_smi not in self.fp_cache:
                    c_fp = compute_molecule_fingerprint(c_smi, n_bits=2048)
                    self.fp_cache[c_smi] = (
                        c_fp
                        if c_fp is not None
                        else np.zeros(2048, dtype=np.float32)
                    )
                fp_arr = self.fp_cache[c_smi]
                if fp_arr.sum() > 0:
                    cand_fps_list.append(fp_arr)
                    valid_cands.append((c_smi, c_k14))

            if valid_cands:
                cand_fps_tensor = torch.from_numpy(
                    np.array(cand_fps_list, dtype=np.float32)
                ).to(self.device)
                with torch.no_grad():
                    mol_latents = self.mol_encoder(
                        cand_fps_tensor
                    )  # (N_cands, d_latent)

                # Track Reciprocal Rank Fusion scores across spectra
                # rrf_scores[idx] = sum_s (weight_s / (rrf_k + rank_s))
                rrf_scores = np.zeros(len(valid_cands), dtype=np.float32)

                for s in spectra:
                    mzs = s["mzs"]
                    ints = s["ints"]
                    prec_mz = s["precursor_mz"]
                    ce_feat = s["ce_feat"]
                    inst_idx = s["inst_idx"]

                    # Information entropy of spectrum to weight clean, rich spectra higher
                    if len(ints) > 0 and ints.sum() > 0:
                        norm_ints = ints / ints.sum()
                        spec_entropy = -float(
                            np.sum(norm_ints * np.log(norm_ints + 1e-12))
                        )
                        spec_weight = 1.0 + min(spec_entropy / 3.0, 1.5)
                    else:
                        spec_weight = 1.0

                    # Prepare peak tensors for Peak Transformer
                    n_p = max(len(mzs), 1)
                    padded_mzs = np.zeros((1, n_p), dtype=np.float32)
                    padded_ints = np.zeros((1, n_p), dtype=np.float32)
                    mask = np.zeros((1, n_p), dtype=bool)

                    if len(mzs) > 0:
                        padded_mzs[0, : len(mzs)] = mzs
                        padded_ints[0, : len(ints)] = ints
                        mask[0, : len(mzs)] = True

                    with torch.no_grad():
                        spec_latent, _ = self.spec_encoder(
                            torch.from_numpy(padded_mzs).to(self.device),
                            torch.from_numpy(padded_ints).to(self.device),
                            torch.from_numpy(mask).to(self.device),
                            torch.tensor([prec_mz], dtype=torch.float32).to(
                                self.device
                            ),
                            torch.from_numpy(ce_feat).unsqueeze(0).to(
                                self.device
                            ),
                            torch.tensor([inst_idx], dtype=torch.int64).to(
                                self.device
                            ),
                        )
                        # Cosine similarities: (1, d) @ (N, d).T -> (N,)
                        sims = (
                            torch.matmul(spec_latent, mol_latents.T)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )

                    # Determine ranking for this specific spectrum
                    ranking_order = np.argsort(-sims)
                    for rank, cand_idx in enumerate(ranking_order, start=1):
                        rrf_scores[cand_idx] += spec_weight / (rrf_k + rank)

                # Select best representative spectrum for in-silico fragmentation scoring
                best_spec = max(
                    spectra, key=lambda s: len(s["mzs"]) if s["mzs"] is not None else 0
                )
                rep_mzs = best_spec["mzs"]
                rep_ints = best_spec["ints"]
                rep_prec = best_spec["precursor_mz"]

                # Generate plausible molecular formulas (Seven Golden Rules)
                from src.preprocessing.formula_generator import generate_plausible_formulas
                from rdkit.Chem import rdMolDescriptors
                formulas = generate_plausible_formulas(mean_nm, ppm_tol=ppm_tol, max_candidates=10)
                formula_map = {f.formula: f.score for f in formulas}

                # Combine RRF score with expanded substructure scorer and formula plausibility
                for cand_idx, (c_smi, c_k14) in enumerate(valid_cands):
                    # Skip if already high-confidence Tier 1 match
                    if (
                        c_k14 in candidate_scores
                        and candidate_scores[c_k14][1] >= 5.0
                    ):
                        continue

                    r_score = float(rrf_scores[cand_idx])
                    frag_score = score_candidate_by_fragmentation(
                        c_smi, rep_mzs, rep_ints, precursor_mz=rep_prec
                    )

                    # Molecular formula match bonus
                    try:
                        mol = Chem.MolFromSmiles(c_smi)
                        c_form = rdMolDescriptors.CalcMolFormula(mol) if mol else ""
                    except Exception:
                        c_form = ""
                    formula_bonus = formula_map.get(c_form, 0.0)

                    # Tier 2 combined score: 50% contrastive RRF + 30% fragmentation explainer + 20% formula prior
                    tier2_score = (
                        0.50 * (r_score * 10.0)
                        + 0.30 * frag_score
                        + 0.20 * min(formula_bonus, 1.0)
                    )

                    canonical_k14 = smiles_to_inchikey14(c_smi)
                    if not canonical_k14:
                        continue

                    if (
                        canonical_k14 not in candidate_scores
                        or tier2_score > candidate_scores[canonical_k14][1]
                    ):
                        candidate_scores[canonical_k14] = (c_smi, tier2_score)

        # ---------------------------------------------------------
        # STAGE 3: Tier 3 CPU De Novo Scaffold Assembly (Class 3 Novelty)
        # ---------------------------------------------------------
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
