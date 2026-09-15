"""Zero-Training Deterministic Candidate Ranker.

Combines:
1. High-precision neutral mass de-convolution.
2. Kind & Fiehn Seven Golden Rules molecular formula generation.
3. Expanded in-silico fragmentation & 60+ diagnostic neutral loss explanation.
4. Multi-spectrum collision energy consensus.
5. Strict canonical InChIKey14 deduplication.

Requires ZERO neural network training, ZERO GPU, and produces deterministic, chemically grounded rankings.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.preprocessing.adducts import calculate_neutral_mass
from src.preprocessing.formula_generator import (
    generate_plausible_formulas,
    FormulaCandidate,
)
from src.retrieval.substructure_scorer import score_candidate_by_fragmentation
from src.retrieval.np_scorer import calculate_np_likeness
from src.evaluation.metrics import smiles_to_inchikey14


class DeterministicCandidateRanker:
    """Ranks molecular candidates from external databases (COCONUT) using pure chemical physics rules."""

    def __init__(self, coconut_parquet_path: str = "data/external/coconut_indexed.parquet"):
        print(f"Loading COCONUT database from {coconut_parquet_path}...")
        self.coco_df = pl.read_parquet(coconut_parquet_path)
        self.coco_masses = self.coco_df["exact_mass"].to_numpy()
        self.coco_smiles = self.coco_df["clean_smiles"].to_list()
        self.coco_keys = self.coco_df["inchikey14"].to_list()

        # Cache calculated formulas
        self.formula_cache: Dict[str, str] = {}

    def get_mol_formula(self, smiles: str) -> str:
        """Get or compute Hill molecular formula for SMILES."""
        if smiles in self.formula_cache:
            return self.formula_cache[smiles]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                form = rdMolDescriptors.CalcMolFormula(mol)
                self.formula_cache[smiles] = form
                return form
        except Exception:
            pass
        return ""

    def rank_candidates(
        self,
        spectra: List[Dict[str, any]],
        ppm_tol: float = 12.0,
        max_cands: int = 25,
    ) -> List[Tuple[str, str, float]]:
        """Rank candidates for a single molecule across all its spectra.

        Returns:
            List of (smiles, inchikey14, composite_score) sorted by score descending.
        """
        # 1. Determine consensus neutral mass across spectra
        neutral_masses = []
        for s in spectra:
            nm = calculate_neutral_mass(s["precursor_mz"], s["adduct"])
            if nm is not None and nm > 0:
                neutral_masses.append(nm)

        if not neutral_masses:
            return []

        consensus_nm = float(np.median(neutral_masses))

        # 2. Generate top chemically feasible molecular formulas
        formula_candidates = generate_plausible_formulas(
            neutral_mass=consensus_nm,
            ppm_tol=ppm_tol,
            max_candidates=10,
        )
        formula_score_map: Dict[str, float] = {
            f.formula: f.score for f in formula_candidates
        }

        # 3. Query candidate structures in COCONUT within mass tolerance
        delta = consensus_nm * ppm_tol * 1e-6
        left = np.searchsorted(self.coco_masses, consensus_nm - delta, side="left")
        right = np.searchsorted(self.coco_masses, consensus_nm + delta, side="right")

        cand_smiles_slice = self.coco_smiles[left:right]
        cand_keys_slice = self.coco_keys[left:right]
        cand_masses_slice = self.coco_masses[left:right]

        if len(cand_smiles_slice) == 0:
            return []

        # 4. Score each candidate across spectra
        candidate_scores: Dict[str, Tuple[str, float]] = {}  # k14 -> (smiles, score)

        for c_smi, c_k14, c_mass in zip(cand_smiles_slice, cand_keys_slice, cand_masses_slice):
            # Strict canonical InChIKey14 check
            canonical_k14 = smiles_to_inchikey14(c_smi)
            if not canonical_k14:
                continue

            # Feature 1: Exact mass accuracy score
            ppm_diff = abs(c_mass - consensus_nm) / consensus_nm * 1e6
            mass_score = float(np.exp(-0.5 * (ppm_diff / 4.0) ** 2))

            # Feature 2: Formula match score
            c_form = self.get_mol_formula(c_smi)
            formula_bonus = formula_score_map.get(c_form, 0.0)

            # Feature 3: Multi-energy fragmentation score
            # Real molecules have different diagnostic fragments across different collision energies
            frag_scores = []
            weights = []

            for s in spectra:
                mzs = s["mzs"]
                ints = s["ints"]
                prec_mz = s["precursor_mz"]
                if len(mzs) == 0 or ints.sum() <= 0:
                    continue

                f_score = score_candidate_by_fragmentation(
                    c_smi, mzs, ints, precursor_mz=prec_mz
                )
                frag_scores.append(f_score)

                # Weight by total spectral intensity / number of peaks
                weights.append(min(len(mzs) / 20.0, 1.5))

            if frag_scores:
                mean_frag_score = float(np.average(frag_scores, weights=weights))
            else:
                mean_frag_score = 0.0

            # Feature 4: Natural Product Likeness score
            np_score = calculate_np_likeness(c_smi)

            # Composite deterministic ranking score:
            # 45% fragmentation explanation + 25% formula match + 15% exact mass accuracy + 15% NP-likeness
            composite_score = (
                0.45 * mean_frag_score
                + 0.25 * min(formula_bonus, 1.0)
                + 0.15 * mass_score
                + 0.15 * np_score
            )

            # Deduplicate by canonical skeleton (InChIKey14), keeping highest score
            if (
                canonical_k14 not in candidate_scores
                or composite_score > candidate_scores[canonical_k14][1]
            ):
                candidate_scores[canonical_k14] = (c_smi, composite_score)

        # Sort descending by composite score
        ranked = sorted(
            [(smi, k14, score) for k14, (smi, score) in candidate_scores.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        return ranked[:max_cands]
