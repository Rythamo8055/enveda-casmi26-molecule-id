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
from src.retrieval.substructure_scorer import (
    score_candidate_by_fragmentation,
    score_candidate_multi_energy,
)
from src.retrieval.np_scorer import calculate_np_likeness
from src.retrieval.adduct_hypotheses import get_adduct_hypotheses
from src.retrieval.mass_calibration import calculate_mass_defect_score
from src.evaluation.metrics import smiles_to_inchikey14


class DeterministicCandidateRanker:
    """Ranks molecular candidates from external databases (COCONUT) using pure chemical physics rules."""

    def __init__(self, coconut_parquet_path: str = "data/external/coconut_indexed.parquet"):
        print(f"Loading COCONUT database from {coconut_parquet_path}...")
        self.coco_df = pl.read_parquet(coconut_parquet_path)
        self.coco_masses = self.coco_df["exact_mass"].to_numpy()
        self.coco_smiles = self.coco_df["clean_smiles"].to_list()
        self.coco_keys = self.coco_df["inchikey14"].to_list()
        self.coco_formulas = self.coco_df["molecular_formula"].to_list()

        # Cache calculated formulas and canonical keys
        self.formula_cache: Dict[str, str] = {}
        self.k14_cache: Dict[str, str] = {}

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
        ppm_tol: float = 10.0,
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

        cand_smiles_slice = list(self.coco_smiles[left:right])
        cand_keys_slice = list(self.coco_keys[left:right])
        cand_masses_slice = list(self.coco_masses[left:right])
        cand_forms_slice = list(self.coco_formulas[left:right])
        cand_target_masses = [consensus_nm] * len(cand_smiles_slice)

        # Multi-adduct hypothesis recovery (e.g. in-source water loss [M-H2O+H]+ or [M+Na]+)
        if len(cand_smiles_slice) < 5 and spectra:
            first_spec = spectra[0]
            hypotheses = get_adduct_hypotheses(first_spec["precursor_mz"], first_spec.get("adduct", "[M+H]+"))
            for alt_adduct, alt_nm in hypotheses[1:]:
                alt_delta = alt_nm * ppm_tol * 1e-6
                alt_l = np.searchsorted(self.coco_masses, alt_nm - alt_delta, side="left")
                alt_r = np.searchsorted(self.coco_masses, alt_nm + alt_delta, side="right")
                if alt_r > alt_l:
                    cand_smiles_slice.extend(self.coco_smiles[alt_l:alt_r])
                    cand_keys_slice.extend(self.coco_keys[alt_l:alt_r])
                    cand_masses_slice.extend(self.coco_masses[alt_l:alt_r])
                    cand_forms_slice.extend(self.coco_formulas[alt_l:alt_r])
                    cand_target_masses.extend([alt_nm] * (alt_r - alt_l))

        if len(cand_smiles_slice) == 0:
            return []

        # 4. Score each candidate across spectra using upgraded physics
        candidate_scores: Dict[str, Tuple[str, float]] = {}  # k14 -> (smiles, score)

        for c_smi, c_k14, c_mass, c_form, target_nm in zip(
            cand_smiles_slice, cand_keys_slice, cand_masses_slice, cand_forms_slice, cand_target_masses
        ):
            # Resolve canonical InChIKey14 with fast cache
            if c_smi in self.k14_cache:
                canonical_k14 = self.k14_cache[c_smi]
            else:
                canonical_k14 = smiles_to_inchikey14(c_smi) or c_k14
                self.k14_cache[c_smi] = canonical_k14

            if not canonical_k14:
                continue

            # Feature 1: Calibrated Gaussian mass accuracy score (sigma = 7.0 ppm)
            ppm_diff = abs(c_mass - target_nm) / target_nm * 1e6
            mass_score = float(np.exp(-0.5 * (ppm_diff / 7.0) ** 2))

            # Feature 2: Formula match score
            formula_bonus = formula_score_map.get(c_form, 0.0)

            # Feature 3: Multi-energy fragmentation + base peak + diagnostic neutral losses
            comb_frag, matched_peak_cnt = score_candidate_multi_energy(c_smi, spectra, mz_tolerance=0.02)

            # Feature 4: Natural Product Likeness score
            np_score = calculate_np_likeness(c_smi)

            # Feature 5: Natural product elemental mass defect score
            defect_score = calculate_mass_defect_score(c_mass)

            # Composite deterministic ranking score:
            # 55% multi-energy fragmentation + 20% formula + 12% mass accuracy + 8% NP-likeness + 5% mass defect
            composite_score = (
                0.55 * comb_frag
                + 0.20 * min(formula_bonus, 1.0)
                + 0.12 * mass_score
                + 0.08 * np_score
                + 0.05 * defect_score
            )

            # Subtle secondary peak & NP tie-breaker
            tie_breaker = 1e-4 * matched_peak_cnt + 1e-5 * np_score
            final_score = composite_score + tie_breaker

            # Deduplicate by canonical skeleton (InChIKey14), keeping highest score
            if (
                canonical_k14 not in candidate_scores
                or final_score > candidate_scores[canonical_k14][1]
            ):
                candidate_scores[canonical_k14] = (c_smi, final_score)

        # Sort descending by composite score
        ranked = sorted(
            [(smi, k14, score) for k14, (smi, score) in candidate_scores.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        return ranked[:max_cands]
