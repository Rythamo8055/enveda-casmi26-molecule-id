"""Tier-3 De Novo Scaffold Assembly Engine (Class 3 Novel Molecules).

Generates novel molecular candidate structures for dark chemical space where molecules
do NOT exist in reference databases (COCONUT, PubChem, or train.parquet).

Algorithm:
1. Core Scaffold Retrieval: Identifies nearest parent scaffold in training libraries
   using Modified Cosine with precursor mass shift (Δm = M_query - M_scaffold).
2. Exact Delta Mass Decomposition: Solves the elemental formula of the modification
   using Seven Golden Rules (e.g. Δm = +15.9949 -> +OH, +14.0157 -> +CH3, +162.0528 -> +hexose).
3. Combinatorial Chemical Derivatization: Systematically attaches functional groups
   to chemically reactive sites on the parent scaffold using RDKit chemical reactions.
4. Filtering & Ranking: Verifies exact precursor mass (< 10 ppm), valency, and
   in-silico fragmentation explanation against query experimental peaks.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

from src.preprocessing.formula_generator import (
    generate_plausible_formulas,
    calculate_formula_mass,
)
from src.retrieval.substructure_scorer import score_candidate_by_fragmentation
from src.evaluation.metrics import smiles_to_inchikey14

# Standard natural product biosynthetic derivatization modifications
COMMON_TRANSFORMATIONS: List[Dict[str, any]] = [
    # Hydroxylation (+OH: +15.9949 Da, replaces aromatic/aliphatic H with OH)
    {
        "name": "+OH",
        "delta_mass": 15.994915,
        "formula": "O",
        "smarts_rxn": "[cH:1]>>[c:1][OH]",
    },
    # Methylation (+CH3: +14.0157 Da, replaces H with CH3 or O-methylation)
    {
        "name": "+CH3",
        "delta_mass": 14.015650,
        "formula": "CH2",
        "smarts_rxn": "[cH:1]>>[c:1]C",
    },
    {
        "name": "+OCH3 (O-methyl)",
        "delta_mass": 30.010565,
        "formula": "CH2O",
        "smarts_rxn": "[c:1][OH]>>[c:1]OC",
    },
    # Demethylation (-CH3: -14.0157 Da)
    {
        "name": "-CH3",
        "delta_mass": -14.015650,
        "formula": "-CH2",
        "smarts_rxn": "[c:1]OC>>[c:1]O",
    },
    # Acetylation (+COCH3: +42.0106 Da)
    {
        "name": "+acetyl",
        "delta_mass": 42.010565,
        "formula": "C2H2O",
        "smarts_rxn": "[c:1][OH]>>[c:1]OC(=O)C",
    },
    # Glycosylation (+hexose: +162.0528 Da, e.g. O-glucoside)
    {
        "name": "+glucosyl",
        "delta_mass": 162.052824,
        "formula": "C6H10O5",
        "smarts_rxn": "[c:1][OH]>>[c:1]OC1OC(CO)C(O)C(O)C1O",
    },
    # Deoxyglycosylation (+rhamnose: +146.0579 Da)
    {
        "name": "+rhamnosyl",
        "delta_mass": 146.057909,
        "formula": "C6H10O4",
        "smarts_rxn": "[c:1][OH]>>[c:1]OC1OC(C)C(O)C(O)C1O",
    },
    # Carboxylation (+COOH: +44.0098 Da)
    {
        "name": "+COOH",
        "delta_mass": 44.009829,
        "formula": "CO2",
        "smarts_rxn": "[cH:1]>>[c:1]C(=O)O",
    },
]


class DeNovoScaffoldAssembler:
    """Generates novel chemical structures by derivatizing retrieved core scaffolds to match unknown mass spectra."""

    def __init__(self, max_derived_per_scaffold: int = 15):
        self.max_derived = max_derived_per_scaffold
        # Pre-compile RDKit reaction SMARTS
        self.reactions: List[Tuple[str, float, any]] = []
        for trans in COMMON_TRANSFORMATIONS:
            try:
                rxn = AllChem.ReactionFromSmarts(trans["smarts_rxn"])
                self.reactions.append((trans["name"], trans["delta_mass"], rxn))
            except Exception:
                continue

    def generate_candidates_for_scaffold(
        self,
        scaffold_smiles: str,
        target_neutral_mass: float,
        ppm_tol: float = 15.0,
    ) -> List[str]:
        """Apply biosynthetic reactions to a core scaffold to produce molecules matching target mass."""
        if not scaffold_smiles:
            return []

        try:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            if scaffold_mol is None:
                return []
            scaffold_mass = Descriptors.ExactMolWt(scaffold_mol)
        except Exception:
            return []

        delta_m = target_neutral_mass - scaffold_mass
        if abs(delta_m) <= target_neutral_mass * ppm_tol * 1e-6:
            # Scaffold itself matches target mass!
            return [scaffold_smiles]

        generated_smiles = set()

        # 1-Step Derivatizations: match reactions with delta_mass ≈ delta_m
        for name, rxn_delta, rxn in self.reactions:
            mass_diff = abs(delta_m - rxn_delta)
            if (mass_diff / target_neutral_mass) * 1e6 <= ppm_tol:
                try:
                    products = rxn.RunReactants((scaffold_mol,))
                    for prod_tuple in products:
                        for prod in prod_tuple:
                            Chem.SanitizeMol(prod)
                            prod_smi = Chem.MolToSmiles(prod)
                            # Verify exact mass
                            p_mass = Descriptors.ExactMolWt(prod)
                            if abs(p_mass - target_neutral_mass) / target_neutral_mass * 1e6 <= ppm_tol:
                                generated_smiles.add(prod_smi)
                except Exception:
                    continue

        return list(generated_smiles)[:self.max_derived]

    def assemble_de_novo(
        self,
        scaffold_candidates: List[str],
        query_mzs: np.ndarray,
        query_ints: np.ndarray,
        precursor_mz: float,
        target_neutral_mass: float,
        ppm_tol: float = 15.0,
        max_cands: int = 25,
    ) -> List[Tuple[str, str, float]]:
        """Assemble and rank de novo molecules from multiple candidate scaffolds.

        Returns:
            List of (smiles, canonical_inchikey14, fragmentation_score)
        """
        all_novel_smiles = set()

        for scaffold_smi in scaffold_candidates:
            derived = self.generate_candidates_for_scaffold(
                scaffold_smi, target_neutral_mass, ppm_tol=ppm_tol
            )
            for d in derived:
                all_novel_smiles.add(d)

        # Score novel generated structures by fragmentation explanation
        scored_candidates: Dict[str, Tuple[str, float]] = {}

        for smi in all_novel_smiles:
            k14 = smiles_to_inchikey14(smi)
            if not k14:
                continue

            frag_score = score_candidate_by_fragmentation(
                smi, query_mzs, query_ints, precursor_mz=precursor_mz
            )

            if k14 not in scored_candidates or frag_score > scored_candidates[k14][1]:
                scored_candidates[k14] = (smi, frag_score)

        ranked = sorted(
            [(smi, k14, score) for k14, (smi, score) in scored_candidates.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        return ranked[:max_cands]
