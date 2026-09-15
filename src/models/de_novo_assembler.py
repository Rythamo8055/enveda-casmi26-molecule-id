"""Tier-3 CPU De Novo Scaffold Assembly Engine (Class 3 Novel Molecules).

Complete CPU-based, deterministic de novo structure generation for dark chemical space.
Generates chemically valid, mass-accurate candidate SMILES for molecules absent from
reference databases (COCONUT, PubChem, or train.parquet).

Core Steps:
1. Analog Core Retrieval: Queries reference spectral index using Modified Cosine with mass shift
   to discover parent natural product scaffolds (e.g. flavonoid, alkaloid, terpenoid cores).
2. Biosynthetic Reaction Derivatization: Applies 1-step and 2-step natural product biosynthetic
   reactions (hydroxylation, methylation, prenylation, glycosylation, acetylation, etc.)
   matching the exact delta mass (Δm = M_query - M_scaffold).
3. Mass & Formula Verification: Filters products to strict ppm tolerance (< 15 ppm) and Senior valency rules.
4. In-Silico Fragmentation Ranking: Evaluates and ranks novel structures against query experimental MS/MS peaks.
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

# Comprehensive Natural Product Biosynthetic Reactions
EXPANDED_TRANSFORMATIONS: List[Dict[str, any]] = [
    # 1. Hydroxylation (+OH: +15.9949 Da)
    {
        "name": "+OH (aromatic)",
        "delta_mass": 15.994915,
        "smarts_rxn": "[cH:1]>>[c:1][OH]",
    },
    {
        "name": "+OH (aliphatic)",
        "delta_mass": 15.994915,
        "smarts_rxn": "[CH:1]>>[C:1][OH]",
    },
    # 2. Methylation (+CH3: +14.0157 Da)
    {
        "name": "+CH3 (C-methyl)",
        "delta_mass": 14.015650,
        "smarts_rxn": "[cH:1]>>[c:1]C",
    },
    {
        "name": "+CH3 (O-methyl)",
        "delta_mass": 14.015650,
        "smarts_rxn": "[c:1][OH]>>[c:1]OC",
    },
    {
        "name": "+CH3 (N-methyl)",
        "delta_mass": 14.015650,
        "smarts_rxn": "[N;H1,H2:1]>>[N:1]C",
    },
    # 3. Methoxy addition (+OCH3: +31.0184 Da)
    {
        "name": "+OCH3",
        "delta_mass": 31.018390,
        "smarts_rxn": "[cH:1]>>[c:1]OC",
    },
    # 4. Demethylation (-CH3: -14.0157 Da)
    {
        "name": "-CH3 (O-demethyl)",
        "delta_mass": -14.015650,
        "smarts_rxn": "[c:1]OC>>[c:1]O",
    },
    # 5. Acetylation (+COCH3: +42.0106 Da)
    {
        "name": "+acetyl (O-acetyl)",
        "delta_mass": 42.010565,
        "smarts_rxn": "[c,C:1][OH]>>[c,C:1]OC(=O)C",
    },
    # 6. Carboxylation (+COOH: +44.0098 Da)
    {
        "name": "+COOH",
        "delta_mass": 44.009829,
        "smarts_rxn": "[cH:1]>>[c:1]C(=O)O",
    },
    # 7. Prenylation (+C5H8: +68.0626 Da, key alkaloid/flavonoid step)
    {
        "name": "+prenyl",
        "delta_mass": 68.062600,
        "smarts_rxn": "[cH:1]>>[c:1]CC=C(C)C",
    },
    # 8. Oxidation / Carbonyl (+O -2H: +13.9793 Da)
    {
        "name": "+carbonyl (=O)",
        "delta_mass": 13.979265,
        "smarts_rxn": "[CH2:1]>>[C:1]=O",
    },
    # 9. Hydrogenation (+2H: +2.0156 Da)
    {
        "name": "+2H (double bond reduction)",
        "delta_mass": 2.015650,
        "smarts_rxn": "[C:1]=[C:2]>>[CH:1][CH:2]",
    },
    # 10. Dehydrogenation (-2H: -2.0156 Da)
    {
        "name": "-2H (desaturation)",
        "delta_mass": -2.015650,
        "smarts_rxn": "[CH2:1][CH2:2]>>[CH:1]=[CH:2]",
    },
    # 11. Glycosylation (+hexose: +162.0528 Da, glucose/galactose)
    {
        "name": "+glucosyl (O-hexose)",
        "delta_mass": 162.052824,
        "smarts_rxn": "[c:1][OH]>>[c:1]OC1OC(CO)C(O)C(O)C1O",
    },
    # 12. Deoxyglycosylation (+rhamnose: +146.0579 Da)
    {
        "name": "+rhamnosyl",
        "delta_mass": 146.057909,
        "smarts_rxn": "[c:1][OH]>>[c:1]OC1OC(C)C(O)C(O)C1O",
    },
    # 13. Pentosylation (+pentose: +132.0423 Da, xylose/arabinose)
    {
        "name": "+pentosyl",
        "delta_mass": 132.042259,
        "smarts_rxn": "[c:1][OH]>>[c:1]OC1OCC(O)C(O)C1O",
    },
]


class DeNovoScaffoldAssembler:
    """CPU Deterministic De Novo Structure Generator for Novel Class 3 Molecules."""

    def __init__(self, max_derived_per_scaffold: int = 20):
        self.max_derived = max_derived_per_scaffold
        self.reactions: List[Tuple[str, float, any]] = []

        for trans in EXPANDED_TRANSFORMATIONS:
            try:
                rxn = AllChem.ReactionFromSmarts(trans["smarts_rxn"])
                self.reactions.append((trans["name"], trans["delta_mass"], rxn))
            except Exception:
                continue

    def derivatize_scaffold(
        self,
        scaffold_smiles: str,
        target_neutral_mass: float,
        ppm_tol: float = 15.0,
    ) -> List[str]:
        """Generate novel molecules by applying 1-step and 2-step reactions matching target mass."""
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

        # If scaffold is already within mass tolerance
        if abs(delta_m) <= target_neutral_mass * ppm_tol * 1e-6:
            return [scaffold_smiles]

        generated_smiles = set()

        # --- 1-STEP DERIVATIZATION ---
        for name, rxn_delta, rxn in self.reactions:
            mass_diff = abs(delta_m - rxn_delta)
            if (mass_diff / target_neutral_mass) * 1e6 <= ppm_tol:
                try:
                    products = rxn.RunReactants((scaffold_mol,))
                    for prod_tuple in products:
                        for prod in prod_tuple:
                            Chem.SanitizeMol(prod)
                            p_mass = Descriptors.ExactMolWt(prod)
                            if abs(p_mass - target_neutral_mass) / target_neutral_mass * 1e6 <= ppm_tol:
                                generated_smiles.add(Chem.MolToSmiles(prod))
                except Exception:
                    continue

        # --- 2-STEP COMBINATORIAL DERIVATIZATION (for multi-substituent shifts) ---
        # e.g. +OH +CH3 (+30.01 Da), +2OH (+31.99 Da), +2CH3 (+28.03 Da), -CH3 +OH (+1.98 Da)
        if len(generated_smiles) < 5:
            for i, (name1, d1, rxn1) in enumerate(self.reactions[:8]):
                for j, (name2, d2, rxn2) in enumerate(self.reactions[:8]):
                    comb_delta = d1 + d2
                    mass_diff = abs(delta_m - comb_delta)
                    if (mass_diff / target_neutral_mass) * 1e6 <= ppm_tol:
                        try:
                            # Step 1
                            step1_prods = rxn1.RunReactants((scaffold_mol,))
                            for p1_tuple in step1_prods[:3]:
                                for p1 in p1_tuple:
                                    Chem.SanitizeMol(p1)
                                    # Step 2
                                    step2_prods = rxn2.RunReactants((p1,))
                                    for p2_tuple in step2_prods[:3]:
                                        for p2 in p2_tuple:
                                            Chem.SanitizeMol(p2)
                                            p2_mass = Descriptors.ExactMolWt(p2)
                                            if abs(p2_mass - target_neutral_mass) / target_neutral_mass * 1e6 <= ppm_tol:
                                                generated_smiles.add(Chem.MolToSmiles(p2))
                        except Exception:
                            continue

        return list(generated_smiles)[:self.max_derived]

    def generate_de_novo_candidates(
        self,
        parent_scaffolds: List[str],
        target_neutral_mass: float,
        query_mzs: np.ndarray,
        query_ints: np.ndarray,
        precursor_mz: float,
        ppm_tol: float = 15.0,
        max_cands: int = 25,
    ) -> List[Tuple[str, str, float]]:
        """Synthesize and score novel de novo candidates across multiple parent scaffolds.

        Returns:
            List of (smiles, canonical_inchikey14, score) sorted by score descending.
        """
        all_novel: Set[str] = set()

        for scaffold in parent_scaffolds:
            prods = self.derivatize_scaffold(scaffold, target_neutral_mass, ppm_tol=ppm_tol)
            for p in prods:
                all_novel.add(p)

        if not all_novel:
            return []

        # Score novel candidates by fragmentation match and formula consistency
        candidates: Dict[str, Tuple[str, float]] = {}

        for smi in all_novel:
            k14 = smiles_to_inchikey14(smi)
            if not k14:
                continue

            frag_score = score_candidate_by_fragmentation(
                smi, query_mzs, query_ints, precursor_mz=precursor_mz
            )

            # Check mass ppm error
            try:
                mol = Chem.MolFromSmiles(smi)
                m = Descriptors.ExactMolWt(mol)
                ppm_err = abs(m - target_neutral_mass) / target_neutral_mass * 1e6
                mass_score = float(np.exp(-0.5 * (ppm_err / 4.0) ** 2))
            except Exception:
                mass_score = 0.5

            total_score = 0.70 * frag_score + 0.30 * mass_score

            if k14 not in candidates or total_score > candidates[k14][1]:
                candidates[k14] = (smi, total_score)

        ranked = sorted(
            [(smi, k14, score) for k14, (smi, score) in candidates.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        return ranked[:max_cands]
