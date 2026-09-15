"""Natural Product Likeness and Biosynthetic Prior Scorer.

Evaluates the likelihood that a candidate chemical structure is a genuine
secondary metabolite / natural product based on chemical graph topology,
sp3 hybridization fraction, stereocenter density, and diagnostic biosynthetic patterns.
"""

from typing import Dict, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# Common natural product scaffold SMARTS patterns
NP_SCAFFOLD_SMARTS = {
    # Flavonoid / chromone / coumarin core
    "chromone": Chem.MolFromSmarts("O=C1CC(=O)c2ccccc2O1"),
    "flavone": Chem.MolFromSmarts("O=C1CC(c2ccccc2)Oc2ccccc12"),
    "coumarin": Chem.MolFromSmarts("O=C1OC(=O)c2ccccc12"),
    # Terpenoid isoprene unit (branched C5)
    "isoprene": Chem.MolFromSmarts("CC(=C)C=C"),
    # Steroid / triterpene 6-6-6-5 cyclopentanoperhydrophenanthrene core
    "steroid_core": Chem.MolFromSmarts("C1CCC2C(C1)CCC3C2CCC4CCCC34"),
    # Alkaloid core nitrogen heterocycles
    "indole": Chem.MolFromSmarts("c1ccc2[nH]ccc2c1"),
    "isoquinoline": Chem.MolFromSmarts("c1cnc2ccccc2c1"),
    "piperidine": Chem.MolFromSmarts("C1CCNCC1"),
    "pyrrolidine": Chem.MolFromSmarts("C1CCNC1"),
    # Carbohydrate / pyranose / furanose sugar rings
    "pyranose": Chem.MolFromSmarts("C1OCC(O)C(O)C1O"),
    # Macrolide lactone (ring size >= 12 with ester)
    "lactone_ring": Chem.MolFromSmarts("[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~C(=O)O1"),
}

# In-memory cache to ensure zero redundant RDKit computations on repeated queries
_NP_SCORE_CACHE: Dict[str, float] = {}


def calculate_np_likeness(smiles: str) -> float:
    """Calculate a normalized [0, 1] Natural Product Likeness score for a candidate SMILES.

    Combines:
    1. Fraction of sp3 hybridized carbons (Fsp3): NPs have high Fsp3 (0.3 - 0.7); synthetics are often < 0.2.
    2. Stereochemical complexity: Chiral centers per heavy atom.
    3. Heteroatom balance (O/C and N/C ratios typical of biosynthesis).
    4. Halogen penalty: Organohalogens are rare in terrestrial NPs (except marine secondary metabolites).
    5. Biosynthetic substructure motif bonuses.
    """
    if not smiles:
        return 0.5

    if smiles in _NP_SCORE_CACHE:
        return _NP_SCORE_CACHE[smiles]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _NP_SCORE_CACHE[smiles] = 0.5
            return 0.5

        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy == 0:
            _NP_SCORE_CACHE[smiles] = 0.5
            return 0.5

        # 1. Fsp3 (Fraction of sp3 carbons)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        # Optimal Fsp3 for natural products is between 0.35 and 0.75
        fsp3_score = float(np.exp(-0.5 * ((fsp3 - 0.50) / 0.25) ** 2))

        # 2. Chiral stereocenters
        chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        chiral_density = chiral_centers / num_heavy
        # NPs typically have 0.05 to 0.25 chiral density; purely synthetic compounds often have 0
        chiral_score = float(min(chiral_density * 5.0, 1.0))

        # 3. Elemental composition ratios
        c_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        o_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        n_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        halo_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))

        c_safe = max(c_count, 1)
        oc_ratio = o_count / c_safe
        # NPs are rich in oxygen (polyketides, phenylpropanoids, sugars)
        oc_score = float(min(oc_ratio / 0.40, 1.0) if oc_ratio <= 0.8 else max(0.0, 1.0 - (oc_ratio - 0.8)))

        # Heavy halogen penalty (synthetic combinatorial libraries have many Cl/F/Br)
        halo_penalty = max(0.0, halo_count * 0.15)

        # 4. Biosynthetic motif matches
        motif_matches = 0
        for name, patt in NP_SCAFFOLD_SMARTS.items():
            if patt is not None and mol.HasSubstructMatch(patt):
                motif_matches += 1
        motif_bonus = min(motif_matches * 0.10, 0.30)

        # Composite score
        raw_score = (
            0.30 * fsp3_score
            + 0.25 * chiral_score
            + 0.25 * oc_score
            + motif_bonus
            - halo_penalty
        )

        final_score = float(np.clip(raw_score, 0.0, 1.0))
        _NP_SCORE_CACHE[smiles] = final_score
        return final_score

    except Exception:
        _NP_SCORE_CACHE[smiles] = 0.5
        return 0.5
