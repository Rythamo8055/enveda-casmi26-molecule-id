"""In-silico substructure and fragmentation explanation scorer for candidate molecules."""

from typing import List, Set, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops

# Common neutral loss masses in LC-MS/MS (H2O, CO, CO2, NH3, CH4, O, CH3, etc.)
COMMON_NEUTRAL_LOSSES = [
    18.010565,  # -H2O
    28.006148,  # -CO
    43.989829,  # -CO2
    17.026549,  # -NH3
    16.031300,  # -CH4
    32.026179,  # -CH3OH / -2H2O
    36.021130,  # -2H2O
    46.005479,  # -HCOOH
    60.021129,  # -C2H4O2 / -acetic acid
    162.052824, # -hexose (glucose/galactose)
    146.057909, # -deoxyhexose (rhamnose)
    132.042259, # -pentose (xylose/arabinose)
]


def generate_candidate_fragment_masses(mol: Chem.Mol, max_cuts: int = 1) -> Set[float]:
    """Generate theoretical fragment monoisotopic masses by cleaving single rotatable/acyclic bonds.

    Args:
        mol: RDKit Mol object of the candidate.
        max_cuts: Maximum simultaneous single bonds to break (1 or 2).

    Returns:
        Set of unique theoretical fragment masses (positive and neutral).
    """
    if mol is None:
        return set()

    parent_mass = Descriptors.ExactMolWt(mol)
    fragment_masses = {round(parent_mass, 4)}

    # Add parent neutral losses
    for loss in COMMON_NEUTRAL_LOSSES:
        if parent_mass > loss + 10.0:
            fragment_masses.add(round(parent_mass - loss, 4))

    # Identify breakable single acyclic bonds
    breakable_bonds = []
    for bond in mol.GetBonds():
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
            # Avoid cutting terminal hydrogen bonds
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.GetDegree() > 1 and a2.GetDegree() > 1:
                breakable_bonds.append(bond.GetIdx())

    if not breakable_bonds:
        return fragment_masses

    # Perform single-bond fragmentations
    for b_idx in breakable_bonds[:25]:  # Cap at 25 cuts for speed
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [b_idx], addDummies=False)
            frags = Chem.GetMolFrags(frag_mol, asMols=True)
            for f in frags:
                m = Descriptors.ExactMolWt(f)
                if m > 20.0:
                    fragment_masses.add(round(m, 4))
                    # Also include common neutral losses from fragments
                    for loss in COMMON_NEUTRAL_LOSSES[:4]:
                        if m > loss + 10.0:
                            fragment_masses.add(round(m - loss, 4))
        except Exception:
            continue

    return fragment_masses


def score_candidate_by_fragmentation(
    candidate_smiles: str,
    query_mzs: np.ndarray,
    query_intensities: np.ndarray,
    precursor_mz: float,
    mz_tolerance: float = 0.025,
) -> float:
    """Score how well a candidate structure explains the observed MS/MS fragment peaks.

    Computes the explained intensity fraction and peak match count.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not candidate_smiles:
        return 0.0

    try:
        mol = Chem.MolFromSmiles(candidate_smiles)
        if mol is None:
            return 0.0
    except Exception:
        return 0.0

    frag_masses = generate_candidate_fragment_masses(mol)
    if not frag_masses:
        return 0.0

    # Sort theoretical masses for binary search
    sorted_frags = np.array(sorted(frag_masses), dtype=np.float64)

    total_intensity = np.sum(query_intensities)
    if total_intensity <= 0:
        return 0.0

    matched_intensity = 0.0
    matched_peaks = 0

    # Check observed peaks
    for mz, intensity in zip(query_mzs, query_intensities):
        # Peak could be [frag+H]+ (+1.0073) or [frag-H]- (-1.0073) or neutral
        # Check closest match within mz_tolerance
        idx = np.searchsorted(sorted_frags, mz, side="left")
        hit = False

        for candidate_idx in [idx - 1, idx, idx + 1]:
            if 0 <= candidate_idx < len(sorted_frags):
                diff = abs(sorted_frags[candidate_idx] - mz)
                # Check directly or with +/- 1.0073 (proton adduct on fragment)
                if diff <= mz_tolerance or abs(diff - 1.0073) <= mz_tolerance:
                    hit = True
                    break

        if hit:
            matched_intensity += intensity
            matched_peaks += 1

    explained_ratio = matched_intensity / total_intensity
    peak_ratio = matched_peaks / len(query_mzs) if len(query_mzs) > 0 else 0.0

    # Harmonic balance between explained intensity and peak count
    score = 0.7 * explained_ratio + 0.3 * peak_ratio
    return float(min(max(score, 0.0), 1.0))
