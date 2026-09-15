"""Pairwise Feature Extractor for Spectrum-Candidate Ranking.

Extracts a compact 16-dimensional tabular feature vector combining:
1. Precursor mass accuracy & Seven Golden Rules formula plausibility
2. In-silico fragmentation coverage (intensity, peak ratios, top-5 peaks, neutral losses)
3. Biosynthetic Natural Product Likeness
4. Physico-chemical properties (LogP, TPSA, Fsp3, rotatable bonds, aromatic rings)

All candidate molecular properties are heavily cached in memory to minimize CPU cycles.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from src.retrieval.np_scorer import calculate_np_likeness
from src.retrieval.substructure_scorer import (
    generate_candidate_fragment_masses,
    EXPANDED_NEUTRAL_LOSSES,
)
from src.preprocessing.formula_generator import calculate_rdbe


FEATURE_NAMES = [
    "ppm_error",
    "mass_gaussian_score",
    "formula_golden_prior",
    "rdbe",
    "explained_intensity",
    "explained_peaks_fraction",
    "top5_intensity_explained",
    "matched_neutral_losses",
    "np_likeness_score",
    "fsp3",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "aromatic_rings",
    "heavy_atom_count",
    "h_bond_donors_acceptors",
]

# In-memory candidate descriptor cache: smiles -> dict of scalar props
_CANDIDATE_PROP_CACHE: Dict[str, Dict[str, float]] = {}


def get_candidate_properties(smiles: str) -> Dict[str, float]:
    """Retrieve or compute cached physico-chemical descriptors for a candidate SMILES."""
    if smiles in _CANDIDATE_PROP_CACHE:
        return _CANDIDATE_PROP_CACHE[smiles]

    props = {
        "fsp3": 0.0,
        "rotatable_bonds": 0.0,
        "tpsa": 0.0,
        "logp": 0.0,
        "aromatic_rings": 0.0,
        "heavy_atom_count": 0.0,
        "h_bond_donors_acceptors": 0.0,
        "rdbe": 0.0,
        "np_likeness": 0.5,
    }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            props["fsp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
            props["rotatable_bonds"] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
            props["tpsa"] = float(Descriptors.TPSA(mol))
            props["logp"] = float(Descriptors.MolLogP(mol))
            props["aromatic_rings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
            props["heavy_atom_count"] = float(mol.GetNumHeavyAtoms())
            props["h_bond_donors_acceptors"] = float(
                Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)
            )

            # Formula & RDBE directly from mol
            formula = rdMolDescriptors.CalcMolFormula(mol)
            c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
            h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
            n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
            p = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15)
            halos = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
            props["rdbe"] = float(c - (h / 2.0) + (n / 2.0) + (p / 2.0) - (halos / 2.0) + 1.0)
            props["formula"] = formula
    except Exception:
        pass

    props["np_likeness"] = calculate_np_likeness(smiles)
    _CANDIDATE_PROP_CACHE[smiles] = props
    return props


def extract_candidate_features(
    candidate_smiles: str,
    candidate_mass: float,
    query_mzs: np.ndarray,
    query_intensities: np.ndarray,
    precursor_mz: float,
    consensus_neutral_mass: float,
    formula_score_map: Optional[Dict[str, float]] = None,
    mz_tolerance: float = 0.02,
) -> np.ndarray:
    """Extract 16-dimensional feature vector for a (query, candidate) pair."""
    props = get_candidate_properties(candidate_smiles)

    # 1. Mass accuracy features
    if consensus_neutral_mass > 0:
        ppm_diff = abs(candidate_mass - consensus_neutral_mass) / consensus_neutral_mass * 1e6
        mass_gaussian = float(np.exp(-0.5 * (ppm_diff / 3.0) ** 2))
    else:
        ppm_diff = 15.0
        mass_gaussian = 0.0

    # 2. Formula Golden Rules score
    cand_formula = props.get("formula", "")
    golden_score = 0.0
    if formula_score_map and cand_formula in formula_score_map:
        golden_score = formula_score_map[cand_formula]

    # 3. In-silico fragmentation features
    explained_intensity = 0.0
    explained_peaks_fraction = 0.0
    top5_intensity_explained = 0.0
    matched_neutral_losses = 0.0

    if len(query_mzs) > 0 and query_intensities.sum() > 0:
        total_intensity = float(query_intensities.sum())

        try:
            mol = Chem.MolFromSmiles(candidate_smiles)
            if mol is not None:
                frag_masses = generate_candidate_fragment_masses(mol, max_cuts=2)
                sorted_frags = np.array(sorted(frag_masses), dtype=np.float64)

                matched_int = 0.0
                matched_cnt = 0
                frag_shifts = [0.0, 1.007276, -1.007276]

                # Identify top 5 query peaks by intensity
                top5_indices = set(np.argsort(query_intensities)[-5:])
                top5_total_int = float(query_intensities[list(top5_indices)].sum()) if top5_indices else 1.0
                top5_matched_int = 0.0

                for p_idx, (mz, intensity) in enumerate(zip(query_mzs, query_intensities)):
                    matched = False
                    for shift in frag_shifts:
                        target = mz - shift
                        idx = np.searchsorted(sorted_frags, target, side="left")
                        for c_idx in (idx - 1, idx, idx + 1):
                            if 0 <= c_idx < len(sorted_frags):
                                if abs(sorted_frags[c_idx] - target) <= mz_tolerance:
                                    matched = True
                                    break
                        if matched:
                            break

                    if matched:
                        matched_int += intensity
                        matched_cnt += 1
                        if p_idx in top5_indices:
                            top5_matched_int += intensity

                explained_intensity = matched_int / total_intensity
                explained_peaks_fraction = matched_cnt / len(query_mzs)
                top5_intensity_explained = top5_matched_int / max(top5_total_int, 1e-6)

                # Diagnostic neutral losses from precursor
                for nl_name, nl_mass in EXPANDED_NEUTRAL_LOSSES.items():
                    exp_frag_mz = precursor_mz - nl_mass
                    if exp_frag_mz > 20.0:
                        min_dist = np.min(np.abs(query_mzs - exp_frag_mz))
                        if min_dist <= mz_tolerance:
                            matched_neutral_losses += 1.0

        except Exception:
            pass

    features = np.array(
        [
            float(ppm_diff),
            float(mass_gaussian),
            float(golden_score),
            float(props.get("rdbe", 0.0)),
            float(explained_intensity),
            float(explained_peaks_fraction),
            float(top5_intensity_explained),
            float(matched_neutral_losses),
            float(props.get("np_likeness", 0.5)),
            float(props.get("fsp3", 0.0)),
            float(props.get("rotatable_bonds", 0.0)),
            float(props.get("tpsa", 0.0)),
            float(props.get("logp", 0.0)),
            float(props.get("aromatic_rings", 0.0)),
            float(props.get("heavy_atom_count", 0.0)),
            float(props.get("h_bond_donors_acceptors", 0.0)),
        ],
        dtype=np.float32,
    )

    return features
