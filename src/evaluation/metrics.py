"""Evaluation utilities for Enveda CASMI 2026."""

from typing import List, Sequence
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def smiles_to_inchikey14(smiles: str) -> str:
    """Canonicalize a SMILES string using RDKit tautomer standardization
    and return the 14-character first block of its InChIKey.
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        tautomer = rdMolStandardize.CanonicalTautomer(mol)
        inchikey = Chem.MolToInchiKey(tautomer)
        return inchikey[:14] if inchikey else ""
    except Exception:
        return ""


def calculate_mrr_at_k(
    predictions: Sequence[Sequence[str]],
    ground_truth: Sequence[str],
    k: int = 25,
) -> float:
    """Calculate Mean Reciprocal Rank @ k across a list of molecules.

    Args:
        predictions: List of candidate SMILES lists per molecule (up to k candidates).
        ground_truth: True SMILES string for each molecule.
        k: Maximum rank cutoff (default 25).

    Returns:
        MRR@k float value between 0.0 and 1.0.
    """
    assert len(predictions) == len(ground_truth), "Length mismatch between predictions and labels"

    reciprocal_ranks: List[float] = []

    for cand_list, true_smi in zip(predictions, ground_truth):
        true_key = smiles_to_inchikey14(true_smi)
        if not true_key:
            reciprocal_ranks.append(0.0)
            continue

        seen_keys = set()
        hit_rank = None
        current_rank = 1

        for smi in cand_list[:k]:
            cand_key = smiles_to_inchikey14(smi)
            if not cand_key or cand_key in seen_keys:
                continue
            seen_keys.add(cand_key)

            if cand_key == true_key:
                hit_rank = current_rank
                break
            current_rank += 1

        reciprocal_ranks.append(1.0 / hit_rank if hit_rank is not None else 0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
