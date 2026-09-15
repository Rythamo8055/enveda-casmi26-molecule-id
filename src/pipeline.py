"""Multi-spectrum aggregation, InChIKey14 deduplication, and submission generator."""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import polars as pl

from src.retrieval.spectral_matcher import SpectralLibraryIndex
from src.evaluation.metrics import smiles_to_inchikey14


def aggregate_molecule_predictions(
    molecule_spectra: List[Tuple[float, str, np.ndarray, np.ndarray]],
    index: SpectralLibraryIndex,
    top_k: int = 25,
    ppm_tol: float = 15.0,
    mz_tol: float = 0.02,
) -> str:
    """Aggregate predictions across all spectra of a single molecule and deduplicate.

    Args:
        molecule_spectra: List of (precursor_mz, adduct, ms2_mzs, ms2_intensities).
        index: Precursor-indexed spectral library.
        top_k: Maximum candidate count (default 25).
        ppm_tol: Precursor mass matching tolerance in ppm.
        mz_tol: Fragment peak matching tolerance in Da.

    Returns:
        Semicolon-separated string of up to top_k distinct InChIKey14 candidate SMILES.
    """
    # Map: InChIKey14 -> (best_smiles, total_score, max_score)
    candidate_scores: Dict[str, Tuple[str, float, float]] = {}

    for precursor_mz, adduct, mzs, intensities in molecule_spectra:
        matches = index.query(
            precursor_mz=precursor_mz,
            adduct=adduct,
            query_mzs=mzs,
            query_intensities=intensities,
            ppm_tol=ppm_tol,
            mz_tol=mz_tol,
            top_k=50,
        )

        for smi, inchikey14, score in matches:
            if not inchikey14:
                inchikey14 = smiles_to_inchikey14(smi)
            if not inchikey14:
                continue

            if inchikey14 not in candidate_scores:
                candidate_scores[inchikey14] = (smi, score, score)
            else:
                prev_smi, total_s, max_s = candidate_scores[inchikey14]
                # Aggregate score: sum + boost from max match
                candidate_scores[inchikey14] = (
                    prev_smi,
                    total_s + score,
                    max(max_s, score),
                )

    if not candidate_scores:
        return ""

    # Sort candidates by combined score (max_score primary, total_score secondary)
    ranked = sorted(
        candidate_scores.values(),
        key=lambda x: (x[2], x[1]),
        reverse=True,
    )

    # Extract up to top_k SMILES
    top_smiles = [item[0] for item in ranked[:top_k]]
    return ";".join(top_smiles)


def generate_submission(
    test_df: pl.DataFrame,
    index: SpectralLibraryIndex,
    output_path: str = "submission.csv",
    top_k: int = 25,
    ppm_tol: float = 15.0,
    mz_tol: float = 0.02,
) -> pl.DataFrame:
    """Run end-to-end baseline inference on test spectra and save submission.csv."""
    # Group spectra by molecule_id
    grouped = defaultdict(list)
    for row in test_df.iter_rows(named=True):
        m_id = row["molecule_id"]
        prec_mz = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        mzs = np.array(row["ms2_mzs"], dtype=np.float64)
        ints = np.array(row["ms2_normalized_intensities"], dtype=np.float64)
        grouped[m_id].append((prec_mz, adduct, mzs, ints))

    results = []
    for m_id, spectra in grouped.items():
        smiles_pred = aggregate_molecule_predictions(
            spectra, index, top_k=top_k, ppm_tol=ppm_tol, mz_tol=mz_tol
        )
        results.append({"molecule_id": m_id, "smiles": smiles_pred})

    sub_df = pl.DataFrame(results)
    sub_df.write_csv(output_path)
    print(f"Saved submission with {len(sub_df)} molecules to {output_path}")
    return sub_df
