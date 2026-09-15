"""Enveda CASMI 2026: Fast Spectral Library Matcher Baseline.

Self-contained script ready to run directly in a Kaggle Notebook or locally.
Executes Tier-1 exact library search with InChIKey14 deduplication.
"""

import os
import numpy as np
import polars as pl
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from numba import njit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# --- 1. ADDUCT MASS OFFSETS ---
ADDUCT_OFFSETS: Dict[str, float] = {
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033823,
    "[M-H2O+H]+": -17.003289,
    "[M-2H2O+H]+": -35.013854,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M-H]-": -1.007276,
    "[M-H2O-H]-": -19.017841,
    "[M+CH2O2-H]-": 44.998203,
    "[M+Cl]-": 34.969402,
}


def calculate_neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    """Calculate neutral mass from precursor m/z and adduct."""
    offset = ADDUCT_OFFSETS.get(adduct)
    return precursor_mz - offset if offset is not None else None


def smiles_to_inchikey14(smiles: str) -> str:
    """Canonicalize tautomer and return 14-character connectivity hash."""
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


# --- 2. NUMBA SPECTRAL COSINE SIMILARITY ---
@njit(fastmath=True)
def fast_cosine_similarity(
    mzs1: np.ndarray,
    intensities1: np.ndarray,
    mzs2: np.ndarray,
    intensities2: np.ndarray,
    mz_tolerance: float = 0.02,
) -> float:
    """Fast square-root weighted cosine similarity between sorted peak lists."""
    n1 = len(mzs1)
    n2 = len(mzs2)
    if n1 == 0 or n2 == 0:
        return 0.0

    w1 = np.sqrt(intensities1)
    w2 = np.sqrt(intensities2)
    norm1 = np.sqrt(np.sum(w1 * w1))
    norm2 = np.sqrt(np.sum(w2 * w2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    dot_product = 0.0
    j_start = 0

    for i in range(n1):
        mz1 = mzs1[i]
        best_match_j = -1
        min_diff = mz_tolerance

        for j in range(j_start, n2):
            diff = mzs2[j] - mz1
            if diff < -mz_tolerance:
                j_start = j
                continue
            if diff > mz_tolerance:
                break
            abs_diff = abs(diff)
            if abs_diff <= min_diff:
                min_diff = abs_diff
                best_match_j = j

        if best_match_j != -1:
            dot_product += w1[i] * w2[best_match_j]

    sim = dot_product / (norm1 * norm2)
    return float(min(max(sim, 0.0), 1.0))


# --- 3. PRECURSOR-INDEXED SPECTRAL LIBRARY ---
class FastSpectralIndex:
    def __init__(
        self,
        neutral_masses: np.ndarray,
        smiles_list: List[str],
        inchikey14_list: List[str],
        peak_mzs_list: List[np.ndarray],
        peak_intensities_list: List[np.ndarray],
    ):
        order = np.argsort(neutral_masses)
        self.neutral_masses = neutral_masses[order]
        self.smiles = [smiles_list[i] for i in order]
        self.inchikey14 = [inchikey14_list[i] for i in order]
        self.peak_mzs = [peak_mzs_list[i] for i in order]
        self.peak_intensities = [peak_intensities_list[i] for i in order]

    def query(
        self,
        precursor_mz: float,
        adduct: str,
        query_mzs: np.ndarray,
        query_intensities: np.ndarray,
        ppm_tol: float = 15.0,
        mz_tol: float = 0.02,
        top_k: int = 50,
    ) -> List[Tuple[str, str, float]]:
        neutral_m = calculate_neutral_mass(precursor_mz, adduct)
        if neutral_m is None or neutral_m <= 0:
            return []

        delta = neutral_m * ppm_tol * 1e-6
        left = np.searchsorted(self.neutral_masses, neutral_m - delta, side="left")
        right = np.searchsorted(self.neutral_masses, neutral_m + delta, side="right")

        if left >= right:
            return []

        results = []
        for idx in range(left, right):
            sim = fast_cosine_similarity(
                query_mzs,
                query_intensities,
                self.peak_mzs[idx],
                self.peak_intensities[idx],
                mz_tolerance=mz_tol,
            )
            if sim > 0.0:
                results.append((self.smiles[idx], self.inchikey14[idx], sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# --- 4. MULTI-SPECTRUM AGGREGATOR ---
def predict_molecules(
    test_df: pl.DataFrame,
    index: FastSpectralIndex,
    output_file: str = "submission.csv",
    ppm_tol: float = 15.0,
    mz_tol: float = 0.02,
    max_cands: int = 25,
) -> pl.DataFrame:
    grouped = defaultdict(list)
    for row in test_df.iter_rows(named=True):
        m_id = row["molecule_id"]
        prec = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        mzs = np.array(row["ms2_mzs"], dtype=np.float64)
        ints = np.array(row["ms2_normalized_intensities"], dtype=np.float64)
        grouped[m_id].append((prec, adduct, mzs, ints))

    print(f"Aggregating predictions for {len(grouped)} test molecules...")
    submissions = []

    for m_id, spectra in grouped.items():
        candidate_scores: Dict[str, Tuple[str, float, float]] = {}

        for prec, adduct, mzs, ints in spectra:
            hits = index.query(prec, adduct, mzs, ints, ppm_tol=ppm_tol, mz_tol=mz_tol)
            for smi, inchikey14, sim in hits:
                if not inchikey14:
                    inchikey14 = smiles_to_inchikey14(smi)
                if not inchikey14:
                    continue

                if inchikey14 not in candidate_scores:
                    candidate_scores[inchikey14] = (smi, sim, sim)
                else:
                    prev_smi, total_s, max_s = candidate_scores[inchikey14]
                    candidate_scores[inchikey14] = (
                        prev_smi,
                        total_s + sim,
                        max(max_s, sim),
                    )

        if candidate_scores:
            ranked = sorted(
                candidate_scores.values(),
                key=lambda x: (x[2], x[1]),
                reverse=True,
            )
            top_smiles = [c[0] for c in ranked[:max_cands]]
            smiles_str = ";".join(top_smiles)
        else:
            smiles_str = ""

        submissions.append({"molecule_id": m_id, "smiles": smiles_str})

    sub_df = pl.DataFrame(submissions)
    sub_df.write_csv(output_file)
    print(f"Generated submission successfully at {output_file}")
    return sub_df


if __name__ == "__main__":
    # Determine dataset location (Kaggle or local)
    kaggle_dir = "/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"
    local_dir = "data"
    data_dir = kaggle_dir if os.path.exists(kaggle_dir) else local_dir

    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading training data...")
        train_df = pl.read_parquet(
            train_path,
            columns=[
                "normalized_smiles",
                "inchikey14",
                "precursor_mz",
                "adduct",
                "ms2_mzs",
                "ms2_normalized_intensities",
            ],
        )

        print("Computing neutral masses via vector mapping...")
        # Map adduct strings to offsets
        adduct_col = train_df["adduct"].to_list()
        prec_col = train_df["precursor_mz"].to_numpy()

        neutral_masses = []
        valid_mask = []
        for prec, add in zip(prec_col, adduct_col):
            off = ADDUCT_OFFSETS.get(add)
            if off is not None and (prec - off) > 0:
                neutral_masses.append(prec - off)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        valid_mask = np.array(valid_mask, dtype=bool)
        print(f"Valid spectra with recognized adducts: {valid_mask.sum()} / {len(train_df)}")

        sub_train = train_df.filter(pl.Series(valid_mask))
        neutral_masses = np.array(neutral_masses, dtype=np.float32)

        # Prune each spectrum to top-128 peaks sorted by m/z for optimal speed and memory
        print("Pruning peaks to top-128 and preparing index...")
        pruned_mzs = []
        pruned_ints = []
        for mzs_raw, ints_raw in zip(sub_train["ms2_mzs"], sub_train["ms2_normalized_intensities"]):
            m = np.array(mzs_raw, dtype=np.float32)
            i = np.array(ints_raw, dtype=np.float32)
            if len(m) > 128:
                top_idx = np.argpartition(i, -128)[-128:]
                top_idx = top_idx[np.argsort(m[top_idx])]
                m = m[top_idx]
                i = i[top_idx]
            pruned_mzs.append(m)
            pruned_ints.append(i)

        print(f"Building spectral index with {len(sub_train)} spectra...")
        index = FastSpectralIndex(
            neutral_masses=neutral_masses,
            smiles_list=sub_train["normalized_smiles"].to_list(),
            inchikey14_list=sub_train["inchikey14"].to_list(),
            peak_mzs_list=pruned_mzs,
            peak_intensities_list=pruned_ints,
        )

        print("Loading test data...")
        test_df = pl.read_parquet(test_path)
        predict_molecules(test_df, index, output_file="submission.csv")
    else:
        print(f"Data files not found in {data_dir}. Ensure train.parquet and test.parquet exist.")
