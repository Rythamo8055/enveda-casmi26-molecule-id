"""Enveda CASMI 2026: Fast Spectral Library Matcher Baseline (Optimized Two-Stage).

Self-contained script ready to run directly in a Kaggle Notebook or locally.
1. Fast Row-Group Scan: Identifies candidate library spectra matching test neutral masses.
2. Numba Spectral Cosine: Computes square-root cosine similarity against matched candidates.
3. Multi-Spectrum Fusion: Aggregates scores per molecule_id.
4. InChIKey14 Deduplication: Guarantees 25 unique connectivity slots in submission.csv.
"""

import os
import time
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa
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
    """Calculate neutral monoisotopic mass from precursor m/z and adduct string."""
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
        ppm_tol: float = 20.0,
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
    ppm_tol: float = 20.0,
    mz_tol: float = 0.02,
    max_cands: int = 25,
) -> pl.DataFrame:
    grouped = defaultdict(list)
    for row in test_df.iter_rows(named=True):
        m_id = row["molecule_id"]
        prec = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        mzs = np.array(row["ms2_mzs"], dtype=np.float32)
        ints = np.array(row["ms2_normalized_intensities"], dtype=np.float32)
        # Sort peaks by mz
        order = np.argsort(mzs)
        grouped[m_id].append((prec, adduct, mzs[order], ints[order]))

    print(f"Aggregating predictions for {len(grouped)} test molecules...")
    submissions = []
    matched_count = 0

    for m_id, spectra in grouped.items():
        candidate_scores: Dict[str, Tuple[str, float, float]] = {}

        for prec, adduct, mzs, ints in spectra:
            hits = index.query(prec, adduct, mzs, ints, ppm_tol=ppm_tol, mz_tol=mz_tol)
            for smi, raw_inchikey14, sim in hits:
                # Always canonicalize to guarantee RDKit evaluation compatibility & strict uniqueness
                canonical_k14 = smiles_to_inchikey14(smi)
                if not canonical_k14:
                    continue  # Drop unparseable / kekulization-failed SMILES

                if canonical_k14 not in candidate_scores:
                    candidate_scores[canonical_k14] = (smi, sim, sim)
                else:
                    prev_smi, total_s, max_s = candidate_scores[canonical_k14]
                    candidate_scores[canonical_k14] = (
                        prev_smi,
                        total_s + sim,
                        max(max_s, sim),
                    )

        if candidate_scores:
            matched_count += 1
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

    print(f"Molecules with library spectral matches: {matched_count} / {len(grouped)}")
    sub_df = pl.DataFrame(submissions)
    sub_df.write_csv(output_file)
    print(f"Generated submission successfully at {output_file}")
    return sub_df


if __name__ == "__main__":
    t_start = time.time()
    kaggle_dir = "/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"
    local_dir = "data"
    data_dir = kaggle_dir if os.path.exists(kaggle_dir) else local_dir

    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    out_path = os.path.join(data_dir, "submission.csv") if data_dir == "data" else "submission.csv"

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Reading test spectra...")
        test_df = pl.read_parquet(test_path)

        # Build test neutral mass intervals (25 ppm window)
        test_intervals = []
        for row in test_df.select(["precursor_mz", "adduct"]).iter_rows(named=True):
            nm = calculate_neutral_mass(row["precursor_mz"], row["adduct"])
            if nm is not None and nm > 0:
                delta = nm * 25e-6
                test_intervals.append((nm - delta, nm + delta))

        test_intervals.sort()
        merged = []
        for start, end in test_intervals:
            if not merged or merged[-1][1] < start:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        starts = np.array([m[0] for m in merged], dtype=np.float64)
        ends = np.array([m[1] for m in merged], dtype=np.float64)

        def in_intervals(masses: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(ends, masses, side="left")
            valid = idx < len(starts)
            mask = np.zeros(len(masses), dtype=bool)
            mask[valid] = masses[valid] >= starts[idx[valid]]
            return mask

        print(f"Merged test intervals: {len(merged)}. Scanning train.parquet...")
        pq_file = pq.ParquetFile(train_path)
        matching_tables = []

        for i in range(pq_file.num_row_groups):
            meta_table = pq_file.read_row_group(i, columns=["precursor_mz", "adduct"])
            precursors = meta_table["precursor_mz"].to_numpy()
            adducts = meta_table["adduct"].to_pylist()

            neutral_m = np.zeros(len(precursors), dtype=np.float64)
            has_adduct = np.zeros(len(precursors), dtype=bool)
            for j, (p, a) in enumerate(zip(precursors, adducts)):
                off = ADDUCT_OFFSETS.get(a)
                if off is not None:
                    neutral_m[j] = p - off
                    has_adduct[j] = True

            hit_mask = has_adduct & in_intervals(neutral_m)
            hit_indices = np.where(hit_mask)[0]

            if len(hit_indices) > 0:
                full_rg = pq_file.read_row_group(
                    i,
                    columns=[
                        "normalized_smiles",
                        "inchikey14",
                        "precursor_mz",
                        "adduct",
                        "ms2_mzs",
                        "ms2_normalized_intensities",
                    ],
                )
                matching_tables.append(full_rg.take(hit_indices))

        matched_arrow = pa.concat_tables(matching_tables)
        candidate_train = pl.from_arrow(matched_arrow)
        print(f"Matched {len(candidate_train)} candidate library spectra in {time.time() - t_start:.2f}s!")

        # Compute neutral masses for candidate library spectra
        neutral_masses = []
        for p, a in zip(candidate_train["precursor_mz"], candidate_train["adduct"]):
            neutral_masses.append(calculate_neutral_mass(p, a))

        print("Pruning library spectra peaks to top 128...")
        pruned_mzs = []
        pruned_ints = []
        for mzs_raw, ints_raw in zip(
            candidate_train["ms2_mzs"], candidate_train["ms2_normalized_intensities"]
        ):
            m = np.array(mzs_raw, dtype=np.float32)
            i = np.array(ints_raw, dtype=np.float32)
            if len(m) > 128:
                top_idx = np.argpartition(i, -128)[-128:]
                top_idx = top_idx[np.argsort(m[top_idx])]
                m = m[top_idx]
                i = i[top_idx]
            else:
                order = np.argsort(m)
                m = m[order]
                i = i[order]
            pruned_mzs.append(m)
            pruned_ints.append(i)

        print("Building FastSpectralIndex...")
        index = FastSpectralIndex(
            neutral_masses=np.array(neutral_masses, dtype=np.float32),
            smiles_list=candidate_train["normalized_smiles"].to_list(),
            inchikey14_list=candidate_train["inchikey14"].to_list(),
            peak_mzs_list=pruned_mzs,
            peak_intensities_list=pruned_ints,
        )

        print("Running predictions on test set...")
        predict_molecules(test_df, index, output_file=out_path)
        print(f"Total end-to-end execution time: {time.time() - t_start:.2f}s!")
    else:
        print(f"Data files not found in {data_dir}.")
