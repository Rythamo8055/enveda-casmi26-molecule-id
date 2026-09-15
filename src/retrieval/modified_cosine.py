"""High-performance Modified Cosine Spectral Similarity with Precursor Mass Shift (GNPS Algorithm).

Modified Cosine matches peaks between query and library spectra in two ways:
1. Direct Peak Match: mz_q ≈ mz_lib (unmodified substructure fragment).
2. Shifted Peak Match: mz_q ≈ mz_lib + (precursor_q - precursor_lib) (fragment containing modification).

Enables identification of core scaffolds and natural product analogs (e.g. +OH, +CH3, +glycosyl)
even when the exact molecule is absent from public libraries.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from numba import njit


@njit(fastmath=True)
def modified_cosine_similarity(
    mzs_q: np.ndarray,
    ints_q: np.ndarray,
    precursor_q: float,
    mzs_lib: np.ndarray,
    ints_lib: np.ndarray,
    precursor_lib: float,
    mz_tolerance: float = 0.02,
    min_matched_peaks: int = 2,
) -> Tuple[float, int]:
    """Compute modified cosine score between query and reference spectra.

    Returns:
        similarity (float between 0.0 and 1.0)
        matched_peak_count (int)
    """
    nq = len(mzs_q)
    nlib = len(mzs_lib)
    if nq == 0 or nlib == 0:
        return 0.0, 0

    delta_precursor = precursor_q - precursor_lib

    # Square-root intensity transformation
    wq = np.sqrt(ints_q)
    wlib = np.sqrt(ints_lib)

    norm_q = np.sqrt(np.sum(wq * wq))
    norm_lib = np.sqrt(np.sum(wlib * wlib))

    if norm_q == 0.0 or norm_lib == 0.0:
        return 0.0, 0

    dot_product = 0.0
    matched_count = 0

    # Track which library peaks have already been matched to avoid double-counting
    lib_matched = np.zeros(nlib, dtype=np.bool_)

    # 1. First pass: Match Direct Peaks (mz_q ≈ mz_lib)
    for i in range(nq):
        mz_i = mzs_q[i]
        best_j = -1
        min_diff = mz_tolerance

        for j in range(nlib):
            if lib_matched[j]:
                continue
            diff = abs(mzs_lib[j] - mz_i)
            if diff <= min_diff:
                min_diff = diff
                best_j = j

        if best_j != -1:
            dot_product += wq[i] * wlib[best_j]
            lib_matched[best_j] = True
            matched_count += 1

    # 2. Second pass: Match Shifted Peaks (mz_q ≈ mz_lib + delta_precursor)
    # Only if delta_precursor is non-negligible (> 0.05 Da)
    if abs(delta_precursor) > 0.05:
        for i in range(nq):
            mz_i = mzs_q[i]
            target_lib_mz = mz_i - delta_precursor
            if target_lib_mz <= 0.0:
                continue

            best_j = -1
            min_diff = mz_tolerance

            for j in range(nlib):
                if lib_matched[j]:
                    continue
                diff = abs(mzs_lib[j] - target_lib_mz)
                if diff <= min_diff:
                    min_diff = diff
                    best_j = j

            if best_j != -1:
                dot_product += wq[i] * wlib[best_j]
                lib_matched[best_j] = True
                matched_count += 1

    if matched_count < min_matched_peaks:
        return 0.0, matched_count

    score = dot_product / (norm_q * norm_lib)
    return float(min(max(score, 0.0), 1.0)), matched_count
