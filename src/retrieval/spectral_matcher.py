"""High-speed spectral library search and scoring."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from numba import njit

from src.preprocessing.adducts import calculate_neutral_mass


@njit(fastmath=True)
def fast_cosine_similarity(
    mzs1: np.ndarray,
    intensities1: np.ndarray,
    mzs2: np.ndarray,
    intensities2: np.ndarray,
    mz_tolerance: float = 0.02,
) -> float:
    """Compute weighted cosine dot-product similarity between two peak lists.

    Uses square-root intensity weighting. Matches peaks within mz_tolerance.
    """
    n1 = len(mzs1)
    n2 = len(mzs2)
    if n1 == 0 or n2 == 0:
        return 0.0

    # Square-root intensity transformation
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

        # Two-pointer linear scan on sorted mzs
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

    similarity = dot_product / (norm1 * norm2)
    return float(min(max(similarity, 0.0), 1.0))


class SpectralLibraryIndex:
    """Precursor-indexed mass spectral library for sub-millisecond retrieval."""

    def __init__(
        self,
        neutral_masses: np.ndarray,
        smiles_list: List[str],
        inchikey14_list: List[str],
        peak_mzs_list: List[np.ndarray],
        peak_intensities_list: List[np.ndarray],
        collision_energies: Optional[List[float]] = None,
    ):
        """Store sorted library spectra indexed by neutral mass."""
        assert len(neutral_masses) == len(smiles_list)
        # Sort by neutral mass for fast binary search
        order = np.argsort(neutral_masses)
        self.neutral_masses = neutral_masses[order]
        self.smiles = [smiles_list[i] for i in order]
        self.inchikey14 = [inchikey14_list[i] for i in order]
        self.peak_mzs = [peak_mzs_list[i] for i in order]
        self.peak_intensities = [peak_intensities_list[i] for i in order]
        if collision_energies is not None:
            self.collision_energies = [collision_energies[i] for i in order]
        else:
            self.collision_energies = None

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
        """Find candidate matches within mass tolerance and rank by spectral similarity.

        Returns:
            List of (smiles, inchikey14, similarity_score) sorted by score descending.
        """
        query_neutral = calculate_neutral_mass(precursor_mz, adduct)
        if query_neutral is None or query_neutral <= 0:
            return []

        delta_m = query_neutral * ppm_tol * 1e-6
        min_m = query_neutral - delta_m
        max_m = query_neutral + delta_m

        # Binary search range
        left_idx = np.searchsorted(self.neutral_masses, min_m, side="left")
        right_idx = np.searchsorted(self.neutral_masses, max_m, side="right")

        if left_idx >= right_idx:
            return []

        candidates = []
        for idx in range(left_idx, right_idx):
            score = fast_cosine_similarity(
                query_mzs,
                query_intensities,
                self.peak_mzs[idx],
                self.peak_intensities[idx],
                mz_tolerance=mz_tol,
            )
            if score > 0.0:
                candidates.append((self.smiles[idx], self.inchikey14[idx], score))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def query_analog(
        self,
        precursor_mz: float,
        adduct: str,
        query_mzs: np.ndarray,
        query_intensities: np.ndarray,
        max_mass_shift: float = 80.0,
        mz_tol: float = 0.02,
        min_matched_peaks: int = 3,
        top_k: int = 20,
    ) -> List[Tuple[str, str, float]]:
        """Find analog matches with precursor mass shifts (e.g. +OH, +CH3, +glycosyl)

        using Modified Cosine.
        """
        from src.retrieval.modified_cosine import modified_cosine_similarity

        query_neutral = calculate_neutral_mass(precursor_mz, adduct)
        if query_neutral is None or query_neutral <= 0:
            return []

        min_m = max(query_neutral - max_mass_shift, 50.0)
        max_m = query_neutral + max_mass_shift

        left_idx = np.searchsorted(self.neutral_masses, min_m, side="left")
        right_idx = np.searchsorted(self.neutral_masses, max_m, side="right")

        if left_idx >= right_idx:
            return []

        # Subsample if search window is very large (> 2000 spectra)
        total_in_window = right_idx - left_idx
        step = max(total_in_window // 1000, 1)

        candidates = []
        for idx in range(left_idx, right_idx, step):
            score, n_matched = modified_cosine_similarity(
                query_mzs,
                query_intensities,
                query_neutral,
                self.peak_mzs[idx],
                self.peak_intensities[idx],
                self.neutral_masses[idx],
                mz_tolerance=mz_tol,
                min_matched_peaks=min_matched_peaks,
            )
            if score > 0.40:
                candidates.append((self.smiles[idx], self.inchikey14[idx], score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

