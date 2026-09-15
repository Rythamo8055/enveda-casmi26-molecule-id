"""Adduct definitions and neutral mass calculation utilities."""

from typing import Dict, Optional

# Exact monoisotopic masses of adducts / modifications
# Monoisotopic masses: H=1.007825032, Na=22.989769, K=38.963706, Cl35=34.968853, H2O=18.010565, Formic Acid CH2O2=46.005479
# Net mass added to neutral molecule M:
ADDUCT_OFFSETS: Dict[str, float] = {
    # Positive mode adducts
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033823,
    "[M-H2O+H]+": -17.003289,
    "[M-2H2O+H]+": -35.013854,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    # Negative mode adducts
    "[M-H]-": -1.007276,
    "[M-H2O-H]-": -19.017841,
    "[M+CH2O2-H]-": 44.998203,
    "[M+Cl]-": 34.969402,
}


def calculate_neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    """Calculate the neutral monoisotopic mass M from precursor m/z and adduct string.

    Args:
        precursor_mz: Measured m/z of the precursor ion.
        adduct: Adduct string (e.g. '[M+H]+', '[M+Na]+').

    Returns:
        Neutral monoisotopic mass as float, or None if adduct is unrecognized.
    """
    offset = ADDUCT_OFFSETS.get(adduct)
    if offset is None:
        return None
    return precursor_mz - offset


def calculate_precursor_mz(neutral_mass: float, adduct: str) -> Optional[float]:
    """Calculate the expected precursor m/z for a given neutral mass and adduct."""
    offset = ADDUCT_OFFSETS.get(adduct)
    if offset is None:
        return None
    return neutral_mass + offset


def is_within_ppm(m1: float, m2: float, ppm_tol: float = 15.0) -> bool:
    """Check whether two masses are within ppm tolerance."""
    if m1 <= 0 or m2 <= 0:
        return False
    return abs(m1 - m2) / m2 * 1e6 <= ppm_tol
