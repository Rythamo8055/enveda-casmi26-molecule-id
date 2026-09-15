"""Multi-Adduct Hypothesis Generator for LC-MS/MS Precursor Recovery.

Natural product mass spectra frequently exhibit in-source neutral loss (e.g. -H2O)
or adduct misannotation (e.g. [M+NH4]+ or [M+Na]+ instead of [M+H]+).

This module generates alternative neutral mass hypotheses to recover candidate molecules
that would otherwise suffer 0% recall due to shifted precursor mass windows.
"""

from typing import Dict, List, Optional, Tuple

from src.preprocessing.adducts import calculate_neutral_mass, ADDUCT_OFFSETS


# Alternative adduct hypotheses to evaluate if nominal retrieval is poor
ALTERNATIVE_ADDUCT_MAP: Dict[str, List[Tuple[str, float]]] = {
    # Positive ion mode alternatives
    "[M+H]+": [
        ("[M-H2O+H]+", -17.003289),  # In-source water loss: true M = prec + 17.0033
        ("[M+NH4]+", 18.033823),     # Ammonium adduct: true M = prec - 18.0338
        ("[M+Na]+", 22.989218),      # Sodium adduct: true M = prec - 22.9892
    ],
    "[M+NH4]+": [
        ("[M+H]+", 1.007276),
        ("[M+Na]+", 22.989218),
    ],
    "[M+Na]+": [
        ("[M+H]+", 1.007276),
        ("[M+K]+", 38.963158),
    ],
    "[M-H2O+H]+": [
        ("[M+H]+", 1.007276),
    ],
    # Negative ion mode alternatives
    "[M-H]-": [
        ("[M-H2O-H]-", -19.017841),  # In-source water loss: true M = prec + 19.0178
        ("[M+CH2O2-H]-", 44.998203), # Formate adduct: true M = prec - 44.9982
    ],
    "[M+CH2O2-H]-": [
        ("[M-H]-", -1.007276),
    ],
}


def get_adduct_hypotheses(
    precursor_mz: float,
    nominal_adduct: str,
    include_alternatives: bool = True,
) -> List[Tuple[str, float]]:
    """Return list of (adduct_name, neutral_mass) hypotheses for precursor m/z.

    The first tuple is always the nominal adduct calculation.
    """
    hypotheses = []

    nominal_nm = calculate_neutral_mass(precursor_mz, nominal_adduct)
    if nominal_nm is not None and nominal_nm > 0:
        hypotheses.append((nominal_adduct, nominal_nm))

    if include_alternatives and nominal_adduct in ALTERNATIVE_ADDUCT_MAP:
        for alt_adduct, offset in ALTERNATIVE_ADDUCT_MAP[nominal_adduct]:
            alt_nm = precursor_mz - offset
            if alt_nm > 30.0:
                hypotheses.append((alt_adduct, alt_nm))

    return hypotheses
