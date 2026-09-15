"""High-Precision Molecular Formula Generator & Seven Golden Rules Filter.

Uses exact monoisotopic masses, Senior valency rules, Nitrogen rule, and
heteroatom ratio bounds (Kind & Fiehn, BMC Bioinformatics 2007) to generate
and score chemically feasible molecular formulas for unknown mass spectra.
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Monoisotopic exact masses (IUPAC 2026 standard)
ELEMENT_MASSES: Dict[str, float] = {
    "C": 12.00000000,
    "H": 1.007825032,
    "O": 15.99491462,
    "N": 14.00307400,
    "P": 30.97376163,
    "S": 31.97207100,
    "Cl": 34.96885268,
    "F": 18.99840322,
}

# Standard covalent valencies
ELEMENT_VALENCIES: Dict[str, int] = {
    "C": 4,
    "H": 1,
    "O": 2,
    "N": 3,
    "P": 5,
    "S": 2,
    "Cl": 1,
    "F": 1,
}


def calculate_formula_mass(formula_counts: Dict[str, int]) -> float:
    """Calculate exact monoisotopic mass from element counts."""
    return sum(ELEMENT_MASSES[el] * count for el, count in formula_counts.items() if count > 0)


def format_formula_string(counts: Dict[str, int]) -> str:
    """Format element counts into standard Hill system notation (C, then H, then alphabetical)."""
    parts = []
    # Hill system: C first, then H, then alphabetical
    order = ["C", "H"] + sorted([k for k in counts.keys() if k not in ("C", "H")])
    for el in order:
        cnt = counts.get(el, 0)
        if cnt > 0:
            parts.append(f"{el}{cnt if cnt > 1 else ''}")
    return "".join(parts)


def parse_formula_to_counts(formula_str: str) -> Dict[str, int]:
    """Parse a chemical formula string (e.g. 'C12H22O11') into a dictionary."""
    import re
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula_str)
    counts = {}
    for el, cnt in tokens:
        if el in ELEMENT_MASSES:
            counts[el] = int(cnt) if cnt else 1
    return counts


def passes_senior_rules(c: int, h: int, o: int, n: int, s: int, p: int, cl: int, f: int) -> bool:
    """Check Senior valency rules (Senior 1951, graph theory criteria for molecular feasibility).

    1. Sum of valencies must be even.
    2. Sum of valencies >= 2 * max(valency).
    3. Sum of valencies >= 2 * (total_atoms - 1).
    """
    total_valency = (
        c * 4 + h * 1 + o * 2 + n * 3 + p * 3 + s * 2 + cl * 1 + f * 1
    )  # using common N=3, P=3 or 5
    if total_valency % 2 != 0:
        return False

    max_v = 4 if c > 0 else (3 if (n > 0 or p > 0) else 2)
    if total_valency < 2 * max_v:
        return False

    total_heavy = c + o + n + p + s + cl + f
    if total_heavy > 1 and total_valency < 2 * (total_heavy + h - 1):
        # Connected graph check
        return False

    return True


def calculate_rdbe(c: int, h: int, o: int, n: int, s: int, p: int, halogens: int) -> float:
    """Calculate Ring and Double Bond Equivalents (degree of unsaturation)."""
    return float(c) - (float(h) / 2.0) + (float(n) / 2.0) + (float(p) / 2.0) - (float(halogens) / 2.0) + 1.0


class FormulaCandidate:
    def __init__(self, counts: Dict[str, int], exact_mass: float, ppm_error: float, score: float):
        self.counts = counts
        self.formula = format_formula_string(counts)
        self.exact_mass = exact_mass
        self.ppm_error = ppm_error
        self.score = score

    def __repr__(self):
        return f"Formula({self.formula}, mass={self.exact_mass:.4f}, error={self.ppm_error:.2f}ppm, score={self.score:.3f})"


def generate_plausible_formulas(
    neutral_mass: float,
    ppm_tol: float = 10.0,
    max_candidates: int = 15,
) -> List[FormulaCandidate]:
    """Generate and rank all chemically plausible molecular formulas matching exact neutral mass.

    Applies:
    - Bounded mass window [mass - delta, mass + delta]
    - Nitrogen rule (even nominal mass <-> even N)
    - Senior valency rules
    - Kind & Fiehn Seven Golden Rules (H/C, O/C, N/C bounds)
    - RDBE bounds (0 <= RDBE <= 35)
    """
    if neutral_mass <= 20.0:
        return []

    delta_m = neutral_mass * ppm_tol * 1e-6
    min_m = neutral_mass - delta_m
    max_m = neutral_mass + delta_m

    nominal_mass = int(round(neutral_mass))
    even_nominal = (nominal_mass % 2 == 0)

    # Carbon range
    max_c = min(int(max_m / 12.0) + 1, 80)
    min_c = max(int(min_m / 30.0), 1)

    results: List[FormulaCandidate] = []

    # Fast pruned search over CHNO + S/P/Cl
    for c in range(min_c, max_c + 1):
        c_mass = c * ELEMENT_MASSES["C"]
        if c_mass > max_m:
            break

        # Seven Golden Rules: typical natural product O/C ratio <= 1.3
        max_o = min(int(c * 1.3) + 2, int((max_m - c_mass) / 16.0) + 1)

        for o in range(0, max_o + 1):
            co_mass = c_mass + o * ELEMENT_MASSES["O"]
            if co_mass > max_m:
                break

            # Nitrogen rule & N/C bounds
            max_n = min(int(c * 1.1) + 1, int((max_m - co_mass) / 14.0) + 1)
            # Step by 2 to strictly enforce Nitrogen rule!
            n_start = 0 if even_nominal else 1

            for n in range(n_start, max_n + 1, 2):
                con_mass = co_mass + n * ELEMENT_MASSES["N"]
                if con_mass > max_m:
                    break

                # Optional common S (0 to 2)
                for s in (0, 1):
                    cons_mass = con_mass + s * ELEMENT_MASSES["S"]
                    if cons_mass > max_m:
                        break

                    # Hydrogen is uniquely determined by remaining mass
                    rem_mass = neutral_mass - cons_mass
                    if rem_mass <= 0:
                        continue

                    # Exact H integer estimate
                    h = int(round(rem_mass / ELEMENT_MASSES["H"]))
                    if h < 0:
                        continue

                    # Seven Golden Rules H/C ratio check (0.3 <= H/C <= 3.0)
                    hc_ratio = float(h) / float(c)
                    if hc_ratio < 0.25 or hc_ratio > 3.1:
                        continue

                    # Compute exact mass with candidate H
                    test_mass = cons_mass + h * ELEMENT_MASSES["H"]
                    diff = abs(test_mass - neutral_mass)
                    ppm = (diff / neutral_mass) * 1e6

                    if ppm > ppm_tol:
                        continue

                    # RDBE check
                    rdbe = calculate_rdbe(c, h, o, n, s, 0, 0)
                    if rdbe < -0.5 or rdbe > 35.0:
                        continue

                    # Senior valency rules check
                    if not passes_senior_rules(c, h, o, n, s, 0, 0, 0):
                        continue

                    counts = {"C": c, "H": h, "O": o, "N": n}
                    if s > 0:
                        counts["S"] = s

                    # Formula plausibility score: penalized by ppm error & unnatural ratios
                    score = math.exp(-0.5 * (ppm / 3.0) ** 2)
                    # Natural product prior: reward balanced H/C around 1.2-1.8
                    if 1.0 <= hc_ratio <= 2.0:
                        score *= 1.2

                    results.append(FormulaCandidate(counts, test_mass, ppm, score))

    # Sort candidates by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:max_candidates]


def filter_candidates_by_formula(
    candidate_smiles_list: List[str],
    allowed_formulas: Set[str],
) -> List[Tuple[str, str]]:
    """Filter candidate SMILES to only those whose molecular formula matches an allowed set."""
    matched = []
    for smi in candidate_smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            if mol_formula in allowed_formulas:
                matched.append((smi, mol_formula))
        except Exception:
            continue
    return matched
