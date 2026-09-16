import json

nb = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [text + '\n']
    })

def add_code(text):
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.split('\n')]
    })

# --- CELL 0: Header ---
add_md('''# CASMI 2026: Dual-Channel Gated COCONUT Retrieval Engine (v15)
### The SOTA Leaderboard Architecture (Targeting 0.25 - 0.28+ MRR)

This pipeline directly addresses the rank cannibalization problem of v14 by introducing:
1. **Protected Strong-Library Gate:** When an experimental library match achieves a blended similarity $\\ge 0.55$, it is **strictly guaranteed Top-8 rank** (never demoted by database heuristics).
2. **24-Adduct Accurate Neutral-Mass De-adducting:** Neutralizes precursor ions across all 24 electrospray ionization adducts with exact isotopic masses.
3. **Dual-Channel Matching (Aligned Dot Product):**
   - **Fragment Channel:** Direct $m/z$ alignment within $\\pm 0.015$ Da.
   - **Neutral-Loss Channel:** Precursor-shifted alignment ($M_{\\text{prec}} - m/z$) to capture core-skeleton analogs.
   - **Composite Score:** $0.70 \\times \\text{Fragment} + 0.30 \\times \\text{Loss}$.
4. **Natural Product Source Prior:** Small $+0.02$ boost for structures confirmed in reference NP repositories (GNPS, RIKEN, MassBank, MONA).
5. **COCONUT Natural Products Integration:** 422,926 structures indexed by exact mass, seamlessly populating candidate slots when library matches are weak or absent.
6. **Strict InChIKey14 Deduplication:** Exactly 25 unique connectivity candidates per test molecule.
''')

# --- CELL 1: Environment & File Discovery ---
add_code('''# Cell 1: Environment & File Discovery
import glob
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
from tqdm.auto import tqdm

def find_file(name: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        hits = glob.glob(f"data/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"File {name} not found!")
    return sorted(hits, key=len)[0]

TRAIN = find_file("train.parquet")
TEST = find_file("test.parquet")
SAMPLE = find_file("sample_submission.csv")

# Find COCONUT dataset
COCONUT = None
for fname in ["coconut_indexed.parquet", "COCONUT4MetFrag_april.csv", "coconut_dataset.parquet"]:
    try:
        COCONUT = find_file(fname)
        break
    except FileNotFoundError:
        continue

print(f"Train:   {TRAIN}")
print(f"Test:    {TEST}")
print(f"Sample:  {SAMPLE}")
print(f"COCONUT: {COCONUT}")
''')

# --- CELL 2: 24-Adduct Neutral Mass Engine ---
add_code('''# Cell 2: 24-Adduct Accurate Neutral Mass Calculations
MASS = {
    "C": 12.0, "H": 1.0078250319, "N": 14.0030740052,
    "O": 15.9949146221, "P": 30.97376151, "S": 31.97207069,
    "F": 18.9984032, "Cl": 34.96885271, "Br": 78.9183376,
    "I": 126.904473, "Na": 22.9897692809, "K": 38.9637064864,
}
PROTON = 1.0072764665
ELECTRON = 0.0005485799
H2O = 2 * MASS["H"] + MASS["O"]
NH4 = MASS["N"] + 4 * MASS["H"]
FORMATE = MASS["C"] + 2 * MASS["H"] + 2 * MASS["O"]
ACETATE = 2 * MASS["C"] + 4 * MASS["H"] + 2 * MASS["O"]

ADDUCTS = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, NH4 - ELECTRON),
    "[M+Na]+": (1, 1, MASS["Na"] - ELECTRON),
    "[M+K]+": (1, 1, MASS["K"] - ELECTRON),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[M]+": (1, 1, -ELECTRON),
    "[M-H2O]+": (1, 1, -ELECTRON - H2O),
    "[M-H]-": (1, 1, -PROTON),
    "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON),
    "[M+Cl]-": (1, 1, MASS["Cl"] + ELECTRON),
    "[M]-": (1, 1, ELECTRON),
    "[M-2H]-": (1, 2, -2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, MASS["Na"] - ELECTRON),
    "[2M+NH4]+": (2, 1, NH4 - ELECTRON),
    "[2M-H]-": (2, 1, -PROTON),
    "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON),
    "[2M+Na-2H]-": (2, 1, MASS["Na"] - 2 * PROTON),
    "[2M+K]+": (2, 1, MASS["K"] - ELECTRON),
}

def neutral_mass_one(precursor_mz: float, adduct: str) -> float:
    spec = ADDUCTS.get(adduct)
    if spec is None:
        return np.nan
    molecules, charge, delta = spec
    return (float(precursor_mz) * charge - delta) / molecules

def neutral_mass_array(precursor_mz: np.ndarray, adduct: np.ndarray) -> np.ndarray:
    out = np.full(len(precursor_mz), np.nan, dtype=np.float64)
    for name, (molecules, charge, delta) in ADDUCTS.items():
        mask = adduct == name
        if mask.any():
            out[mask] = (precursor_mz[mask] * charge - delta) / molecules
    return out
''')

# --- CELL 3: Dual-Channel Similarity Kernels ---
add_code('''# Cell 3: Spectral Preprocessing & Aligned Dot Product (Numba)
PEAK_TOL = 0.015
MASS_TOL = 0.012
NOISE_FLOOR = 0.005
MAX_PEAKS = 160
TOPN = 25

@njit(cache=True, fastmath=True)
def aligned_dot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, tol: float = 0.015) -> float:
    i = 0
    j = 0
    score = 0.0
    n1 = len(x1)
    n2 = len(x2)
    while i < n1 and j < n2:
        delta = x1[i] - x2[j]
        if delta < -tol:
            i += 1
        elif delta > tol:
            j += 1
        else:
            score += y1[i] * y2[j]
            i += 1
            j += 1
    return score

def prepare_spectrum(mz, intensity, precursor_mz):
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    good = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    mz = mz[good]
    intensity = intensity[good]
    if len(intensity) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    keep = intensity >= NOISE_FLOOR * intensity.max()
    mz = mz[keep]
    intensity = intensity[keep]

    if len(intensity) > MAX_PEAKS:
        idx = np.argpartition(intensity, -MAX_PEAKS)[-MAX_PEAKS:]
        mz = mz[idx]
        intensity = intensity[idx]

    # Peak intensity weighting: sqrt(I)
    weight = np.sqrt(intensity)
    norm = np.linalg.norm(weight)
    if norm > 1e-12:
        weight /= norm

    # Direct fragment arrays
    order = np.argsort(mz)
    frag_mz = np.ascontiguousarray(mz[order])
    frag_weight = np.ascontiguousarray(weight[order])

    # Neutral loss arrays (precursor - mz)
    loss = float(precursor_mz) - mz
    valid_loss = loss > 0
    loss = loss[valid_loss]
    loss_weight = weight[valid_loss]
    loss_order = np.argsort(loss)

    return (
        frag_mz,
        frag_weight,
        np.ascontiguousarray(loss[loss_order]),
        np.ascontiguousarray(loss_weight[loss_order]),
    )
''')

# --- CELL 4: Streamed Reference Library Indexing ---
add_code('''# Cell 4: Load and Index Reference Library & COCONUT Database
print("Loading reference library & COCONUT database...")
t0 = time.time()

# 1. Read test spectra and compute test neutral mass intervals (25 ppm window)
test = pq.read_table(TEST).to_pandas()
sample = pd.read_csv(SAMPLE)
print(f"Loaded test spectra: {len(test):,} rows across {test['molecule_id'].nunique()} unique molecules.")

test_intervals = []
for _, row in test[["precursor_mz", "adduct"]].iterrows():
    nm = neutral_mass_one(row["precursor_mz"], row["adduct"])
    if np.isfinite(nm) and nm > 0:
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

def in_test_intervals(masses: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ends, masses, side="left")
    valid = idx < len(starts)
    mask = np.zeros(len(masses), dtype=bool)
    mask[valid] = masses[valid] >= starts[idx[valid]]
    return mask

# 2. Stream matching row-groups from train.parquet
pq_file = pq.ParquetFile(TRAIN)
matching_tables = []
t_scan = time.time()

for i in range(pq_file.num_row_groups):
    meta = pq_file.read_row_group(i, columns=["precursor_mz", "adduct"])
    prec_arr = meta["precursor_mz"].to_numpy()
    adduct_arr = np.asarray(meta["adduct"].to_pylist(), dtype=object)

    nms = neutral_mass_array(prec_arr, adduct_arr)
    valid_mask = np.isfinite(nms) & in_test_intervals(nms)
    hit_indices = np.where(valid_mask)[0]

    if len(hit_indices) > 0:
        full_rg = pq_file.read_row_group(
            i,
            columns=[
                "inchikey14", "normalized_smiles", "adduct", "precursor_mz",
                "ms2_mzs", "ms2_normalized_intensities", "ingest_lib",
            ],
        )
        matching_tables.append(full_rg.take(hit_indices))

matched_arrow = pa.concat_tables(matching_tables)
print(f"Matched {len(matched_arrow):,} candidate library spectra in {time.time() - t_scan:.2f}s!")

# Build library representation
mz_col = matched_arrow.column("ms2_mzs").combine_chunks()
int_col = matched_arrow.column("ms2_normalized_intensities").combine_chunks()
offsets = mz_col.offsets.to_numpy().astype(np.int64)
all_mz = mz_col.values.to_numpy(zero_copy_only=False).astype(np.float32)
all_int = int_col.values.to_numpy(zero_copy_only=False).astype(np.float32)

precursor = matched_arrow.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
adduct = np.asarray(matched_arrow.column("adduct").cast(pa.string()).to_pylist(), dtype=object)
inchikey = np.asarray(matched_arrow.column("inchikey14").cast(pa.string()).to_pylist(), dtype=object)
smiles = np.asarray(matched_arrow.column("normalized_smiles").cast(pa.string()).to_pylist(), dtype=object)
source = np.asarray(matched_arrow.column("ingest_lib").cast(pa.string()).to_pylist(), dtype=object)

neutral = neutral_mass_array(precursor, adduct)
order = np.argsort(np.where(np.isfinite(neutral), neutral, np.inf), kind="mergesort")
best_smiles = {}
source_sets = defaultdict(set)
for key, smi, src in zip(inchikey, smiles, source):
    if key and smi and key not in best_smiles:
        best_smiles[key] = smi
    if key and src:
        source_sets[key].add(src)

library = {
    "offsets": offsets, "all_mz": all_mz, "all_int": all_int,
    "precursor": precursor, "adduct": adduct, "inchikey": inchikey,
    "source": source, "neutral": neutral, "order": order,
    "sorted_neutral": neutral[order], "best_smiles": best_smiles,
    "source_sets": source_sets,
}
print(f"Indexed {len(library['order']):,} candidate library spectra ({len(best_smiles):,} structures).")

# 3. Load COCONUT natural products
if COCONUT and COCONUT.endswith(".parquet"):
    coco_df = pl.read_parquet(COCONUT).to_pandas()
    coco_df = coco_df.dropna(subset=["exact_mass", "clean_smiles", "inchikey14"])
    coco_df = coco_df.drop_duplicates("inchikey14").sort_values("exact_mass").reset_index(drop=True)
    coconut = {
        "mass": coco_df["exact_mass"].to_numpy(),
        "key": coco_df["inchikey14"].to_numpy(),
        "smiles": coco_df["clean_smiles"].to_numpy(),
    }
    print(f"Loaded {len(coco_df):,} unique COCONUT structures from parquet.")
elif COCONUT and COCONUT.endswith(".csv"):
    coco_df = pd.read_csv(COCONUT)
    coco_df["inchikey14"] = coco_df["inchikey"].astype(str).str[:14]
    coco_df = coco_df.drop_duplicates("inchikey14").sort_values("exact_mass").reset_index(drop=True)
    coconut = {
        "mass": coco_df["exact_mass"].to_numpy(),
        "key": coco_df["inchikey14"].to_numpy(),
        "smiles": coco_df["clean_smiles"].to_numpy(),
    }
    print(f"Loaded {len(coco_df):,} unique COCONUT structures from CSV.")
else:
    coconut = {"mass": np.array([]), "key": np.array([]), "smiles": np.array([])}
    print("Warning: COCONUT dataset not available.")
''')

# --- CELL 5: Evidence Retrieval with Dual Channel & NP Prior ---
add_code('''# Cell 5: Evidence Retrieval per Molecule
NP_SOURCES = {"gnps", "riken", "pluskal_ms2", "massbank", "mona", "msdial", "masaryk", "spectraverse", "enveda-np-examples"}

def candidate_window(target_mass: float, tol: float = MASS_TOL):
    lo = np.searchsorted(library["sorted_neutral"], target_mass - tol, side="left")
    hi = np.searchsorted(library["sorted_neutral"], target_mass + tol, side="right")
    return library["order"][lo:hi]

def coconut_candidates(target_mass: float, tol: float = MASS_TOL, limit: int = 120):
    if len(coconut["mass"]) == 0:
        return []
    lo = np.searchsorted(coconut["mass"], target_mass - tol, side="left")
    hi = np.searchsorted(coconut["mass"], target_mass + tol, side="right")
    if hi <= lo:
        return []
    idx = np.arange(lo, hi)
    idx = idx[np.argsort(np.abs(coconut["mass"][idx] - target_mass), kind="stable")]
    return [(coconut["key"][i], coconut["smiles"][i]) for i in idx[:limit]]

def retrieve_molecule(group):
    per_query = []
    masses = []
    for row in group.itertuples():
        neutral = neutral_mass_one(row.precursor_mz, row.adduct)
        if not np.isfinite(neutral):
            continue
        masses.append(neutral)
        qfm, qfw, qlm, qlw = prepare_spectrum(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz)
        if len(qfm) == 0:
            continue
        q_scores = {}
        for cand in candidate_window(neutral):
            begin = int(library["offsets"][cand])
            end = int(library["offsets"][cand + 1])
            if end <= begin:
                continue
            cfm, cfw, clm, clw = prepare_spectrum(
                library["all_mz"][begin:end],
                library["all_int"][begin:end],
                library["precursor"][cand],
            )
            direct = aligned_dot(qfm, qfw, cfm, cfw, PEAK_TOL)
            loss = aligned_dot(qlm, qlw, clm, clw, PEAK_TOL)
            key = library["inchikey"][cand]
            previous = q_scores.get(key)
            if previous is None or max(direct, loss) > max(previous[0], previous[1]):
                q_scores[key] = (float(direct), float(loss))
        per_query.append(q_scores)

    evidence = {}
    all_keys = set().union(*(scores.keys() for scores in per_query)) if per_query else set()
    for key in all_keys:
        pairs = [scores[key] for scores in per_query if key in scores]
        blended = sorted((0.70 * d + 0.30 * l for d, l in pairs), reverse=True)
        direct_max = max(d for d, _ in pairs)
        loss_max = max(l for _, l in pairs)
        second = blended[1] if len(blended) > 1 else 0.0
        support = len(pairs)
        source_prior = 1.0 if library["source_sets"][key] & NP_SOURCES else 0.0
        evidence[key] = (direct_max, loss_max, blended[0], second, support, source_prior)

    target_mass = float(np.median(masses)) if masses else np.nan
    return evidence, target_mass

print("Retrieving evidence across all test molecules...")
t_ret = time.time()
retrieved = {}
for molecule_id, group in tqdm(test.groupby("molecule_id", sort=False), desc="retrieving"):
    retrieved[molecule_id] = retrieve_molecule(group)
print(f"Retrieved evidence for {len(retrieved)} molecules in {time.time() - t_ret:.1f}s!")
''')

# --- CELL 6: Gated COCONUT Assembly (Resolving Cannibalization) ---
add_code('''# Cell 6: Gated COCONUT Assembly (Strict Hierarchy Preservation)
def library_rank(evidence):
    scored = []
    for key, (direct, loss, blend, second, support, np_prior) in evidence.items():
        # Score combining primary blend, secondary spectrum confirmation, multi-spectrum support, and NP source prior
        score = blend + 0.12 * second + 0.01 * min(support, 4) + 0.02 * np_prior
        scored.append((key, score, blend))
    return sorted(scored, key=lambda item: (-item[1], item[0]))

def assemble_gated_coconut(molecule_id: str) -> str:
    evidence, target_mass = retrieved[molecule_id]
    ranked_library = library_rank(evidence)
    coco = coconut_candidates(target_mass) if np.isfinite(target_mass) else []
    output = []
    seen = set()

    def push(key, smiles):
        if key in seen or not smiles:
            return
        seen.add(key)
        output.append(smiles)

    # 1. STRONG LIBRARY MATCHES (Ground truth tier, blend >= 0.55) -> GUARANTEED TOP RANKS!
    # These experimental hits are never displaced by theoretical database candidates.
    strong = [item for item in ranked_library if item[2] >= 0.55]
    weak = [item for item in ranked_library if item[2] < 0.55]

    for key, _, _ in strong[:8]:
        push(key, library["best_smiles"].get(key))

    # 2. COCONUT NATURAL PRODUCTS (Class 2 retrieval for unknown/novel spectra)
    for key, smiles in coco:
        push(key, smiles)
        if len(output) >= TOPN:
            break

    # 3. WEAK LIBRARY MATCHES (Appended to fill any remaining candidate slots)
    for key, _, _ in weak:
        push(key, library["best_smiles"].get(key))
        if len(output) >= TOPN:
            break

    # 4. Fallback if still under 25
    if len(output) < TOPN:
        for smiles in library["best_smiles"].values():
            push(f"fallback:{smiles}", smiles)
            if len(output) >= TOPN:
                break

    return ";".join(output[:TOPN])
''')

# --- CELL 7: Generate and Validate Submission ---
add_code('''# Cell 7: Generate Submission File & Verify Contract
print("Generating final predictions for submission...")
final_smiles = [assemble_gated_coconut(mid) for mid in sample["molecule_id"]]
sub = pd.DataFrame({
    "molecule_id": sample["molecule_id"],
    "smiles": final_smiles,
})

sub.to_csv("/kaggle/working/submission.csv", index=False)
sub.to_csv("/kaggle/working/submission_v4.csv", index=False)
print("Submission saved successfully!")

# Strict Validation Checks
assert list(sub.columns) == ["molecule_id", "smiles"], "Invalid columns!"
assert sub["molecule_id"].equals(sample["molecule_id"]), "Molecule IDs do not match sample_submission!"
assert sub["molecule_id"].is_unique, "Duplicate molecule IDs found!"
assert sub["smiles"].notna().all(), "Null values in smiles column!"
counts = sub["smiles"].str.count(";") + 1
assert (counts == TOPN).all(), f"Candidate counts not exactly {TOPN}: {counts.unique()}"

print(f"\\n=== Verification Summary ===")
print(f"Total Rows: {len(sub)}")
print(f"Candidate counts per row: exactly {TOPN}")
print("First 3 rows:")
for _, r in sub.head(3).iterrows():
    cands = r["smiles"].split(";")
    print(f"  {r['molecule_id']} -> {len(cands)} candidates | Top: {cands[0][:45]}...")
print("\\nAll checks PASSED! Ready for competition evaluation.")
''')

with open('/home/rythamo/some/kaggle_train_v4/pipeline_v4.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Generated /home/rythamo/some/kaggle_train_v4/pipeline_v4.ipynb for Version 15!")
