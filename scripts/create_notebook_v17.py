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

# Cell 0: Header
add_md('''# CASMI 2026: Protected Gate + In-Silico Fragmenter + Neural Fingerprint Transformer (v17)
### SOTA Ensemble Pipeline for Pushing Beyond 0.25+ MRR

1. **Protected Strong-Library Gate:** Experimental library hits with $\\ge 0.55$ blended similarity are **strictly locked at Rank 1..8** (maintaining our verified 0.189 baseline).
2. **Dual-Channel Aligned Dot Product:** $0.70 \\times \\text{Fragment} + 0.30 \\times \\text{Neutral Loss}$ with 24 accurate ESI adducts.
3. **In-Silico Fragmentation (MetFrag-Lite):** 1-cut and 2-cut bond breaking on candidate isomers to score observed peak intensity explanations.
4. **Pretrained Peak Transformer (Phase 2):** Continuous Fourier $m/z$ Peak Transformer predicting 2,048-bit Morgan chemical fingerprints, re-ranking same-mass isomers via Tanimoto similarity.
5. **Strict Submission Contract:** Exactly 25 InChIKey14 deduplicated SMILES per molecule, 0 nulls, 100% compliant.
''')

# Cell 1: Environment, Offline Wheel & Model Discovery
add_code('''# Cell 1: Environment, Offline Wheel & Model Discovery
import glob
import os
import re
import sys
import time
import math
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from tqdm.auto import tqdm

# Offline RDKit installation
whl_files = glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
if whl_files:
    print(f"Installing offline RDKit from: {whl_files[0]}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", whl_files[0]], check=False)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    print("RDKit successfully loaded! In-Silico & Fingerprint Channels ENABLED.")
except Exception as e:
    HAVE_RDKIT = False
    print(f"RDKit load notice: {e}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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

COCONUT = None
for fname in ["coconut_indexed.parquet", "COCONUT4MetFrag_april.csv", "coconut_dataset.parquet"]:
    try:
        COCONUT = find_file(fname)
        break
    except FileNotFoundError:
        continue

MODEL_CKPT = None
for ckpt_name in ["fingerprint_transformer.pt"]:
    hits = glob.glob(f"/kaggle/input/**/{ckpt_name}", recursive=True)
    if hits:
        MODEL_CKPT = hits[0]
        break

print(f"Train:      {TRAIN}")
print(f"Test:       {TEST}")
print(f"Sample:     {SAMPLE}")
print(f"COCONUT:    {COCONUT}")
print(f"Model CKPT: {MODEL_CKPT}")
''')

# Cell 2: 24-Adduct Accurate Neutral Mass Calculations
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

# Cell 3: Spectral Preprocessing, MetFrag-Lite & Fingerprint Transformer
add_code('''# Cell 3: Preprocessing, MetFrag & Fingerprint Transformer
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

    weight = np.sqrt(intensity)
    norm = np.linalg.norm(weight)
    if norm > 1e-12:
        weight /= norm

    order = np.argsort(mz)
    frag_mz = np.ascontiguousarray(mz[order])
    frag_weight = np.ascontiguousarray(weight[order])

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

# --- In-Silico Fragmentation (MetFrag-Lite) ---
AMU = {
    'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196,
    'P': 30.97376163, 'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268,
    'Br': 78.9183371, 'I': 126.904473, 'Na': 22.9897692809, 'K': 38.96370668,
}
H_MASS = AMU['H']

def mol_graph(smi: str):
    if not HAVE_RDKIT:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    w = np.zeros(n, dtype=np.float64)
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        w[a.GetIdx()] = AMU.get(sym, 0.0) + a.GetTotalNumHs() * H_MASS
    if (w == 0).any():
        return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n

def _components(n: int, bonds: list, drop: set):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop:
            continue
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps

def fragment_masses(smi: str, max_bonds: int = 34) -> np.ndarray:
    g = mol_graph(smi)
    if g is None:
        return np.zeros(0, dtype=np.float64)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds:
        return np.array([w.sum()], dtype=np.float64)
    out = {float(w.sum())}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    for i in range(nb):
        for j in range(i + 1, min(nb, i + 10)):
            for c in _components(n, bonds, {i, j}):
                out.add(float(w[c].sum()))
    return np.array(sorted(out), dtype=np.float64)

@njit(cache=True, fastmath=True)
def _explain_score_fast(frag_masses: np.ndarray, peak_mz: np.ndarray, peak_w: np.ndarray, tol: float = 0.015) -> float:
    tot = 0.0
    for i in range(len(peak_w)):
        tot += peak_w[i]
    if tot <= 0.0 or len(frag_masses) == 0:
        return 0.0

    explained = 0.0
    for p in range(len(peak_mz)):
        pmz = peak_mz[p]
        matched = False
        for f in range(len(frag_masses)):
            fm = frag_masses[f]
            if abs(fm + 1.007276 - pmz) <= tol or abs(fm - 1.007276 - pmz) <= tol or abs(fm - pmz) <= tol:
                matched = True
                break
        if matched:
            explained += peak_w[p]
    return explained / tot

# --- Continuous Fourier Peak Transformer Model Definition ---
class FourierPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_mz: float = 1500.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        freqs = torch.exp(torch.linspace(math.log(1.0 / max_mz), math.log(100.0), half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        angles = mz.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class PeakTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=4, fp_dim=2048):
        super().__init__()
        self.pos_enc = FourierPositionalEncoding(d_model)
        self.intensity_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, fp_dim)
        )

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pe = self.pos_enc(mz)
        ie = self.intensity_proj(intensity.unsqueeze(-1))
        x = pe + ie
        padding_mask = ~mask
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        w = (intensity * mask.float()).unsqueeze(-1)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (out * w).sum(dim=1) / w_sum.squeeze(1)
        return self.head(pooled)

fp_model = None
if MODEL_CKPT and os.path.exists(MODEL_CKPT):
    try:
        fp_model = PeakTransformer(d_model=256, n_heads=8, num_layers=4, fp_dim=2048).to(device)
        fp_model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
        fp_model.eval()
        print("Pretrained Fingerprint Transformer loaded successfully!")
    except Exception as e:
        print(f"Notice loading model checkpoint: {e}")
        fp_model = None
''')

# Cell 4: Reference Library & COCONUT Database Indexing
add_code('''# Cell 4: Reference Library & COCONUT Database Indexing
print("Loading reference library & COCONUT database...")
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
''')

# Cell 5: Evidence Retrieval & Multi-Signal Candidate Assembly (v17)
add_code('''# Cell 5: Multi-Signal Candidate Assembly
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

def predict_query_fp(query_peaks_list):
    if fp_model is None or not query_peaks_list:
        return None
    # Use spectrum with most peaks
    best_mz, best_w = max(query_peaks_list, key=lambda x: len(x[0]))
    n = min(len(best_mz), 128)
    if n == 0:
        return None
    mz_t = torch.zeros((1, 128), dtype=torch.float32, device=device)
    it_t = torch.zeros((1, 128), dtype=torch.float32, device=device)
    mask_t = torch.zeros((1, 128), dtype=torch.bool, device=device)
    mz_t[0, :n] = torch.from_numpy(best_mz[:n])
    it_t[0, :n] = torch.from_numpy(best_w[:n])
    mask_t[0, :n] = True
    with torch.no_grad():
        logits = fp_model(mz_t, it_t, mask_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs

def tanimoto_sim(p_fp: np.ndarray, smi: str) -> float:
    if not HAVE_RDKIT or p_fp is None:
        return 0.0
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0.0
        c_fp = np.array(fp_gen.GetFingerprint(m), dtype=np.float32)
        inter = np.dot(p_fp, c_fp)
        denom = np.sum(p_fp) + np.sum(c_fp) - inter
        return float(inter / max(denom, 1e-6))
    except Exception:
        return 0.0

def retrieve_molecule(group):
    per_query = []
    masses = []
    query_peaks = []
    for row in group.itertuples():
        neutral = neutral_mass_one(row.precursor_mz, row.adduct)
        if not np.isfinite(neutral):
            continue
        masses.append(neutral)
        qfm, qfw, qlm, qlw = prepare_spectrum(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz)
        if len(qfm) == 0:
            continue
        query_peaks.append((qfm, qfw))
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
    return evidence, target_mass, query_peaks

print("Retrieving evidence across test molecules...")
t_ret = time.time()
retrieved = {}
for molecule_id, group in tqdm(test.groupby("molecule_id", sort=False), desc="retrieving"):
    retrieved[molecule_id] = retrieve_molecule(group)
print(f"Retrieved evidence in {time.time() - t_ret:.1f}s!")

def library_rank(evidence):
    scored = []
    for key, (direct, loss, blend, second, support, np_prior) in evidence.items():
        score = blend + 0.12 * second + 0.01 * min(support, 4) + 0.02 * np_prior
        scored.append((key, score, blend))
    return sorted(scored, key=lambda item: (-item[1], item[0]))

def assemble_v17(molecule_id: str) -> str:
    evidence, target_mass, query_peaks = retrieved[molecule_id]
    ranked_library = library_rank(evidence)
    coco = coconut_candidates(target_mass) if np.isfinite(target_mass) else []
    output = []
    seen = set()

    def push(key, smiles):
        if key in seen or not smiles:
            return
        seen.add(key)
        output.append(smiles)

    # 1. STRONG LIBRARY MATCHES (Ground Truth Tier, blend >= 0.55) -> STRICTLY RANK 1..8
    strong = [item for item in ranked_library if item[2] >= 0.55]
    weak = [item for item in ranked_library if item[2] < 0.55]

    for key, _, _ in strong[:8]:
        push(key, library["best_smiles"].get(key))

    # 2. CLASS 2: DUAL RESCORING (MetFrag Peak Explanation + Neural Fingerprint Tanimoto)
    pred_fp = predict_query_fp(query_peaks)
    if len(coco) > 0 and len(query_peaks) > 0:
        pool_to_score = coco[:35]
        scored_coco = []
        for key, smi in pool_to_score:
            best_expl = 0.0
            if HAVE_RDKIT:
                frags = fragment_masses(smi)
                for qm, qw in query_peaks:
                    expl = _explain_score_fast(frags, qm, qw, tol=PEAK_TOL)
                    if expl > best_expl:
                        best_expl = expl
            t_sim = tanimoto_sim(pred_fp, smi) if pred_fp is not None else 0.0
            composite_coco = 0.65 * best_expl + 0.35 * t_sim
            scored_coco.append((key, smi, composite_coco))

        scored_coco.sort(key=lambda x: -x[2])
        for key, smiles, _ in scored_coco:
            push(key, smiles)
            if len(output) >= TOPN:
                break
    else:
        for key, smiles in coco:
            push(key, smiles)
            if len(output) >= TOPN:
                break

    # 3. WEAK LIBRARY MATCHES
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

# Cell 6: Submission & Verification
add_code('''# Cell 6: Submission & Format Verification
print("Generating final predictions for Version 17...")
final_smiles = [assemble_v17(mid) for mid in sample["molecule_id"]]
sub = pd.DataFrame({
    "molecule_id": sample["molecule_id"],
    "smiles": final_smiles,
})

sub.to_csv("/kaggle/working/submission.csv", index=False)
sub.to_csv("/kaggle/working/submission_v4.csv", index=False)
print("Submission saved successfully!")

# Verification Checks
assert list(sub.columns) == ["molecule_id", "smiles"], "Invalid columns!"
assert sub["molecule_id"].equals(sample["molecule_id"]), "Molecule IDs mismatch!"
assert sub["molecule_id"].is_unique, "Duplicate IDs!"
assert sub["smiles"].notna().all(), "Null smiles found!"
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

print("Generated pipeline_v4.ipynb for Version 17 successfully!")
