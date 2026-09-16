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
add_md('''# CASMI 2026: The Master Fusion Engine (Version 18)
### Combining Mass-Shifted Analog Propagation + Calibrated GBDT Ranker + Dual-Channel Protected Gate + In-Silico MetFrag

- **Channel 1 (Class 1 Protection):** Direct library match with protected gate ($\\ge 0.55$) maintaining our 0.189 baseline.
- **Channel 2 (Class 2 Structural Analogs):** Mass-shifted ($\pm 200$ Da) analog propagation across 711,705 structures using fast bit-packed 6,930-bit Tanimoto matrix multiplication.
- **Channel 3 (Candidate Scoring Head):** Calibrated `HistGradientBoostingClassifier` trained with leaderboard-optimized class weights ($W_1 = 0.42$) on 25 ranking features.
- **Channel 4 (In-Silico Peak Explanation):** MetFrag-Lite 1-cut & 2-cut bond cleavage scoring observed peak intensity fractions.
- **Strict Format:** Exactly 25 InChIKey14 deduplicated SMILES per molecule, 0 nulls, 100% compliant.
''')

# Cell 1: Environment & Setup
add_code('''# Cell 1: Environment & Setup
import os, sys, glob, time, pickle, math, subprocess
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
from numba import njit, prange
from tqdm.auto import tqdm

t0 = time.time()

# Offline RDKit installation
whl_files = glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
if whl_files:
    print(f"Installing offline RDKit from: {whl_files[0]}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", whl_files[0]], check=False)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
    print("RDKit loaded successfully!")
except Exception as e:
    HAVE_RDKIT = False
    print(f"RDKit load notice: {e}")

def find(name: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        hits = glob.glob(f"data/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"File {name} not found!")
    return sorted(hits, key=len)[0]

TRAIN = find("train.parquet")
TEST = find("test.parquet")
SAMPLE = find("sample_submission.csv")
RANK_TRAIN = find("rank_train.npz")
COCO_FP = find("coco_fp.npy")
FP_BITS = find("fp_bits.npy")

print(f"Train:      {TRAIN}")
print(f"Test:       {TEST}")
print(f"Rank Train: {RANK_TRAIN}")
print(f"COCO FP:    {COCO_FP}")
''')

# Cell 2: Configuration
add_code('''# Cell 2: Competition Configurations
class CFG:
    PPM_WIN      = 10.0   # neutral-mass window (+-10ppm -> 0.521 Class-2 MRR)
    PPM_FALLBACK = 30.0   # fallback if tight window returns nothing
    INT_FLOOR    = 0.002  # peak intensity floor
    MAX_PEAKS    = 256    # keep top 256 peaks
    MZ_TOL       = 0.01   # Da tolerance
    INT_POWER    = 1.0    # linear intensity power with entropy
    ENT_WEIGHT   = True   # spectral entropy weighting (Li et al. 2021)
    ANALOG_WIN   = 200.0  # +-200 Da mass-shifted analog window
    N_ANALOG     = 80     # maximum analogs per molecule
    SIM_POWER    = 3.0    # sim^3 weighting for analog propagation
    W1           = 0.42   # leaderboard-calibrated Class-1 weight (2.6x raw share)
    GBM = dict(max_depth=4, max_iter=220, learning_rate=0.07, min_samples_leaf=60, l2_regularization=1.0)
    TOPN         = 25     # exactly 25 candidates per test molecule
''')

# Cell 3: Spectral Cleaning & Entropy Kernels
add_code('''# Cell 3: Spectral Cleaning & Entropy Kernels
@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i
            j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        val = it[idx[i]] ** power
        oi[i] = val
        s += val
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi

@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0
    j = 0
    n = len(qmz)
    m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    tot = 0.0
    buf = np.empty(n + m, np.float64)
    b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]; i += 1; b += 1
        elif d > tol:
            buf[b] = cp[j]; j += 1; b += 1
        else:
            buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if -0.001 < shift < 0.001: return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b

@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out

@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])
    return out
''')

# Cell 4: 24-Adduct Accurate Neutral Mass Engine
add_code('''# Cell 4: Accurate Neutral Mass Engine
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']
FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']

ADDUCTS = {
 "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS['Na'] - E),
 "[M+K]+": (1, 1, MASS['K'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
 "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),
 "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
 "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
 "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
 "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
 "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
 "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E), "[2M+NH4]+": (2, 1, NH4 - E),
 "[2M+K]+": (2, 1, MASS['K'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
 "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
 "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}

def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any(): out[m] = (mz[m] * z - d) / n
    return out
''')

# Cell 5: Pool & Precomputed Bit-Packed Multi-Fingerprint Space (Fixed Keys)
add_code('''# Cell 5: Candidate Pool & Bit-Packed Multi-Fingerprints
BITS = np.load(FP_BITS)
_g = {}

def _fp_init():
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[BITS]
        return fp, float(ExactMolWt(m))
    except Exception:
        return None

class Pool:
    def __init__(self, fp, mass, keys, smiles, nbits):
        o = np.argsort(mass)
        self._fp = fp[o]
        self.mass = mass[o]
        self.keys = np.asarray(keys, dtype=object)[o]
        self.smiles = np.asarray(smiles, dtype=object)[o]
        self.nbits = nbits
        self.k2i = {k: i for i, k in enumerate(self.keys)}

    def window(self, t, ppm):
        a = np.searchsorted(self.mass, t * (1 - ppm / 1e6), 'left')
        b = np.searchsorted(self.mass, t * (1 + ppm / 1e6), 'right')
        return np.arange(a, b)

    def fps(self, idx):
        return np.unpackbits(np.asarray(self._fp[idx]), axis=1)[:, :self.nbits]

def build_pool():
    t_p = time.time()
    d = os.path.dirname(COCO_FP)
    cm = pickle.load(open(os.path.join(d, 'coco_meta.pkl'), 'rb'))
    co_fp = np.load(COCO_FP)
    co_mass = np.load(os.path.join(d, 'coco_mass.npy'))
    print(f"Loaded COCONUT pool: {len(co_mass):,} structures.")

    # Rebuild training-set structures at runtime
    tr = pq.read_table(TRAIN, columns=['inchikey14', 'normalized_smiles']).to_pandas()
    tr = tr.dropna().drop_duplicates('inchikey14')
    tr = tr[~tr.inchikey14.isin(set(cm['keys']))]

    print(f"Adding {len(tr):,} non-COCONUT training structures to candidate pool...")
    from multiprocessing import Pool as MPool
    with MPool(4) as mp:
        res = mp.map(fp_and_mass, list(tr.normalized_smiles), chunksize=500)

    ok = [i for i, r in enumerate(res) if r is not None]
    tr_fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
    tr_mass = np.array([res[i][1] for i in ok])
    tr_keys = tr.inchikey14.values[ok]
    tr_smis = tr.normalized_smiles.values[ok]

    fp = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([np.asarray(cm['keys'], dtype=object), tr_keys])
    smis = np.concatenate([np.asarray(cm['smiles'], dtype=object), tr_smis])
    good = np.isfinite(mass)

    p = Pool(fp[good], mass[good], keys[good], smis[good], cm['nbits'])
    print(f"Complete Candidate Pool: {len(p.mass):,} structures in {time.time() - t_p:.1f}s!")
    return p
''')

# Cell 6: Library Loader & Single-Spectrum Representative Builder
add_code('''# Cell 6: Reference Library & Representative Analog Index
def load_library(path):
    t_l = time.time()
    t = pq.read_table(path, columns=['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz',
                                     'ms2_mzs', 'ms2_normalized_intensities'])
    mzc = t.column('ms2_mzs').combine_chunks()
    itc = t.column('ms2_normalized_intensities').combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
    add = np.asarray(t.column('adduct').cast(pa.string()).to_pylist(), dtype=object)
    ik = np.asarray(t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
    smi = np.asarray(t.column('normalized_smiles').cast(pa.string()).to_pylist(), dtype=object)
    nm = neutral_mass(prec, add)
    ok = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')
    best = {}
    for k, s in zip(ik, smi):
        if k and s and k not in best: best[k] = s
    print(f"Library indexed: {len(off)-1:,} spectra across {len(best):,} structures in {time.time() - t_l:.1f}s.")
    return dict(off=off, mz=allmz, it=allin, nm=nm, ik=ik, best=best,
                order=order, snm=nm[order], n_ok=int(ok.sum()))

def lib_window(L, target, tol):
    lo = np.searchsorted(L['snm'][:L['n_ok']], target - tol, 'left')
    hi = np.searchsorted(L['snm'][:L['n_ok']], target + tol, 'right')
    return L['order'][lo:hi]

def build_rep(L):
    npk = np.diff(L['off'])
    best = {}
    ik = L['ik']
    for i in range(len(ik)):
        k = ik[i]
        if k and (k not in best or npk[i] > npk[best[k]]):
            best[k] = i
    rep = np.array(sorted(best.values()))
    nm = L['nm'][rep]
    ok = np.isfinite(nm)
    rep = rep[ok]
    nm = nm[ok]
    key = ik[rep]
    o = np.argsort(nm)
    return rep[o], key[o], nm[o]

def clean_spec(mz, it):
    return _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                  CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)

def lib_sim(L, specs, target):
    cand = lib_window(L, target, target * CFG.PPM_WIN / 1e6)
    if len(cand) == 0: return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spec(mz, it)
        if len(qm) == 0: continue
        sc = search(qm, qp, cand, L['off'], L['mz'], L['it'],
                    CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
        for c, s in zip(cand, sc):
            k = L['ik'][c]
            if s > agg.get(k, -1.0): agg[k] = float(s)
    return agg

def analog_sim(L, specs, target, rep, rep_key, rep_nm):
    lo = np.searchsorted(rep_nm, target - CFG.ANALOG_WIN, 'left')
    hi = np.searchsorted(rep_nm, target + CFG.ANALOG_WIN, 'right')
    cand = rep[lo:hi]
    if len(cand) == 0: return []
    shift = (target - rep_nm[lo:hi]).astype(np.float32)
    ckey = rep_key[lo:hi]
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spec(mz, it)
        if len(qm) == 0: continue
        sc = search_shift(qm, qp, cand, L['off'], L['mz'], L['it'],
                          CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                          CFG.INT_POWER, CFG.ENT_WEIGHT, shift)
        for c, k, s in zip(cand, ckey, sc):
            if s > agg.get(k, -1.0): agg[k] = float(s)
    return sorted(agg.items(), key=lambda x: -x[1])[:CFG.N_ANALOG]
''')

# Cell 7: MetFrag-Lite In-Silico Channel
add_code('''# Cell 7: In-Silico Fragmenter (MetFrag-Lite)
AMU = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196,
           P=30.97376163, S=31.97207100, F=18.99840322, Cl=34.96885268,
           Br=78.9183371, I=126.904473, Na=22.9897692809, K=38.96370668)
H_MASS = AMU['H']

def mol_graph(smi: str):
    if not HAVE_RDKIT: return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n, dtype=np.float64)
    for a in m.GetAtoms():
        sym = a.GetSymbol()
        w[a.GetIdx()] = AMU.get(sym, 0.0) + a.GetTotalNumHs() * H_MASS
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n

def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps

def fragment_masses(smi: str, max_bonds: int = 34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0, dtype=np.float64)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()], dtype=np.float64)
    out = {float(w.sum())}
    for i in range(nb):
        for c in _components(n, bonds, {i}): out.add(float(w[c].sum()))
    for i in range(nb):
        for j in range(i + 1, min(nb, i + 10)):
            for c in _components(n, bonds, {i, j}): out.add(float(w[c].sum()))
    return np.array(sorted(out), dtype=np.float64)

@njit(cache=True, fastmath=True)
def explain_score(frag_masses, peak_mz, peak_w, mode, tol=0.01):
    tot = 0.0
    for i in range(len(peak_w)): tot += peak_w[i]
    if tot <= 0.0 or len(frag_masses) == 0: return 0.0
    explained = 0.0
    for p in range(len(peak_mz)):
        pmz = peak_mz[p]
        matched = False
        for f in range(len(frag_masses)):
            fm = frag_masses[f]
            if abs(fm + 1.007276 - pmz) <= tol or abs(fm - 1.007276 - pmz) <= tol or abs(fm - pmz) <= tol:
                matched = True; break
        if matched: explained += peak_w[p]
    return explained / tot

from multiprocessing import Pool as MPool
def _frag_masses_safe(smi):
    try: return fragment_masses(smi)
    except Exception: return np.zeros(0)

def frag_scores(cand_smiles, specs, mode, workers=4):
    if not HAVE_RDKIT or len(cand_smiles) == 0: return None
    with MPool(workers) as mp:
        frags = mp.map(_frag_masses_safe, cand_smiles, chunksize=8)
    peaks = []
    for mz, it in specs:
        m2, i2 = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)
        peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
    out = np.zeros(len(cand_smiles), np.float32)
    for j, f in enumerate(frags):
        out[j] = max((explain_score(f, a, b, mode=mode, tol=CFG.MZ_TOL) for a, b in peaks), default=0.0)
    return out
''')

# Cell 8: Feature Engineering & Calibrated GBDT Ranker
add_code('''# Cell 8: Calibrated GBDT Candidate Ranker
from sklearn.ensemble import HistGradientBoostingClassifier

def _rank_norm(x):
    o = np.argsort(-x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)

def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)

def rank_features(cand_fp, cand_lib, analog_fp, analog_sim, frag=None):
    nc = cand_fp.shape[0]
    cf = cand_fp.astype(np.float32)
    cs = cf.sum(1)
    lv = np.asarray(cand_lib, np.float32)
    lvmax = float(lv.max()) if nc else 0.0

    if analog_fp is not None and len(analog_sim):
        af = analog_fp.astype(np.float32)
        asum = af.sum(1)
        inter = cf @ af.T
        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)
        w = np.clip(np.asarray(analog_sim, np.float32), 0, None)
        ap = (tan * (w ** CFG.SIM_POWER)[None, :]).max(1)
        a1 = (tan * w[None, :]).max(1)
        best_tan = tan.max(1)
        top_tan = tan[:, 0]
        top_sim = float(w[0])
        mean_tan = (tan * (w ** CFG.SIM_POWER)[None, :]).sum(1) / ((w ** CFG.SIM_POWER).sum() + 1e-9)
    else:
        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32)
        top_sim = 0.0
    apmax = float(ap.max()) if nc else 0.0

    f = [
        lv, _z(lv), _rank_norm(lv), lv - lvmax, np.full(nc, lvmax, np.float32),
        ap, _z(ap), _rank_norm(ap), ap - apmax, np.full(nc, apmax, np.float32),
        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim, np.float32),
        cs, np.full(nc, float(nc), np.float32),
    ]

    if frag is not None:
        fv = np.asarray(frag, np.float32)
        fmax = float(fv.max()) if nc else 0.0
        f += [fv, _z(fv), _rank_norm(fv), fv - fmax, np.full(nc, fmax, np.float32)]
    else:
        f += [np.zeros(nc, np.float32)] * 5

    f += [lv * ap, np.maximum(lv, ap), lv - ap]
    return np.column_stack(f).astype(np.float32)

# Load shipped simulation features and fit ranker
print("Fitting calibrated HistGradientBoosting ranker...")
z = np.load(RANK_TRAIN)
NFEAT = z['X'].shape[1]
W = np.where(z['M'] == 0, CFG.W1, 1.0 - CFG.W1)
RANKER = HistGradientBoostingClassifier(**CFG.GBM)
RANKER.fit(z['X'], z['Y'], sample_weight=W)
print(f"Ranker successfully trained on {z['X'].shape[0]:,} candidate rows x {NFEAT} features (W1={CFG.W1})!")
''')

# Cell 9: Master Inference & Submission Pipeline
add_code('''# Cell 9: Master Inference Loop
pool = build_pool()
L = load_library(TRAIN)
rep, rep_key, rep_nm = build_rep(L)
print(f"Analog reference set: {len(rep):,} representative spectra.")

te = pq.read_table(TEST).to_pandas()
te['nm'] = neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
print(f"Processing {len(te):,} spectra across {len(mols):,} unique test molecules...")

sample = pd.read_csv(SAMPLE)
rows = {}

for gi, (mid, sub) in enumerate(tqdm(mols, desc="Identifying Molecules")):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    if len(nms) == 0:
        continue
    target = float(np.median(nms))
    specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]

    lib_hits = lib_sim(L, specs, target)
    analogs = analog_sim(L, specs, target, rep, rep_key, rep_nm)

    cand = pool.window(target, CFG.PPM_WIN)
    if len(cand) == 0:
        cand = pool.window(target, CFG.PPM_FALLBACK)

    output = []
    seen = set()

    def push(key, smiles):
        if key in seen or not smiles: return
        seen.add(key)
        output.append(smiles)

    # 1. PROTECTED STRONG LIBRARY GATE (Preserve 0.189 Baseline)
    strong_lib = sorted([(k, v) for k, v in lib_hits.items() if v >= 0.55], key=lambda x: -x[1])
    for k, v in strong_lib[:8]:
        push(k, L['best'].get(k))

    # 2. CALIBRATED GBDT RANKER + METFRAG-LITE + ANALOG PROPAGATION
    if len(cand):
        cfp = pool.fps(cand)
        lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
        ids, sims = [], []
        for k, s in analogs:
            i = pool.k2i.get(k, -1)
            if i >= 0:
                ids.append(i)
                sims.append(s)
        afp = pool.fps(np.array(ids)) if ids else None
        
        # Compute in-silico frag scores on candidate pool
        cand_smiles_list = [pool.smiles[c] for c in cand]
        mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))
        fsc = frag_scores(cand_smiles_list, specs, mode)
        
        X = rank_features(cfp, lv, afp, np.array(sims, np.float32), frag=fsc)
        prob = RANKER.predict_proba(X)[:, 1]
        ranked_cand_idx = cand[np.argsort(-prob)]
        
        for c in ranked_cand_idx:
            push(pool.keys[c], pool.smiles[c])
            if len(output) >= CFG.TOPN:
                break

    # 3. Fallback library hits if still under 25
    weak_lib = sorted([(k, v) for k, v in lib_hits.items() if v < 0.55], key=lambda x: -x[1])
    for k, v in weak_lib:
        push(k, L['best'].get(k))
        if len(output) >= CFG.TOPN:
            break

    # 4. Universal fallback
    if len(output) < CFG.TOPN:
        for s in L['best'].values():
            push(f"fallback:{s}", s)
            if len(output) >= CFG.TOPN:
                break

    rows[mid] = ";".join(output[:CFG.TOPN])

# Assemble final submission
final_smiles = [rows.get(mid, "") for mid in sample["molecule_id"]]
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
assert (counts == CFG.TOPN).all(), f"Candidate counts not exactly {CFG.TOPN}: {counts.unique()}"

print(f"\\n=== Master Verification Summary ===")
print(f"Total Rows: {len(sub)}")
print(f"Candidates per row: exactly {CFG.TOPN}")
print(f"All checks PASSED! Version 18 is ready to conquer the leaderboard.")
''')

with open('/home/rythamo/some/kaggle_train_v4/pipeline_v4.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Generated pipeline_v4.ipynb for Version 18 (Fixed) successfully!")
