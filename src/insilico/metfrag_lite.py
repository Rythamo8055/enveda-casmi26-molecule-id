"""In-Silico Fragmentation Engine (MetFrag-Lite).
Performs 1-cut and 2-cut bond breaking on candidate molecular graphs
to compute observed MS2 peak intensity coverage fraction.
"""
import numpy as np
from numba import njit
try:
    from rdkit import Chem
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False

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
