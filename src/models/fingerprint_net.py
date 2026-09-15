"""Deep Spectrum-to-Fingerprint neural network for candidate molecular retrieval."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem


def featurize_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    num_bins: int = 2000,
    max_mz: float = 1000.0,
) -> np.ndarray:
    """Convert variable peak list into a fixed-size dense feature vector.

    Features:
    1. Binned fragment peaks (0 to max_mz, shape: num_bins)
    2. Binned neutral losses (0 to max_mz, shape: num_bins)
    3. Precursor m/z scalar (shape: 1)
    Total dimension: 2 * num_bins + 1 (default 4001)
    """
    bin_size = max_mz / num_bins
    frag_bins = np.zeros(num_bins, dtype=np.float32)
    loss_bins = np.zeros(num_bins, dtype=np.float32)

    # Square-root intensity transformation
    w_ints = np.sqrt(np.maximum(intensities, 0.0))

    for mz, w in zip(mzs, w_ints):
        # 1. Fragment peak binning
        if 0 < mz < max_mz:
            b_idx = int(mz / bin_size)
            if b_idx < num_bins:
                frag_bins[b_idx] = max(frag_bins[b_idx], w)

        # 2. Neutral loss binning
        neutral_loss = precursor_mz - mz
        if 0 < neutral_loss < max_mz:
            l_idx = int(neutral_loss / bin_size)
            if l_idx < num_bins:
                loss_bins[l_idx] = max(loss_bins[l_idx], w)

    # Normalize vectors by L2 norm
    f_norm = np.linalg.norm(frag_bins)
    if f_norm > 0:
        frag_bins /= f_norm

    l_norm = np.linalg.norm(loss_bins)
    if l_norm > 0:
        loss_bins /= l_norm

    prec_scalar = np.array([precursor_mz / max_mz], dtype=np.float32)
    return np.concatenate([frag_bins, loss_bins, prec_scalar])


from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Pre-initialize generator
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> Optional[np.ndarray]:
    """Generate binary Morgan fingerprint (ECFP4) from SMILES using modern MorganGenerator."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = _MORGAN_GEN.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        return arr
    except Exception:
        return None


def continuous_tanimoto_similarity(pred_fp: np.ndarray, cand_fp: np.ndarray) -> float:
    """Compute continuous Tanimoto similarity between predicted probability vector and candidate bit vector."""
    dot = np.dot(pred_fp, cand_fp)
    denom = np.sum(pred_fp) + np.sum(cand_fp) - dot
    if denom <= 0:
        return 0.0
    return float(dot / denom)


class SpectrumFingerprintNet(nn.Module):
    """Deep Neural Network mapping MS/MS spectrum features to 2048-bit molecular fingerprints."""

    def __init__(self, in_features: int = 4001, out_features: int = 2048, hidden_dim: int = 1024):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.block1(x)
        h2 = self.block2(h1) + h1  # Residual skip connection
        return self.head(h2)
