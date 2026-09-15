"""PyTorch Dataset & Collation for Contrastive Peak Transformer (P0-1).

Features:
- Strict structure-disjoint train/val splitting based on InChIKey14.
- High-efficiency peak list sorting, intensity ranking, and padding collation.
- Dynamic collision energy feature parsing: [mean, min, max, has_ce].
- Data augmentations: peak dropout, intensity jitter, precursor m/z noise.
- Caching of Morgan fingerprints for fast epoch throughput.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.models.peak_transformer import (
    compute_molecule_fingerprint,
    get_instrument_idx,
)


def parse_collision_energy(ce_val) -> Tuple[float, float, float, float]:
    """Parse raw collision energy (list of floats, scalar, or None) into normalized [mean, min, max, has_ce]."""
    if ce_val is None:
        return 0.0, 0.0, 0.0, 0.0
    if isinstance(ce_val, (int, float)):
        if np.isnan(ce_val):
            return 0.0, 0.0, 0.0, 0.0
        v = float(ce_val) / 100.0  # Normalize ~ [0, 1] range for 0-100 eV
        return v, v, v, 1.0
    if isinstance(ce_val, (list, np.ndarray, Sequence)):
        valid = [float(x) for x in ce_val if x is not None and not np.isnan(x)]
        if not valid:
            return 0.0, 0.0, 0.0, 0.0
        mean_ce = (sum(valid) / len(valid)) / 100.0
        min_ce = min(valid) / 100.0
        max_ce = max(valid) / 100.0
        return mean_ce, min_ce, max_ce, 1.0
    return 0.0, 0.0, 0.0, 0.0


class MassSpecDataset(Dataset):
    """Dataset for tandem mass spectra paired with molecular structures."""

    def __init__(
        self,
        df: pl.DataFrame,
        max_peaks: int = 128,
        augment: bool = False,
        peak_dropout_prob: float = 0.1,
        intensity_noise_std: float = 0.05,
    ):
        """Args:
            df: Polars DataFrame containing columns:
                'normalized_smiles', 'inchikey14', 'precursor_mz', 'adduct',
                'instrument_type', 'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities'
            max_peaks: Maximum number of fragment peaks per spectrum.
            augment: Whether to apply data augmentation.
        """
        self.max_peaks = max_peaks
        self.augment = augment
        self.peak_dropout_prob = peak_dropout_prob
        self.intensity_noise_std = intensity_noise_std

        # Extract columns as python/numpy lists for fast index access
        self.smiles = df["normalized_smiles"].to_list()
        self.inchikey14 = df["inchikey14"].to_list()
        self.precursor_mzs = df["precursor_mz"].to_numpy().astype(np.float32)
        self.instruments = [get_instrument_idx(x) for x in df["instrument_type"].to_list()]
        self.ce_features = np.array(
            [parse_collision_energy(x) for x in df["collision_energy_ev"].to_list()],
            dtype=np.float32,
        )

        self.raw_mzs = df["ms2_mzs"].to_list()
        self.raw_ints = df["ms2_normalized_intensities"].to_list()

        # Cache molecular fingerprints (by unique smiles) to avoid recomputing in training loops
        unique_smiles = set(self.smiles)
        self.fp_cache: Dict[str, np.ndarray] = {}
        for smi in unique_smiles:
            fp = compute_molecule_fingerprint(smi, n_bits=2048)
            if fp is not None:
                self.fp_cache[smi] = fp
            else:
                self.fp_cache[smi] = np.zeros(2048, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        mzs_raw = self.raw_mzs[idx]
        ints_raw = self.raw_ints[idx]
        prec_mz = float(self.precursor_mzs[idx])

        mzs = np.array(mzs_raw if mzs_raw is not None else [], dtype=np.float32)
        ints = np.array(ints_raw if ints_raw is not None else [], dtype=np.float32)

        # Apply peak dropout & intensity jitter if augmenting
        if self.augment and len(mzs) > 2:
            # 1. Random peak dropout
            keep_mask = np.random.rand(len(mzs)) > self.peak_dropout_prob
            if np.any(keep_mask):
                mzs = mzs[keep_mask]
                ints = ints[keep_mask]

            # 2. Intensity jitter
            noise = np.random.normal(0.0, self.intensity_noise_std, size=len(ints)).astype(np.float32)
            ints = np.maximum(ints * (1.0 + noise), 0.0)

            # 3. Precursor m/z noise (±5 ppm)
            prec_mz += prec_mz * float(np.random.uniform(-5e-6, 5e-6))

        # Select top peaks by intensity if exceeding max_peaks
        if len(mzs) > self.max_peaks:
            top_idx = np.argpartition(ints, -self.max_peaks)[-self.max_peaks:]
            # Sort selected peaks by m/z
            order = np.argsort(mzs[top_idx])
            mzs = mzs[top_idx][order]
            ints = ints[top_idx][order]
        elif len(mzs) > 0:
            order = np.argsort(mzs)
            mzs = mzs[order]
            ints = ints[order]

        smi = self.smiles[idx]
        fp = self.fp_cache.get(smi, np.zeros(2048, dtype=np.float32))

        return {
            "mzs": mzs,
            "ints": ints,
            "precursor_mz": prec_mz,
            "ce_feature": self.ce_features[idx],
            "instrument_idx": self.instruments[idx],
            "fingerprint": fp,
            "inchikey14": self.inchikey14[idx],
            "smiles": smi,
        }


def collate_spectrum_batch(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length peak lists with padding and attention masks."""
    batch_size = len(batch)

    # Determine maximum number of peaks in this specific batch
    max_batch_peaks = max(len(item["mzs"]) for item in batch)
    max_batch_peaks = max(max_batch_peaks, 1)  # At least 1 for safety

    padded_mzs = np.zeros((batch_size, max_batch_peaks), dtype=np.float32)
    padded_ints = np.zeros((batch_size, max_batch_peaks), dtype=np.float32)
    peak_mask = np.zeros((batch_size, max_batch_peaks), dtype=bool)

    precursor_mzs = np.zeros(batch_size, dtype=np.float32)
    ce_features = np.zeros((batch_size, 4), dtype=np.float32)
    instruments = np.zeros(batch_size, dtype=np.int64)
    fingerprints = np.zeros((batch_size, 2048), dtype=np.float32)
    inchikey14s = []
    smiles_list = []

    for i, item in enumerate(batch):
        n_p = len(item["mzs"])
        if n_p > 0:
            padded_mzs[i, :n_p] = item["mzs"]
            padded_ints[i, :n_p] = item["ints"]
            peak_mask[i, :n_p] = True

        precursor_mzs[i] = item["precursor_mz"]
        ce_features[i] = item["ce_feature"]
        instruments[i] = item["instrument_idx"]
        fingerprints[i] = item["fingerprint"]
        inchikey14s.append(item["inchikey14"])
        smiles_list.append(item["smiles"])

    return {
        "peak_mzs": torch.from_numpy(padded_mzs),
        "peak_intensities": torch.from_numpy(padded_ints),
        "peak_mask": torch.from_numpy(peak_mask),
        "precursor_mzs": torch.from_numpy(precursor_mzs),
        "collision_energies": torch.from_numpy(ce_features),
        "instrument_indices": torch.from_numpy(instruments),
        "fingerprints": torch.from_numpy(fingerprints),
        "inchikey14s": inchikey14s,
        "smiles": smiles_list,
    }


def create_structure_disjoint_datasets(
    train_parquet_path: str = "data/train.parquet",
    val_lib: str = "enveda-np-examples",
    max_train_samples: Optional[int] = None,
    max_peaks: int = 128,
    seed: int = 42,
) -> Tuple[MassSpecDataset, MassSpecDataset]:
    """Create strictly structure-disjoint train and validation datasets.

    Any InChIKey14 present in the validation library (e.g. enveda-np-examples)
    is completely purged from the training set across all other libraries.
    """
    print(f"Reading parquet from {train_parquet_path}...")
    needed_cols = [
        "ingest_lib",
        "normalized_smiles",
        "inchikey14",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy_ev",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]
    df = pl.read_parquet(train_parquet_path, columns=needed_cols)

    # 1. Extract validation set
    val_df = df.filter(pl.col("ingest_lib") == val_lib)
    val_keys: Set[str] = set(val_df["inchikey14"].unique().to_list())
    print(f"Validation set ({val_lib}): {len(val_df)} spectra across {len(val_keys)} unique InChIKey14s.")

    # 2. Extract training set by strictly excluding any validation InChIKey14
    train_df = df.filter(~pl.col("inchikey14").is_in(val_keys))
    print(f"Purged all {len(val_keys)} validation molecules from training data.")
    print(f"Remaining training pool: {len(train_df)} spectra.")

    # Priority to natural product libraries in training
    np_libs = ["gnps", "riken", "massbank", "mona"]
    train_np = train_df.filter(pl.col("ingest_lib").is_in(np_libs))
    train_other = train_df.filter(~pl.col("ingest_lib").is_in(np_libs))

    if max_train_samples is not None and len(train_df) > max_train_samples:
        # Sample proportionally or fill with NP libraries first
        if len(train_np) < max_train_samples:
            needed = max_train_samples - len(train_np)
            sampled_other = train_other.sample(n=needed, seed=seed)
            train_df = pl.concat([train_np, sampled_other])
        else:
            train_df = train_np.sample(n=max_train_samples, seed=seed)
        print(f"Sampled training set to {len(train_df)} spectra.")

    train_ds = MassSpecDataset(train_df, max_peaks=max_peaks, augment=True)
    val_ds = MassSpecDataset(val_df, max_peaks=max_peaks, augment=False)

    return train_ds, val_ds
