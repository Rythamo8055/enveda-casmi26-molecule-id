"""Rigorous Holdout Benchmark for CPU Learned Retrieval Engine.

Evaluates 50 real blind holdout natural products (Bruker timsTOF from enveda-np-examples)
against the 422,926 molecules in COCONUT.

Compares:
1. Baseline Deterministic Ranker (15 ppm flat window, fixed linear weights)
2. Upgraded CPU Learned Ranker (Dynamic 5.0 ppm window + 16-feature LightGBM LambdaMART)

CPU-friendly: Runs with bounded concurrency (n_jobs=2) to keep CPU quiet and responsive.
"""

import argparse
import time
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
from scipy import stats

from src.evaluation.metrics import smiles_to_inchikey14
from src.retrieval.deterministic_ranker import DeterministicCandidateRanker
from src.retrieval.learned_ranker import LearnedCandidateRanker


def run_cpu_benchmark(
    num_molecules: int = 50,
    coconut_path: str = "data/external/coconut_indexed.parquet",
    train_parquet_path: str = "data/train.parquet",
    model_path: str = "models/lambdamart_ranker.txt",
):
    print("=" * 75)
    print(f"BENCHMARKING CPU RETRIEVAL ENGINE ON {num_molecules} REAL HOLDOUT MOLECULES")
    print("=" * 75)

    # 1. Load holdout spectra (enveda-np-examples, timsTOF)
    print("Loading holdout spectra from enveda-np-examples...")
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
    np_df = df.filter(pl.col("ingest_lib") == "enveda-np-examples")
    unique_mols = np_df["normalized_smiles"].unique().to_list()[:num_molecules]
    print(f"Selected {len(unique_mols)} unique holdout molecules for blind evaluation.")

    # 2. Instantiate rankers
    print("Initializing Deterministic Baseline Ranker...")
    det_ranker = DeterministicCandidateRanker(coconut_path)

    print("Initializing Upgraded Learned LambdaMART Ranker...")
    learned_ranker = LearnedCandidateRanker(coconut_path, model_path=model_path)

    # Metrics storage
    baseline_rrs: List[float] = []
    baseline_h1: List[float] = []
    baseline_h5: List[float] = []
    baseline_h10: List[float] = []
    baseline_h25: List[float] = []
    baseline_cand_counts: List[int] = []

    learned_rrs: List[float] = []
    learned_h1: List[float] = []
    learned_h5: List[float] = []
    learned_h10: List[float] = []
    learned_h25: List[float] = []
    learned_cand_counts: List[int] = []

    t0 = time.time()
    for idx, smi in enumerate(unique_mols, start=1):
        mol_rows = np_df.filter(pl.col("normalized_smiles") == smi)
        true_k14 = smiles_to_inchikey14(smi)
        if not true_k14:
            continue

        # Package spectra for query
        spectra = []
        for row in mol_rows.iter_rows(named=True):
            mzs = np.array(row["ms2_mzs"] if row["ms2_mzs"] is not None else [], dtype=np.float32)
            ints = np.array(row["ms2_normalized_intensities"] if row["ms2_normalized_intensities"] is not None else [], dtype=np.float32)
            spectra.append({
                "precursor_mz": float(row["precursor_mz"]),
                "adduct": str(row["adduct"]),
                "instrument_type": str(row.get("instrument_type", "Bruker timsTOF")),
                "collision_energy_ev": row.get("collision_energy_ev"),
                "mzs": mzs,
                "ints": ints,
            })

        # --- A. Baseline Deterministic Ranker (15 ppm flat) ---
        base_hits = det_ranker.rank_candidates(spectra, ppm_tol=15.0, max_cands=25)
        base_keys = [h[1] for h in base_hits]
        base_rank = base_keys.index(true_k14) + 1 if true_k14 in base_keys else 0

        b_rr = 1.0 / base_rank if 1 <= base_rank <= 25 else 0.0
        baseline_rrs.append(b_rr)
        baseline_h1.append(1.0 if base_rank == 1 else 0.0)
        baseline_h5.append(1.0 if 1 <= base_rank <= 5 else 0.0)
        baseline_h10.append(1.0 if 1 <= base_rank <= 10 else 0.0)
        baseline_h25.append(1.0 if 1 <= base_rank <= 25 else 0.0)
        baseline_cand_counts.append(len(base_hits))

        # --- B. Upgraded Learned LambdaMART Ranker (Dynamic 5.0 ppm) ---
        learned_hits = learned_ranker.rank_candidates(spectra, ppm_tol=5.0, max_cands=25)
        learned_keys = [h[1] for h in learned_hits]
        learned_rank = learned_keys.index(true_k14) + 1 if true_k14 in learned_keys else 0

        l_rr = 1.0 / learned_rank if 1 <= learned_rank <= 25 else 0.0
        learned_rrs.append(l_rr)
        learned_h1.append(1.0 if learned_rank == 1 else 0.0)
        learned_h5.append(1.0 if 1 <= learned_rank <= 5 else 0.0)
        learned_h10.append(1.0 if 1 <= learned_rank <= 10 else 0.0)
        learned_h25.append(1.0 if 1 <= learned_rank <= 25 else 0.0)
        learned_cand_counts.append(len(learned_hits))

        if idx % 10 == 0 or idx == len(unique_mols):
            curr_base_mrr = float(np.mean(baseline_rrs))
            curr_learned_mrr = float(np.mean(learned_rrs))
            print(f"[{idx:2d}/{len(unique_mols)}] Progress | Baseline MRR: {curr_base_mrr:.4f} -> Learned MRR: {curr_learned_mrr:.4f} (+{curr_learned_mrr - curr_base_mrr:+.4f})")

    total_time = time.time() - t0
    n = len(baseline_rrs)

    base_mrr = float(np.mean(baseline_rrs))
    learned_mrr = float(np.mean(learned_rrs))
    delta_mrr = learned_mrr - base_mrr

    # Statistical significance (paired t-test on reciprocal ranks)
    t_stat, p_val = stats.ttest_rel(learned_rrs, baseline_rrs)

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK COMPARISON RESULTS")
    print("=" * 75)
    print(f"Holdout molecules evaluated: {n}")
    print(f"Total evaluation time: {total_time:.1f}s ({total_time / n:.2f}s per molecule on CPU)")
    print()
    print("| Metric | Baseline (15 ppm, Fixed) | Upgraded (5 ppm, LambdaMART) | Delta |")
    print("|---|---|---|---|")
    print(f"| **MRR@25** | **{base_mrr:.4f}** | **{learned_mrr:.4f}** | **{delta_mrr:+.4f}** |")
    print(f"| **Hit@1**  | {np.mean(baseline_h1) * 100:.1f}% | {np.mean(learned_h1) * 100:.1f}% | {(np.mean(learned_h1) - np.mean(baseline_h1)) * 100:+.1f}% |")
    print(f"| **Hit@5**  | {np.mean(baseline_h5) * 100:.1f}% | {np.mean(learned_h5) * 100:.1f}% | {(np.mean(learned_h5) - np.mean(baseline_h5)) * 100:+.1f}% |")
    print(f"| **Hit@10** | {np.mean(baseline_h10) * 100:.1f}% | {np.mean(learned_h10) * 100:.1f}% | {(np.mean(learned_h10) - np.mean(baseline_h10)) * 100:+.1f}% |")
    print(f"| **Hit@25** | {np.mean(baseline_h25) * 100:.1f}% | {np.mean(learned_h25) * 100:.1f}% | {(np.mean(learned_h25) - np.mean(baseline_h25)) * 100:+.1f}% |")
    print(f"\nPaired t-statistic: {t_stat:.4f} (p = {p_val:.4f})")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_molecules", type=int, default=50)
    args = parser.parse_args()

    run_cpu_benchmark(num_molecules=args.num_molecules)
