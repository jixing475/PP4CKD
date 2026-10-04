"""
Evaluation Engine for PP4CKD / PPB3.

Calculates multi-target classification metrics:
  - Top-1 to Top-10 Recall & Precision on Global Target Set (7,676 targets)
  - Top-1 to Top-10 Recall & Precision on CKDdb Renal Disease Subset (623 targets)
  - Micro-averaged Precision-Recall AUC (Micro-AUPR)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import sparse
import torch
from sklearn.metrics import average_precision_score


def compute_topk_metrics_tensor(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k_list: List[int] = list(range(1, 11)),
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute Top-K Recall and Top-K Precision directly on GPU/CPU tensor.

    Parameters
    ----------
    logits : torch.Tensor of shape (B, N)
        Predicted logits.
    targets : torch.Tensor of shape (B, N)
        Ground truth binary activity matrix.
    k_list : list of int
        List of ranks to compute (e.g. 1 to 10).

    Returns
    -------
    tuple of (recalls_dict, precisions_dict)
    """
    max_k = max(k_list)
    topk_indices = torch.topk(logits, k=max_k, dim=-1).indices

    hits = torch.gather(targets, dim=-1, index=topk_indices)
    pos_counts = targets.sum(dim=-1)
    valid_mask = pos_counts > 0

    recalls = {}
    precisions = {}

    if valid_mask.sum() == 0:
        for k in k_list:
            recalls[f"recall_top_{k}"] = 0.0
            precisions[f"precision_top_{k}"] = 0.0
        return recalls, precisions

    hits_valid = hits[valid_mask]
    pos_counts_valid = pos_counts[valid_mask]

    for k in k_list:
        hits_k = hits_valid[:, :k].sum(dim=-1).float()
        rec = (hits_k / pos_counts_valid).mean().item()
        prec = (hits_k / float(k)).mean().item()
        recalls[f"recall_top_{k}"] = rec
        precisions[f"precision_top_{k}"] = prec

    return recalls, precisions


def evaluate_fold_predictions(
    logits_all: np.ndarray,
    targets_csr: sparse.csr_matrix,
    ckd_indices: np.ndarray,
    batch_size: int = 10000,
) -> Dict:
    """
    Evaluate validation fold across global targets and CKDdb subset.

    Parameters
    ----------
    logits_all : np.ndarray of shape (N_samples, N_targets)
    targets_csr : sparse.csr_matrix of shape (N_samples, N_targets)
    ckd_indices : np.ndarray of shape (N_ckd,)
    batch_size : int, default=10000

    Returns
    -------
    dict containing 'overall' and 'ckd_subset' metrics
    """
    n_samples = len(logits_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overall_recalls_accum = {f"recall_top_{k}": [] for k in range(1, 11)}
    overall_precisions_accum = {f"precision_top_{k}": [] for k in range(1, 11)}

    ckd_recalls_accum = {f"recall_top_{k}": [] for k in range(1, 11)}
    ckd_precisions_accum = {f"precision_top_{k}": [] for k in range(1, 11)}

    ckd_indices_tensor = torch.tensor(ckd_indices, dtype=torch.long, device=device)

    for i in range(0, n_samples, batch_size):
        end = min(n_samples, i + batch_size)
        sub_logits = torch.from_numpy(logits_all[i:end]).to(device)
        sub_targets = torch.from_numpy(targets_csr[i:end].toarray()).to(device)

        # 1. Overall Top-K
        recs, precs = compute_topk_metrics_tensor(sub_logits, sub_targets)
        for k in range(1, 11):
            overall_recalls_accum[f"recall_top_{k}"].append(recs[f"recall_top_{k}"])
            overall_precisions_accum[f"precision_top_{k}"].append(precs[f"precision_top_{k}"])

        # 2. CKD Subset Top-K
        sub_ckd_logits = torch.index_select(sub_logits, dim=-1, index=ckd_indices_tensor)
        sub_ckd_targets = torch.index_select(sub_targets, dim=-1, index=ckd_indices_tensor)
        ckd_recs, ckd_precs = compute_topk_metrics_tensor(sub_ckd_logits, sub_ckd_targets)
        for k in range(1, 11):
            ckd_recalls_accum[f"recall_top_{k}"].append(ckd_recs[f"recall_top_{k}"])
            ckd_precisions_accum[f"precision_top_{k}"].append(ckd_precs[f"precision_top_{k}"])

    metrics = {
        "overall": {
            "recalls": {k: float(np.mean(v)) for k, v in overall_recalls_accum.items()},
            "precisions": {k: float(np.mean(v)) for k, v in overall_precisions_accum.items()},
        },
        "ckd_subset": {
            "recalls": {k: float(np.mean(v)) for k, v in ckd_recalls_accum.items()},
            "precisions": {k: float(np.mean(v)) for k, v in ckd_precisions_accum.items()},
        },
    }

    # Micro-AUPR
    try:
        sample_size = min(10000, n_samples)
        sample_logits = logits_all[:sample_size]
        sample_probs = 1.0 / (1.0 + np.exp(-sample_logits))
        sample_targets = targets_csr[:sample_size].toarray()

        micro_aupr = average_precision_score(sample_targets.ravel(), sample_probs.ravel())
        metrics["overall"]["micro_aupr"] = float(micro_aupr)

        ckd_sample_targets = sample_targets[:, ckd_indices]
        ckd_sample_probs = sample_probs[:, ckd_indices]
        ckd_micro_aupr = average_precision_score(ckd_sample_targets.ravel(), ckd_sample_probs.ravel())
        metrics["ckd_subset"]["micro_aupr"] = float(ckd_micro_aupr)
    except Exception:
        metrics["overall"]["micro_aupr"] = 0.0
        metrics["ckd_subset"]["micro_aupr"] = 0.0

    return metrics


def summarize_benchmarks(cv_metrics_path: str, ckd_metrics_path: str, output_json: Optional[str] = None):
    """Generate consolidated academic summary table."""
    with open(cv_metrics_path) as f:
        cv_data = json.load(f)
    with open(ckd_metrics_path) as f:
        ckd_data = json.load(f)

    fps = ["ECFP4", "Fused", "ECFP6", "AtomPair", "Layered", "RDKit", "MHFP6"]
    summary = {}

    print("\n" + "=" * 92)
    print("PP4CKD ChEMBL 36 10-Fold CV Performance (Global Dataset - 7,676 Targets)")
    print("=" * 92)
    print(f"{'Fingerprint':<12} | {'Folds':<6} | {'Top-1 Rec':<12} | {'Top-5 Rec':<12} | {'Top-10 Rec':<12} | {'Top-10 Prec':<12} | {'Micro-AUPR':<10}")
    print("-" * 92)

    for fp in fps:
        folds = cv_data.get(fp, [])
        if not folds:
            continue
        r1 = [f["overall"]["recalls"]["recall_top_1"] for f in folds]
        r5 = [f["overall"]["recalls"]["recall_top_5"] for f in folds]
        r10 = [f["overall"]["recalls"]["recall_top_10"] for f in folds]
        p10 = [f["overall"]["precisions"]["precision_top_10"] for f in folds]
        aupr = [f["overall"].get("micro_aupr", 0.0) for f in folds]

        summary[fp] = {
            "n_folds_completed": len(folds),
            "overall": {
                "recall_top_1": {"mean": float(np.mean(r1)), "std": float(np.std(r1))},
                "recall_top_5": {"mean": float(np.mean(r5)), "std": float(np.std(r5))},
                "recall_top_10": {"mean": float(np.mean(r10)), "std": float(np.std(r10))},
                "precision_top_10": {"mean": float(np.mean(p10)), "std": float(np.std(p10))},
                "micro_aupr": {"mean": float(np.mean(aupr)), "std": float(np.std(aupr))},
            },
        }
        print(f"{fp:<12} | {len(folds):>2}/10  | {np.mean(r1)*100:>5.2f}±{np.std(r1)*100:<4.2f}% | {np.mean(r5)*100:>5.2f}±{np.std(r5)*100:<4.2f}% | {np.mean(r10)*100:>5.2f}±{np.std(r10)*100:<4.2f}% | {np.mean(p10)*100:>5.2f}±{np.std(p10)*100:<4.2f}% | {np.mean(aupr):>8.4f}")

    print("=" * 92)

    print("\n" + "=" * 92)
    print("PP4CKD ChEMBL 36 Kidney Disease Subset Performance (623 CKDdb Targets)")
    print("=" * 92)
    print(f"{'Fingerprint':<12} | {'Folds':<6} | {'Top-1 Rec':<12} | {'Top-5 Rec':<12} | {'Top-10 Rec':<12} | {'Top-10 Prec':<12} | {'Micro-AUPR':<10}")
    print("-" * 92)

    for fp in fps:
        folds = ckd_data.get(fp, [])
        if not folds:
            continue
        r1 = [f["ckd_subset"]["recalls"]["recall_top_1"] for f in folds]
        r5 = [f["ckd_subset"]["recalls"]["recall_top_5"] for f in folds]
        r10 = [f["ckd_subset"]["recalls"]["recall_top_10"] for f in folds]
        p10 = [f["ckd_subset"]["precisions"]["precision_top_10"] for f in folds]
        aupr = [f["ckd_subset"].get("micro_aupr", 0.0) for f in folds]

        summary[fp]["ckd_subset"] = {
            "recall_top_1": {"mean": float(np.mean(r1)), "std": float(np.std(r1))},
            "recall_top_5": {"mean": float(np.mean(r5)), "std": float(np.std(r5))},
            "recall_top_10": {"mean": float(np.mean(r10)), "std": float(np.std(r10))},
            "precision_top_10": {"mean": float(np.mean(p10)), "std": float(np.std(p10))},
            "micro_aupr": {"mean": float(np.mean(aupr)), "std": float(np.std(aupr))},
        }
        print(f"{fp:<12} | {len(folds):>2}/10  | {np.mean(r1)*100:>5.2f}±{np.std(r1)*100:<4.2f}% | {np.mean(r5)*100:>5.2f}±{np.std(r5)*100:<4.2f}% | {np.mean(r10)*100:>5.2f}±{np.std(r10)*100:<4.2f}% | {np.mean(p10)*100:>5.2f}±{np.std(p10)*100:<4.2f}% | {np.mean(aupr):>8.4f}")

    print("=" * 92)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved consolidated summary to: {output_json}\n")


def main():
    parser = argparse.ArgumentParser(description="PP4CKD Benchmark Evaluator")
    parser.add_argument("--cv-metrics", type=str, default="examples/benchmark_results/cv_metrics.json")
    parser.add_argument("--ckd-metrics", type=str, default="examples/benchmark_results/ckd_subset_metrics.json")
    parser.add_argument("--output-json", type=str, default="examples/benchmark_results/benchmark_summary.json")
    args = parser.parse_args()

    summarize_benchmarks(args.cv_metrics, args.ckd_metrics, args.output_json)


if __name__ == "__main__":
    main()
