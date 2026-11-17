#!/usr/bin/env python3
"""
Weighted Vanilla Tsetlin Machine (`tmu` TMClassifier) with:
  - Built-in `tmu.data.MNIST` dataset (and optional `tmu.preprocessing.standard_binarizer.binarizer.StandardBinarizer`)
  - Voting threshold T = 1500, Clauses per class = 2000, Epochs = 50
  - Automated Specificity (s) Sweep to identify the optimal s before running the full 50-epoch training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tmu.data import MNIST
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

import tm_ticket.model  # noqa: F401 (applies NumPy 2.x uint32(~0) patch for tmu)
from tm_ticket.model import PrunableTMClassifier
from tm_ticket.sanitization import (
    compute_model_stats,
    length_based_filter,
    rule_based_canonicalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weighted Vanilla TM on tmu.data.MNIST with Specificity Sweep & 50-Epoch Training"
    )
    parser.add_argument("--clauses", type=int, default=2000, help="Clauses per class (default: 2000)")
    parser.add_argument("--T", type=int, default=1500, help="Voting threshold T (default: 1500)")
    parser.add_argument("--s", type=float, default=None, help="Specificity s (if omitted, runs specificity sweep first)")
    parser.add_argument(
        "--s-candidates",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 7.5, 10.0, 15.0],
        help="Candidate specificity values to test in sweep",
    )
    parser.add_argument("--sweep-epochs", type=int, default=8, help="Epochs per candidate in specificity sweep")
    parser.add_argument("--sweep-samples", type=int, default=20000, help="Training samples used during fast s sweep")
    parser.add_argument("--epochs", type=int, default=50, help="Full training epochs for best s (default: 50)")
    parser.add_argument(
        "--binarizer",
        type=str,
        default="tmu_mnist",
        choices=["tmu_mnist", "tmu_standard"],
        help="Use tmu.data.MNIST built-in binarizer ('tmu_mnist') or tmu StandardBinarizer ('tmu_standard')",
    )
    parser.add_argument("--max-bits-per-feature", type=int, default=2, help="Bits per pixel if using tmu_standard")
    parser.add_argument("--platform", type=str, default="CPU", choices=["CPU", "CUDA", "GPU"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/weighted_vanilla_tm_T1500_50ep.json")
    return parser.parse_args()


def load_tmu_mnist(binarizer_mode: str = "tmu_mnist", max_bits: int = 2):
    """Loads MNIST using tmu.data.MNIST (and optionally tmu's StandardBinarizer)."""
    mnist_obj = MNIST()
    if binarizer_mode == "tmu_mnist":
        data = mnist_obj.get()
        X_train = np.ascontiguousarray(data["x_train"], dtype=np.uint32)
        Y_train = np.ascontiguousarray(data["y_train"], dtype=np.uint32)
        X_test = np.ascontiguousarray(data["x_test"], dtype=np.uint32)
        Y_test = np.ascontiguousarray(data["y_test"], dtype=np.uint32)
    else:
        raw = mnist_obj._retrieve_dataset()
        x_tr_flat = raw["x_train"].reshape(raw["x_train"].shape[0], -1)
        x_te_flat = raw["x_test"].reshape(raw["x_test"].shape[0], -1)
        sb = StandardBinarizer(max_bits_per_feature=max_bits)
        X_train = np.ascontiguousarray(sb.fit_transform(x_tr_flat), dtype=np.uint32)
        X_test = np.ascontiguousarray(sb.transform(x_te_flat), dtype=np.uint32)
        Y_train = np.ascontiguousarray(raw["y_train"], dtype=np.uint32)
        Y_test = np.ascontiguousarray(raw["y_test"], dtype=np.uint32)
    return X_train, Y_train, X_test, Y_test


def run_specificity_sweep(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    candidates: List[float],
    clauses: int,
    T: int,
    sweep_epochs: int,
    sweep_samples: int,
    platform: str,
    seed: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates candidate specificity values `s` on a stratified split of `X_train`
    with `T = 1500` to find the specificity that maximizes validation accuracy
    and inspect how `s` shapes clause length and short/empty clause counts.
    """
    print("=" * 78)
    print(
        f"Phase 1: Specificity (s) Sweep | Candidates: {candidates} | "
        f"Clauses/class: {clauses} | T: {T} | Sweep Epochs: {sweep_epochs}"
    )
    print("=" * 78)

    X_sub, X_val, Y_sub, Y_val = train_test_split(
        X_train,
        Y_train,
        train_size=sweep_samples,
        test_size=5000,
        random_state=seed,
        stratify=Y_train,
    )
    X_sub = np.ascontiguousarray(X_sub, dtype=np.uint32)
    Y_sub = np.ascontiguousarray(Y_sub, dtype=np.uint32)
    X_val = np.ascontiguousarray(X_val, dtype=np.uint32)
    Y_val = np.ascontiguousarray(Y_val, dtype=np.uint32)

    sweep_results: List[Dict[str, Any]] = []
    best_s = candidates[0]
    best_val_acc = -1.0

    for s_val in candidates:
        t0 = time.time()
        tm = PrunableTMClassifier(
            number_of_clauses=clauses,
            T=T,
            s=s_val,
            platform=platform,
            weighted_clauses=True,
            seed=seed,
        )
        tm.init(X_sub, Y_sub)

        best_ep_acc = 0.0
        for ep in range(1, sweep_epochs + 1):
            tm.fit(X_sub, Y_sub)
            if ep == sweep_epochs or ep == sweep_epochs // 2:
                acc = tm.evaluate_accuracy(X_val, Y_val)
                best_ep_acc = max(best_ep_acc, acc)

        elapsed = time.time() - t0
        s1_mask, s1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
        _, short_l1 = length_based_filter(tm, s1_mask, min_literals=1)
        _, short_l2 = length_based_filter(tm, s1_mask, min_literals=2)
        stats = compute_model_stats(tm, mask=s1_mask, stage1_counts=s1_counts)

        row = {
            "s": s_val,
            "val_accuracy": round(best_ep_acc, 3),
            "mean_literal_length": stats.mean_literal_length,
            "empty_clauses": s1_counts["empty"],
            "duplicate_clauses": s1_counts["duplicate"],
            "short_clauses_l_le_1": short_l1,
            "short_clauses_l_le_2": short_l2,
            "time_sec": round(elapsed, 2),
        }
        sweep_results.append(row)

        print(
            f"  [s = {s_val:5.1f}] Val Acc: {best_ep_acc:.2f}% | "
            f"Mean Literals: {stats.mean_literal_length:5.2f} | "
            f"Empty: {s1_counts['empty']:4d} | Short(<=2): {short_l2:4d} | "
            f"Time: {elapsed:.1f}s",
            flush=True,
        )

        if best_ep_acc > best_val_acc:
            best_val_acc = best_ep_acc
            best_s = s_val

    print(f"\n>>> Selected Optimal Specificity: s = {best_s} (Val Acc: {best_val_acc:.2f}%)\n")
    return best_s, sweep_results


def main() -> None:
    args = parse_args()

    X_train, Y_train, X_test, Y_test = load_tmu_mnist(
        binarizer_mode=args.binarizer, max_bits=args.max_bits_per_feature
    )
    print(
        f"Loaded tmu MNIST ({args.binarizer}) -> "
        f"X_train: {X_train.shape}, X_test: {X_test.shape}"
    )

    sweep_results: List[Dict[str, Any]] = []
    if args.s is None:
        chosen_s, sweep_results = run_specificity_sweep(
            X_train=X_train,
            Y_train=Y_train,
            candidates=args.s_candidates,
            clauses=args.clauses,
            T=args.T,
            sweep_epochs=args.sweep_epochs,
            sweep_samples=args.sweep_samples,
            platform=args.platform,
            seed=args.seed,
        )
    else:
        chosen_s = args.s

    print("=" * 78)
    print(
        f"Phase 2: Full {args.epochs}-Epoch Training | "
        f"Clauses/class: {args.clauses} (Total: {args.clauses*10}) | "
        f"T: {args.T} | Optimal s: {chosen_s} | Weighted: True"
    )
    print("=" * 78)

    tm = PrunableTMClassifier(
        number_of_clauses=args.clauses,
        T=args.T,
        s=chosen_s,
        platform=args.platform,
        weighted_clauses=True,
        seed=args.seed,
    )
    tm.init(X_train, Y_train)

    epoch_logs: List[Dict[str, Any]] = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm.fit(X_train, Y_train)
        train_time = time.time() - t0

        t1 = time.time()
        scores_test = tm.compute_class_scores(X_test)
        Y_pred = np.argmax(scores_test, axis=1)
        infer_time = time.time() - t1

        test_acc = float(np.mean(Y_pred == Y_test) * 100.0)
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch

        W = tm.get_weights_matrix()
        mean_abs_w = float(np.mean(np.abs(W)))
        max_abs_w = int(np.max(np.abs(W)))

        row = {
            "epoch": epoch,
            "test_accuracy": round(test_acc, 3),
            "best_test_accuracy": round(best_acc, 3),
            "train_time_sec": round(train_time, 2),
            "infer_time_sec": round(infer_time, 2),
            "mean_abs_weight": round(mean_abs_w, 2),
            "max_abs_weight": max_abs_w,
        }
        epoch_logs.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}% @ Ep {best_epoch}) | "
            f"Mean |w|: {mean_abs_w:.2f} (Max |w|: {max_abs_w}) | "
            f"Train: {train_time:.2f}s | Infer: {infer_time:.2f}s",
            flush=True,
        )

    # Final evaluation & post-training clause analysis
    scores_test = tm.compute_class_scores(X_test)
    Y_pred = np.argmax(scores_test, axis=1)
    final_acc = float(np.mean(Y_pred == Y_test) * 100.0)

    dense_stats = compute_model_stats(tm)
    s1_mask, s1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    _, short_l1 = length_based_filter(tm, s1_mask, min_literals=1)
    _, short_l2 = length_based_filter(tm, s1_mask, min_literals=2)
    _, short_l3 = length_based_filter(tm, s1_mask, min_literals=3)

    tm.apply_retention_mask(s1_mask, zero_weights=True)
    sanitized_acc = tm.evaluate_accuracy(X_test, Y_test)
    sanitized_stats = compute_model_stats(tm, mask=s1_mask, stage1_counts=s1_counts)

    class_names = [
        "Zero", "One", "Two", "Three", "Four",
        "Five", "Six", "Seven", "Eight", "Nine",
    ]
    cls_report = classification_report(
        Y_test, Y_pred, target_names=class_names, digits=4, output_dict=True
    )
    cls_report_str = classification_report(
        Y_test, Y_pred, target_names=class_names, digits=4
    )

    print("\n" + "=" * 78)
    print(f"Final 50-Epoch Weighted Vanilla TM Results (T={args.T}, s={chosen_s})")
    print("=" * 78)
    print(f"  Final Test Accuracy (Epoch {args.epochs}):      {final_acc:.2f}%")
    print(f"  Peak Test Accuracy (Epoch {best_epoch}):       {best_acc:.2f}%")
    print(f"  Test Accuracy after Stage 1 Lossless:  {sanitized_acc:.2f}%")
    print(f"  Total Clauses:                         {dense_stats.total_clauses}")
    print(f"  Empty Clauses (l = 0):                 {s1_counts['empty']}")
    print(f"  Duplicate Clauses Merged:              {s1_counts['duplicate']}")
    print(f"  Short Clauses (l <= 1 / 2 / 3):        {short_l1} / {short_l2} / {short_l3}")
    print(f"  Active Clauses after Stage 1:          {sanitized_stats.active_clauses}")
    print(f"  Mean Included Literals per Clause:     {sanitized_stats.mean_literal_length:.2f}")
    print(f"  Dense Model Footprint:                 {dense_stats.model_size_kb:.2f} KB")
    print(f"  Stage 1 Sanitized Footprint:           {sanitized_stats.model_size_kb:.2f} KB")
    print("\nClass-wise Precision / Recall / F1:")
    print(cls_report_str)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            **vars(args),
            "selected_s": chosen_s,
        },
        "specificity_sweep": sweep_results,
        "final_test_accuracy": round(final_acc, 3),
        "peak_test_accuracy": round(best_acc, 3),
        "peak_epoch": best_epoch,
        "stage1_lossless_test_accuracy": round(sanitized_acc, 3),
        "stage1_removed_counts": s1_counts,
        "short_clause_counts": {
            "l_le_1": short_l1,
            "l_le_2": short_l2,
            "l_le_3": short_l3,
        },
        "active_clauses_after_stage1": sanitized_stats.active_clauses,
        "mean_literal_length": sanitized_stats.mean_literal_length,
        "dense_model_size_kb": dense_stats.model_size_kb,
        "stage1_sanitized_size_kb": sanitized_stats.model_size_kb,
        "epoch_history": epoch_logs,
        "classification_report": cls_report,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved full JSON report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
