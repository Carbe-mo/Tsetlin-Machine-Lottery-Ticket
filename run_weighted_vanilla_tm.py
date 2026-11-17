#!/usr/bin/env python3
"""
Standalone Weighted Vanilla Tsetlin Machine (`TMClassifier(weighted_clauses=True)`)
runner on the full MNIST dataset (60,000 train / 10,000 test) using `tmu`.

Logs per-epoch accuracy, training & inference time, clause weight distributions,
literal length statistics, empty/duplicate clause counts, and per-class
Precision / Recall / F1 (matching Table I of the Clause Sanitization paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import classification_report
from tmu.data import MNIST
from tmu.models.classification.vanilla_classifier import TMClassifier

# Apply the NumPy 2.x uint32(~0) compatibility patch for tmu's ClauseBank
import tm_ticket.model  # noqa: F401
from tm_ticket.model import PrunableTMClassifier
from tm_ticket.sanitization import compute_model_stats, rule_based_canonicalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Weighted Vanilla Tsetlin Machine (tmu TMClassifier) on MNIST"
    )
    parser.add_argument("--clauses", type=int, default=2000, help="Clauses per class (default: 2000)")
    parser.add_argument("--T", type=int, default=5000, help="Voting threshold T (default: 5000 = 2.5*N for weighted TM)")
    parser.add_argument("--s", type=float, default=10.0, help="Specificity s (default: 10.0)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--train-samples", type=int, default=60000, help="Number of MNIST training samples (max 60000)")
    parser.add_argument("--test-samples", type=int, default=10000, help="Number of MNIST test samples (max 10000)")
    parser.add_argument("--platform", type=str, default="CPU", choices=["CPU", "CUDA", "GPU"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/weighted_vanilla_tm_mnist.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("Weighted Vanilla Tsetlin Machine (tmu.models.classification.vanilla_classifier.TMClassifier)")
    print(
        f"Dataset: MNIST | Clauses/class: {args.clauses} (Total: {args.clauses * 10}) | "
        f"T: {args.T} | s: {args.s} | Weighted: True | Epochs: {args.epochs} | Platform: {args.platform}"
    )
    print("=" * 78)

    # Load full MNIST dataset via tmu.data.MNIST (booleanized at pixel > 75)
    dataset = MNIST().get()
    X_train = np.ascontiguousarray(dataset["x_train"][: args.train_samples], dtype=np.uint32)
    Y_train = np.ascontiguousarray(dataset["y_train"][: args.train_samples], dtype=np.uint32)
    X_test = np.ascontiguousarray(dataset["x_test"][: args.test_samples], dtype=np.uint32)
    Y_test = np.ascontiguousarray(dataset["y_test"][: args.test_samples], dtype=np.uint32)

    print(f"Loaded MNIST -> X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    # Instantiate Weighted Vanilla TM
    tm = PrunableTMClassifier(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        platform=args.platform,
        weighted_clauses=True,
        boost_true_positive_feedback=1,
        seed=args.seed,
    )
    tm.init(X_train, Y_train)

    epoch_logs = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm.fit(X_train, Y_train)
        train_time = time.time() - t0

        t1 = time.time()
        scores_test = tm.compute_class_scores(X_test)
        Y_pred = np.argmax(scores_test, axis=1)
        infer_time = time.time() - t1

        test_acc = float(np.mean(Y_pred == Y_test) * 100.0)
        best_acc = max(best_acc, test_acc)

        W = tm.get_weights_matrix()
        mean_abs_w = float(np.mean(np.abs(W)))
        max_abs_w = int(np.max(np.abs(W)))

        log_row = {
            "epoch": epoch,
            "test_accuracy": round(test_acc, 3),
            "best_test_accuracy": round(best_acc, 3),
            "train_time_sec": round(train_time, 2),
            "infer_time_sec": round(infer_time, 2),
            "mean_abs_weight": round(mean_abs_w, 2),
            "max_abs_weight": max_abs_w,
        }
        epoch_logs.append(log_row)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%) | "
            f"Mean |w|: {mean_abs_w:.2f} (Max |w|: {max_abs_w}) | "
            f"Train: {train_time:.2f}s | Infer: {infer_time:.2f}s",
            flush=True,
        )

    # Final evaluation & class-wise metrics
    scores_test = tm.compute_class_scores(X_test)
    Y_pred = np.argmax(scores_test, axis=1)
    final_acc = float(np.mean(Y_pred == Y_test) * 100.0)

    # Also verify exact match with tmu's native TMClassifier.predict on a 500-sample slice
    native_pred_slice = tm.predict(X_test[:500])
    agreement = float(np.mean(native_pred_slice == Y_pred[:500]) * 100.0)

    # Inspect trained clause bank statistics (before and after Stage 1 lossless sanitization)
    dense_stats = compute_model_stats(tm)
    stage1_mask, stage1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    tm.apply_retention_mask(stage1_mask, zero_weights=True)
    sanitized_acc = tm.evaluate_accuracy(X_test, Y_test)
    sanitized_stats = compute_model_stats(tm, mask=stage1_mask, stage1_counts=stage1_counts)

    class_names = [
        "Zero", "One", "Two", "Three", "Four",
        "Five", "Six", "Seven", "Eight", "Nine"
    ]
    cls_report = classification_report(
        Y_test, Y_pred, target_names=class_names, digits=4, output_dict=True
    )
    cls_report_str = classification_report(
        Y_test, Y_pred, target_names=class_names, digits=4
    )

    print("\n" + "=" * 78)
    print("Final Weighted Vanilla TM Results on MNIST")
    print("=" * 78)
    print(f"  Final Test Accuracy (Dense):           {final_acc:.2f}%")
    print(f"  Test Accuracy after Stage 1 Lossless:  {sanitized_acc:.2f}%")
    print(f"  Native tmu.predict() Agreement:        {agreement:.2f}%")
    print(f"  Total Clauses:                         {dense_stats.total_clauses}")
    print(f"  Empty Clauses (l = 0):                 {stage1_counts['empty']}")
    print(f"  Duplicate Clauses Merged:              {stage1_counts['duplicate']}")
    print(f"  Contradictory Clauses:                 {stage1_counts['contradictory']}")
    print(f"  Active Clauses after Stage 1:          {sanitized_stats.active_clauses}")
    print(f"  Mean Included Literals per Clause:     {sanitized_stats.mean_literal_length:.2f}")
    print(f"  Dense Model Footprint:                 {dense_stats.model_size_kb:.2f} KB")
    print(f"  Stage 1 Sanitized Footprint:           {sanitized_stats.model_size_kb:.2f} KB")
    print("\nClass-wise Precision / Recall / F1:")
    print(cls_report_str)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "config": vars(args),
        "final_test_accuracy": round(final_acc, 3),
        "stage1_lossless_test_accuracy": round(sanitized_acc, 3),
        "native_tmu_predict_agreement_pct": round(agreement, 2),
        "dense_model_size_kb": dense_stats.model_size_kb,
        "stage1_sanitized_size_kb": sanitized_stats.model_size_kb,
        "stage1_removed_counts": stage1_counts,
        "active_clauses_after_stage1": sanitized_stats.active_clauses,
        "mean_literal_length": sanitized_stats.mean_literal_length,
        "epoch_history": epoch_logs,
        "classification_report": cls_report,
    }
    out_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Saved detailed JSON report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
