#!/usr/bin/env python3
"""
Main Experiment Runner for TM-Ticket:
Margin-Aware Clause Sanitization & Multiple Lottery Ticket Discovery in Tsetlin Machines.

Usage examples:
  1. Fast local verification (runs in ~2-3 minutes on CPU):
     python run_experiments.py --preset fast

  2. Standard MNIST benchmark (for laptop / single node):
     python run_experiments.py --preset standard

  3. Full publication suite on coder.uia.no (multi-core CPU or CUDA):
     python run_experiments.py --preset full --platform CUDA --clauses 2000 --T 1500 --s 10.0
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tmu.data import MNIST

from tm_ticket.lottery import (
    _init_worker_data,
    compare_optimization_methods,
    discover_multiple_disjoint_tickets,
    run_iterative_lottery_ticket,
    train_dense_tm_with_checkpoints,
)
from tm_ticket.plotting import generate_all_plots


def load_dataset(
    dataset_name: str = "mnist",
    train_samples: int = 50000,
    val_samples: int = 10000,
    test_samples: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads booleanized MNIST (via tmu.data.MNIST) or Fashion-MNIST / KMNIST
    (via torchvision fallback) and creates a stratified Train / Val / Test split.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        data = MNIST().get()
        X_tr_full = data["x_train"].astype(np.uint32)
        Y_tr_full = data["y_train"].astype(np.uint32)
        X_te_full = data["x_test"].astype(np.uint32)
        Y_te_full = data["y_test"].astype(np.uint32)
    else:
        import torchvision
        root_dir = Path("./data_cache")
        if dataset_name in ("fmnist", "fashion_mnist"):
            ds_tr = torchvision.datasets.FashionMNIST(root=root_dir, train=True, download=True)
            ds_te = torchvision.datasets.FashionMNIST(root=root_dir, train=False, download=True)
        elif dataset_name in ("kmnist", "kuzushiji_mnist"):
            ds_tr = torchvision.datasets.KMNIST(root=root_dir, train=True, download=True)
            ds_te = torchvision.datasets.KMNIST(root=root_dir, train=False, download=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        x_tr_raw = ds_tr.data.numpy()
        x_te_raw = ds_te.data.numpy()
        X_tr_full = np.where(x_tr_raw.reshape(x_tr_raw.shape[0], -1) > 75, 1, 0).astype(np.uint32)
        Y_tr_full = ds_tr.targets.numpy().astype(np.uint32)
        X_te_full = np.where(x_te_raw.reshape(x_te_raw.shape[0], -1) > 75, 1, 0).astype(np.uint32)
        Y_te_full = ds_te.targets.numpy().astype(np.uint32)

    # Create stratified train / validation split from the 60,000 training set
    X_tr_split, X_val_split, Y_tr_split, Y_val_split = train_test_split(
        X_tr_full,
        Y_tr_full,
        test_size=10000,
        random_state=seed,
        stratify=Y_tr_full,
    )

    if train_samples < X_tr_split.shape[0]:
        X_tr_split, _, Y_tr_split, _ = train_test_split(
            X_tr_split,
            Y_tr_split,
            train_size=train_samples,
            random_state=seed,
            stratify=Y_tr_split,
        )

    if val_samples < X_val_split.shape[0]:
        X_val_split, _, Y_val_split, _ = train_test_split(
            X_val_split,
            Y_val_split,
            train_size=val_samples,
            random_state=seed,
            stratify=Y_val_split,
        )

    if test_samples < X_te_full.shape[0]:
        X_te_full, _, Y_te_full, _ = train_test_split(
            X_te_full,
            Y_te_full,
            train_size=test_samples,
            random_state=seed,
            stratify=Y_te_full,
        )

    return (
        np.ascontiguousarray(X_tr_split, dtype=np.uint32),
        np.ascontiguousarray(Y_tr_split, dtype=np.uint32),
        np.ascontiguousarray(X_val_split, dtype=np.uint32),
        np.ascontiguousarray(Y_val_split, dtype=np.uint32),
        np.ascontiguousarray(X_te_full, dtype=np.uint32),
        np.ascontiguousarray(Y_te_full, dtype=np.uint32),
    )


def save_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            clean_r = {
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in r.items()
            }
            writer.writerow(clean_r)


def write_markdown_report(results: Dict[str, Any], out_dir: Path) -> Path:
    report_path = out_dir / "SUMMARY_REPORT.md"
    cfg = results["config"]
    dense = results["disjoint_tickets"]["dense_baseline"]
    disj = results["disjoint_tickets"]["disjoint_tickets"]
    iter_rows = results["iterative_lottery"]["iterative_tickets"]
    comp_rows = results["method_comparison"]["method_comparison"]

    lines = [
        "# TM-Ticket: Multiple Lottery Tickets & Clause Optimization Results",
        "",
        "## 1. Configuration & Dense Tsetlin Machine Baseline",
        f"- **Dataset:** `{cfg['dataset'].upper()}` "
        f"(Train: {cfg['train_samples']}, Val: {cfg['val_samples']}, Test: {cfg['test_samples']})",
        f"- **Clauses per class ($N$):** {cfg['clauses']} "
        f"(Total clauses: {dense['total_clauses']})",
        f"- **Hyperparameters:** $T = {cfg['T']}$, $s = {cfg['s']}$, "
        f"Weighted Clauses = `{cfg['weighted']}`, Platform = `{cfg['platform']}`",
        f"- **Dense TM Test Accuracy:** **{dense['test_accuracy']:.2f}%** "
        f"(Val: {dense['val_accuracy']:.2f}%, Footprint: {dense['model_size_kb']:.2f} KB)",
        f"- **Stage 1 Lossless Canonicalization Removed:** "
        f"Empty = {dense['stage1_removed']['empty']}, "
        f"Duplicates Merged = {dense['stage1_removed']['duplicate']}, "
        f"Contradictory = {dense['stage1_removed']['contradictory']}, "
        f"Short ($\\ell \\le 1$) = {dense['short_removed']}",
        "",
        "---",
        "",
        f"## 2. Multiple Mutually Disjoint Winning Tickets from One Trained TM "
        f"(Each {results['disjoint_tickets']['ticket_ratio']*100:.0f}% of Bank, 0% Overlap)",
        "",
        "Every ticket below is extracted from the **same** trained TM with **strictly zero overlap** "
        "($\\mathcal{T}_a \\cap \\mathcal{T}_b = \\emptyset$, pairwise Jaccard similarity = `0.000`):",
        "",
        "| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\\Theta_{e_r}$) + Retrain (%) | Fine-Tune (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for t in disj:
        lines.append(
            f"| **{t['ticket_id']}** | {t['active_clauses']} | {t['model_size_kb']:.2f} | "
            f"{t['compression_ratio']:.2f}x | {t['overlap_with_previous_tickets']} | "
            f"{t['oneshot_renorm_test_acc']:.2f}% | {t['rewind_early_test_acc']:.2f}% | "
            f"{t['finetune_test_acc']:.2f}% | **{t['winning_ticket_test_acc']:.2f}%** | "
            f"**{t['cumulative_ensemble_test_acc']:.2f}%** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Iterative Lottery-Ticket Pruning Spectrum (IMP-TM)",
            "",
            "| Round | Retained (%) | Active Clauses | Size (KB) | One-Shot (%) | Rewind $\\Theta_{e_r}$ (%) | Rewind $\\Theta_0$ (%) | Fine-Tune (%) | Random State Re-init (%) | Random Mask (%) | Scratch Small TM (%) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for r in iter_rows:
        lines.append(
            f"| {r['round']} | {r['actual_retention_ratio']*100:.1f}% | {r['active_clauses']} | "
            f"{r['model_size_kb']:.2f} | {r['oneshot_renorm_test_acc']:.2f}% | "
            f"**{r['rewind_early_test_acc']:.2f}%** | {r['rewind_init_test_acc']:.2f}% | "
            f"{r['finetune_test_acc']:.2f}% | {r['random_reinit_test_acc']:.2f}% | "
            f"{r['random_mask_retrain_test_acc']:.2f}% | {r['scratch_small_test_acc']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Head-to-Head Comparison of All Clause Optimization Techniques",
            "",
            "| Budget (%) | Clause Optimization Method | Active Clauses | Size (KB) | Mean Literals | One-Shot Test Acc (%) | Lottery-Ticket Retrained Acc (%) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )

    for c in comp_rows:
        lines.append(
            f"| {c['budget_ratio']*100:.0f}% | {c['method']} | {c['active_clauses']} | "
            f"{c['model_size_kb']:.2f} | {c['mean_literal_length']:.1f} | "
            f"{c['oneshot_best_test_acc']:.2f}% | **{c['lottery_ticket_retrain_test_acc']:.2f}%** |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TM-Ticket: Clause Sanitization & Multiple Lottery Ticket Discovery for Tsetlin Machines"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="standard",
        choices=["fast", "standard", "full"],
        help="Experiment scale preset: 'fast' (quick verification), 'standard' (single machine), 'full' (coder.uia.no server scale).",
    )
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "kmnist"])
    parser.add_argument("--clauses", type=int, default=None, help="Number of clauses per class (N)")
    parser.add_argument("--T", type=int, default=None, help="Voting threshold T")
    parser.add_argument("--s", type=float, default=5.0, help="Specificity parameter s (default: 5.0 from specificity sweep)")
    parser.add_argument("--max-included-literals", type=int, default=None, help="Optional cap on included literals per clause (e.g. 32)")
    parser.add_argument("--epochs", type=int, default=None, help="Dense training epochs")
    parser.add_argument("--retrain-epochs", type=int, default=None, help="Retraining epochs per ticket")
    parser.add_argument("--rewind-epoch", type=int, default=1, help="Early rewind checkpoint epoch e_r")
    parser.add_argument("--num-disjoint-tickets", type=int, default=4, help="Number of disjoint tickets to extract")
    parser.add_argument("--disjoint-ticket-ratio", type=float, default=0.20, help="Clause ratio per disjoint ticket")
    parser.add_argument("--train-samples", type=int, default=None, help="Training sample count")
    parser.add_argument("--val-samples", type=int, default=None, help="Validation sample count")
    parser.add_argument("--test-samples", type=int, default=None, help="Test sample count")
    parser.add_argument("--platform", type=str, default="CPU", choices=["CPU", "CUDA", "GPU"])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel CPU worker processes for ticket retraining (default: up to 16 on multi-core servers)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply preset defaults unless explicitly overridden on CLI
    if args.preset == "fast":
        clauses = args.clauses or 400
        T = args.T or 300
        epochs = args.epochs or 6
        retrain_epochs = args.retrain_epochs or 4
        train_samples = args.train_samples or 12000
        val_samples = args.val_samples or 3000
        test_samples = args.test_samples or 5000
        schedule = [1.0, 0.70, 0.50, 0.30, 0.20, 0.10]
        comp_budgets = [0.50, 0.30, 0.20, 0.10]
    elif args.preset == "standard":
        clauses = args.clauses or 1000
        T = args.T or 750
        epochs = args.epochs or 10
        retrain_epochs = args.retrain_epochs or 5
        train_samples = args.train_samples or 30000
        val_samples = args.val_samples or 5000
        test_samples = args.test_samples or 10000
        schedule = [1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]
        comp_budgets = [0.50, 0.30, 0.20, 0.10]
    else:  # full (coder.uia.no server preset: 50 epochs dense + 50 epochs retrain per lottery ticket)
        clauses = args.clauses or 2000
        T = args.T or 1500
        epochs = args.epochs or 50
        retrain_epochs = args.retrain_epochs or 50
        train_samples = args.train_samples or 50000
        val_samples = args.val_samples or 10000
        test_samples = args.test_samples or 10000
        schedule = [1.0, 0.80, 0.60, 0.40, 0.30, 0.20, 0.15, 0.10]
        comp_budgets = [0.50, 0.30, 0.20, 0.10]

    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else (16 if cpu_count >= 32 else 1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"TM-Ticket Experiment Suite | Preset: {args.preset.upper()} | Dataset: {args.dataset.upper()}")
    print(
        f"Clauses/class: {clauses} (Total: {clauses*10}) | T={T} | s={args.s} | "
        f"Epochs={epochs} (Rewind e_r={args.rewind_epoch}, Retrain={retrain_epochs}) | "
        f"Platform={args.platform} | Parallel Workers={workers}"
    )
    print("=" * 78)

    t_start = time.time()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(
        dataset_name=args.dataset,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        seed=args.seed,
    )
    print(
        f"Loaded {args.dataset.upper()}: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}",
        flush=True,
    )

    # 1. Train Dense TM & Save Checkpoints (Theta_0, Theta_{e_r}, Theta_final)
    tm, checkpoints, train_history = train_dense_tm_with_checkpoints(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        number_of_clauses=clauses,
        T=T,
        s=args.s,
        epochs=epochs,
        rewind_epoch=args.rewind_epoch,
        platform=args.platform,
        weighted_clauses=True,
        max_included_literals=args.max_included_literals,
        seed=args.seed,
        verbose=True,
    )

    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker_data,
            initargs=(X_train, Y_train, X_val, Y_val, X_test, Y_test),
        )
        if workers > 1
        else None
    )

    try:
        # 2. Discover Multiple Mutually Disjoint Winning Tickets from One Trained TM
        disjoint_res = discover_multiple_disjoint_tickets(
            tm=tm,
            checkpoints=checkpoints,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            num_tickets=args.num_disjoint_tickets,
            ticket_ratio=args.disjoint_ticket_ratio,
            retrain_epochs=retrain_epochs,
            min_literals=1,
            scoring_method="mcs_hybrid",
            executor=executor,
            verbose=True,
        )

        # 3. Run Full Iterative Lottery-Ticket Loop (IMP-TM Spectrum)
        iterative_res = run_iterative_lottery_ticket(
            tm=tm,
            checkpoints=checkpoints,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            retention_schedule=schedule,
            retrain_epochs=retrain_epochs,
            min_literals=1,
            scoring_method="mcs_hybrid",
            run_scratch_baselines=True,
            executor=executor,
            verbose=True,
        )

        # 4. Compare All Clause Optimization Techniques Head-to-Head
        comparison_res = compare_optimization_methods(
            tm=tm,
            checkpoints=checkpoints,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            budget_ratios=comp_budgets,
            retrain_epochs=retrain_epochs,
            executor=executor,
            verbose=True,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    total_time = round(time.time() - t_start, 2)

    all_results = {
        "config": {
            "preset": args.preset,
            "dataset": args.dataset,
            "clauses": clauses,
            "T": T,
            "s": args.s,
            "epochs": epochs,
            "retrain_epochs": retrain_epochs,
            "rewind_epoch": args.rewind_epoch,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "platform": args.platform,
            "weighted": True,
            "seed": args.seed,
            "total_runtime_sec": total_time,
        },
        "dense_training_history": train_history,
        "disjoint_tickets": disjoint_res,
        "iterative_lottery": iterative_res,
        "method_comparison": comparison_res,
    }

    # Save JSON, CSVs, Plots, and Markdown summary
    json_path = out_dir / "experiment_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    save_csv(disjoint_res["disjoint_tickets"], out_dir / "disjoint_lottery_tickets.csv")
    save_csv(iterative_res["iterative_tickets"], out_dir / "iterative_lottery_tickets.csv")
    save_csv(comparison_res["method_comparison"], out_dir / "optimization_methods_comparison.csv")

    generate_all_plots(all_results, out_dir)
    report_path = write_markdown_report(all_results, out_dir)

    print("\n" + "=" * 78)
    print(f"Completed in {total_time:.1f}s! Outputs written to: {out_dir.resolve()}")
    print(f"  - JSON Results:   {json_path}")
    print(f"  - Summary Report: {report_path}")
    print(f"  - Figures:        fig1_iterative_lottery_tickets.pdf, fig2_multiple_disjoint_tickets.pdf, fig3_optimization_methods_comparison.pdf")
    print("=" * 78)


if __name__ == "__main__":
    main()
