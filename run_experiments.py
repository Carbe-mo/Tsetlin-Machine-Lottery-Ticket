#!/usr/bin/env python3
"""
Main Experiment Runner for Parameter-Free TM-Ticket:
Data-Driven Clause Sanitization & Multiple Lottery Ticket Discovery in Tsetlin Machines.

All pruning capacities, margin temperatures, blending weights, stopping criteria (1-SE rule),
and disjoint ticket counts (M_auto) are derived directly from the data without manual
pruning hyperparameters, and every lottery ticket is trained/retrained for 50 epochs.
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


def detect_effective_cpu_quota() -> int:
    """
    Detects the effective CPU core quota in Linux cgroups v2 (`/sys/fs/cgroup/cpu.max`)
    so worker pools match the true container CPU allocation without thread thrashing.
    """
    try:
        cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max_path.exists():
            parts = cpu_max_path.read_text(encoding="utf-8").strip().split()
            if len(parts) == 2 and parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                if period > 0:
                    return max(1, quota // period)
    except Exception:
        pass
    return max(1, min(8, os.cpu_count() or 1))


def load_dataset(
    dataset_name: str = "mnist",
    train_samples: int = 50000,
    val_samples: int = 10000,
    test_samples: int = 10000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads booleanized MNIST (via tmu.data.MNIST) or Fashion-MNIST / KMNIST
    and creates a stratified Train / Val / Test split.
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
    peel_disj = results["disjoint_tickets"]["disjoint_tickets"]
    snake_disj = results["disjoint_tickets"].get("snake_draft_tickets", [])
    iter_res = results["iterative_lottery"]
    iter_rows = iter_res["iterative_tickets"]
    opt_1se = iter_res["optimal_1se_ticket"]
    opt_95ci = iter_res["optimal_95ci_ticket"]
    comp_rows = results["method_comparison"]["method_comparison"]

    lines = [
        "# Parameter-Free TM-Ticket: Data-Driven Lottery Tickets & Clause Optimization (50 Epochs)",
        "",
        "## 1. Configuration & Data-Driven Calibration Summary",
        f"- **Dataset:** `{cfg['dataset'].upper()}` "
        f"(Train: {cfg['train_samples']}, Val: {cfg['val_samples']}, Test: {cfg['test_samples']})",
        f"- **Clauses per class ($N$):** {cfg['clauses']} "
        f"(Total clauses: {dense['total_clauses']})",
        f"- **Training & Retraining Epochs:** **{cfg['epochs']} epochs dense + {cfg['retrain_epochs']} epochs retrain per lottery ticket**",
        f"- **Hyperparameters:** $T = {cfg['T']}$, $s = {cfg['s']}$, "
        f"Weighted Clauses = `{cfg['weighted']}`, Platform = `{cfg['platform']}`",
        f"- **Dense TM Accuracy:** Test = **{dense['test_accuracy']:.2f}%**, Val = **{dense['val_accuracy']:.2f}%** "
        f"(Footprint: `{dense['model_size_kb']:.2f} KB`)",
        f"- **Data-Driven Margin Parameters (Zero Manual Tuning):** "
        f"$\\tau_{{\\mathrm{{data}}}} = {dense['tau_data']}$, "
        f"$e_{{\\mathrm{{val}}}} = {dense['e_val']}$, "
        f"1-SE Tolerance $\\sigma_{{\\mathrm{{val}}}} = \\pm{dense['sigma_val_pct']:.2f}\\%$, "
        f"95%-CI Tolerance $= \\pm{dense['ci95_val_pct']:.2f}\\%$",
        f"- **Stage 1 + Certified Lossless Removed:** "
        f"Empty = {dense['stage1_removed']['empty']}, "
        f"Duplicates Merged = {dense['stage1_removed']['duplicate']}, "
        f"Contradictory = {dense['stage1_removed']['contradictory']}, "
        f"Margin-Certified Lossless ($\\sum |w|C < \\mu/2$) = {iter_res['dense_baseline']['certified_lossless_removed']}",
        f"- **1-SE Optimal Parameter-Free Ticket:** `{opt_1se['level_name']}` "
        f"(**{opt_1se['active_clauses']} clauses**, **{opt_1se['model_size_kb']:.2f} KB**, "
        f"**{opt_1se['compression_ratio']:.2f}x compression**, Test Acc = **{opt_1se['winning_ticket_test_acc']:.2f}%**)",
        f"- **95%-CI Optimal Parameter-Free Ticket:** `{opt_95ci['level_name']}` "
        f"(**{opt_95ci['active_clauses']} clauses**, **{opt_95ci['model_size_kb']:.2f} KB**, "
        f"**{opt_95ci['compression_ratio']:.2f}x compression**, Test Acc = **{opt_95ci['winning_ticket_test_acc']:.2f}%**)",
        "",
        "---",
        "",
        f"## 2. Parameter-Free Mutually Disjoint Winning Tickets "
        f"($M_{{\\mathrm{{auto}}}} = {results['disjoint_tickets']['num_disjoint_tickets']}$, "
        f"Auto Core Size = {results['disjoint_tickets']['ticket_ratio']*100:.1f}% Each, 0% Overlap)",
        "",
        "### 2A. Sequential Top-Down Peeling Disjoint Tickets ($\\mathcal{T}_a \\cap \\mathcal{T}_b = \\emptyset$)",
        "",
        "| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\\Theta_{e_r}$) + 50-Ep Retrain (%) | Fine-Tune 50-Ep (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for t in peel_disj:
        lines.append(
            f"| **{t['ticket_id']}** | {t['active_clauses']} | {t['model_size_kb']:.2f} | "
            f"{t['compression_ratio']:.2f}x | {t['overlap_with_previous_tickets']} | "
            f"{t['oneshot_renorm_test_acc']:.2f}% | {t['rewind_early_test_acc']:.2f}% | "
            f"{t['finetune_test_acc']:.2f}% | **{t['winning_ticket_test_acc']:.2f}%** | "
            f"**{t['cumulative_ensemble_test_acc']:.2f}%** |"
        )

    if snake_disj:
        lines.extend(
            [
                "",
                "### 2B. Balanced Snake-Draft Disjoint Tickets ($\\mathcal{T}_a \\cap \\mathcal{T}_b = \\emptyset$)",
                "",
                "| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\\Theta_{e_r}$) + 50-Ep Retrain (%) | Fine-Tune 50-Ep (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for t in snake_disj:
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
            "## 3. Parameter-Free Data-Driven Lottery-Ticket Hierarchy (50-Epoch Retraining)",
            "",
            "| Round | Data-Driven Capacity Level | Retained (%) | Active Clauses | Size (KB) | One-Shot (%) | Rewind $\\Theta_{e_r}$ (%) | Rewind $\\Theta_0$ (%) | Fine-Tune (%) | Random State (%) | Random Mask (%) | Scratch TM (%) | 1-SE Rule |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for r in iter_rows:
        pass_str = "PASS" if r.get("passes_1se_rule") else ("95%-CI" if r.get("passes_95ci_rule") else "Below")
        lines.append(
            f"| {r['round']} | `{r['level_name']}` | {r['actual_retention_ratio']*100:.1f}% | "
            f"{r['active_clauses']} | {r['model_size_kb']:.2f} | {r['oneshot_renorm_test_acc']:.2f}% | "
            f"**{r['rewind_early_test_acc']:.2f}%** | {r['rewind_init_test_acc']:.2f}% | "
            f"**{r['finetune_test_acc']:.2f}%** | {r['random_reinit_test_acc']:.2f}% | "
            f"{r['random_mask_retrain_test_acc']:.2f}% | {r['scratch_small_test_acc']:.2f}% | {pass_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Head-to-Head Clause Optimization Comparison at Data-Driven Capacities (50-Epoch Retraining)",
            "",
            "| Data-Driven Budget (%) | Clause Optimization Method | Active Clauses | Size (KB) | Mean Literals | One-Shot Test Acc (%) | 50-Ep Lottery-Ticket Retrained Acc (%) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )

    for c in comp_rows:
        lines.append(
            f"| {c['budget_ratio']*100:.1f}% | {c['method']} | {c['active_clauses']} | "
            f"{c['model_size_kb']:.2f} | {c['mean_literal_length']:.1f} | "
            f"{c['oneshot_best_test_acc']:.2f}% | **{c['lottery_ticket_retrain_test_acc']:.2f}%** |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter-Free TM-Ticket: Data-Driven Clause Sanitization & Lottery Ticket Discovery (50 Epochs)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="full",
        choices=["fast", "standard", "full"],
        help="Experiment scale preset (default: 'full' = 50 epochs dense + 50 epochs retrain).",
    )
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "kmnist"])
    parser.add_argument("--clauses", type=int, default=None, help="Number of clauses per class (N)")
    parser.add_argument("--T", type=int, default=None, help="Voting threshold T")
    parser.add_argument("--s", type=float, default=5.0, help="Specificity parameter s (default: 5.0 from sweep)")
    parser.add_argument("--max-included-literals", type=int, default=None, help="Optional cap on included literals")
    parser.add_argument("--epochs", type=int, default=None, help="Dense training epochs (default: 50)")
    parser.add_argument("--retrain-epochs", type=int, default=None, help="Retraining epochs per ticket (default: 50)")
    parser.add_argument("--rewind-epoch", type=int, default=1, help="Early rewind checkpoint epoch e_r")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CPU", "CUDA", "GPU"])
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel CPU worker processes (auto-detected from cgroup CPU quota by default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/mnist_2k_50ep_paramfree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset == "fast":
        clauses = args.clauses or 400
        T = args.T or 300
        epochs = args.epochs or 50
        retrain_epochs = args.retrain_epochs or 50
        train_samples = args.train_samples or 12000
        val_samples = args.val_samples or 3000
        test_samples = args.test_samples or 5000
    elif args.preset == "standard":
        clauses = args.clauses or 1000
        T = args.T or 750
        epochs = args.epochs or 50
        retrain_epochs = args.retrain_epochs or 50
        train_samples = args.train_samples or 30000
        val_samples = args.val_samples or 5000
        test_samples = args.test_samples or 10000
    else:  # full (50 epochs dense + 50 epochs retrain per lottery ticket)
        clauses = args.clauses or 2000
        T = args.T or 1500
        epochs = args.epochs or 50
        retrain_epochs = args.retrain_epochs or 50
        train_samples = args.train_samples or 50000
        val_samples = args.val_samples or 10000
        test_samples = args.test_samples or 10000

    workers = args.workers if args.workers is not None else detect_effective_cpu_quota()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print(
        f"Parameter-Free TM-Ticket Suite | Preset: {args.preset.upper()} | Dataset: {args.dataset.upper()}"
    )
    print(
        f"Clauses/class: {clauses} (Total: {clauses*10}) | T={T} | s={args.s} | "
        f"Epochs={epochs} (Rewind e_r={args.rewind_epoch}, Retrain={retrain_epochs}) | "
        f"Platform={args.platform} | Workers={workers}"
    )
    print("=" * 82, flush=True)

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

    # 1. Train Dense TM for 50 Epochs & Save Checkpoints (Theta_0, Theta_{e_r}, Theta_final)
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
        # 2. Parameter-Free Discovery of Multiple Disjoint Winning Tickets (Peeling + Snake-Draft)
        disjoint_res = discover_multiple_disjoint_tickets(
            tm=tm,
            checkpoints=checkpoints,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            retrain_epochs=retrain_epochs,
            scoring_method="mcs_hybrid",
            executor=executor,
            verbose=True,
        )
        # Save intermediate checkpoint after Section 2
        (out_dir / "partial_disjoint_results.json").write_text(
            json.dumps(disjoint_res, indent=2), encoding="utf-8"
        )

        # 3. Parameter-Free Data-Driven Lottery-Ticket Hierarchy (IMP-TM + 1-SE Rule)
        iterative_res = run_iterative_lottery_ticket(
            tm=tm,
            checkpoints=checkpoints,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            retrain_epochs=retrain_epochs,
            scoring_method="mcs_hybrid",
            run_scratch_baselines=True,
            executor=executor,
            verbose=True,
        )
        (out_dir / "partial_iterative_results.json").write_text(
            json.dumps(iterative_res, indent=2), encoding="utf-8"
        )

        # 4. Compare All Clause Optimization Techniques at Data-Driven Capacities
        comparison_res = compare_optimization_methods(
            tm=tm,
            checkpoints=checkpoints,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
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
            "workers": workers,
            "weighted": True,
            "parameter_free": True,
            "seed": args.seed,
            "total_runtime_sec": total_time,
        },
        "dense_training_history": train_history,
        "disjoint_tickets": disjoint_res,
        "iterative_lottery": iterative_res,
        "method_comparison": comparison_res,
    }

    json_path = out_dir / "experiment_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    save_csv(disjoint_res["disjoint_tickets"], out_dir / "disjoint_peeling_tickets.csv")
    if disjoint_res.get("snake_draft_tickets"):
        save_csv(disjoint_res["snake_draft_tickets"], out_dir / "disjoint_snake_draft_tickets.csv")
    save_csv(iterative_res["iterative_tickets"], out_dir / "iterative_lottery_tickets.csv")
    save_csv(comparison_res["method_comparison"], out_dir / "optimization_methods_comparison.csv")

    generate_all_plots(all_results, out_dir)
    report_path = write_markdown_report(all_results, out_dir)

    print("\n" + "=" * 82)
    print(f"Completed in {total_time:.1f}s! Outputs written to: {out_dir.resolve()}")
    print(f"  - JSON Results:   {json_path}")
    print(f"  - Summary Report: {report_path}")
    print("=" * 82, flush=True)


if __name__ == "__main__":
    main()
