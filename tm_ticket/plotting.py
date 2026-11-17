"""
Publication-quality figure generation (`tm_ticket/plotting.py`).
Saves both PDF (for IEEE LaTeX inclusion) and PNG (for quick viewing / GitHub README).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_all_plots(results: Dict[str, Any], out_dir: Path) -> None:
    """Generates PDF and PNG plots for all completed experiment sections."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
        }
    )

    if "iterative_lottery" in results:
        _plot_iterative_lottery(results["iterative_lottery"], out_dir)

    if "disjoint_tickets" in results:
        _plot_disjoint_tickets(results["disjoint_tickets"], out_dir)

    if "method_comparison" in results:
        _plot_method_comparison(results["method_comparison"], out_dir)


def _plot_iterative_lottery(iter_res: Dict[str, Any], out_dir: Path) -> None:
    rounds = iter_res["iterative_tickets"]
    dense_acc = iter_res["dense_baseline"]["test_accuracy"]

    ratios = [r["actual_retention_ratio"] * 100.0 for r in rounds]
    rw_early = [r["rewind_early_test_acc"] for r in rounds]
    rw_init = [r["rewind_init_test_acc"] for r in rounds]
    finetune = [r["finetune_test_acc"] for r in rounds]
    oneshot = [r["oneshot_renorm_test_acc"] for r in rounds]
    rand_mask = [r["random_mask_retrain_test_acc"] for r in rounds]
    rand_reinit = [r["random_reinit_test_acc"] for r in rounds]
    scratch = [r["scratch_small_test_acc"] for r in rounds]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(
        dense_acc,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Dense TM Baseline ({dense_acc:.2f}%)",
    )

    ax.plot(
        ratios,
        rw_early,
        "o-",
        color="#d62728",
        linewidth=2.2,
        markersize=6,
        label="Winning Ticket (Rewind $\\Theta_{e_r}$ + Retrain)",
    )
    ax.plot(
        ratios,
        finetune,
        "s-",
        color="#ff7f0e",
        linewidth=1.8,
        markersize=5,
        label="Fine-Tuned Ticket ($\\Theta_{\\mathrm{final}}$ + Retrain)",
    )
    ax.plot(
        ratios,
        rw_init,
        "^-",
        color="#9467bd",
        linewidth=1.6,
        markersize=5,
        label="Winning Ticket (Rewind $\\Theta_0$ + Retrain)",
    )
    ax.plot(
        ratios,
        oneshot,
        "d--",
        color="#1f77b4",
        linewidth=1.6,
        markersize=5,
        label="One-Shot TM-Ticket (No Retraining)",
    )
    if any(s > 0 for s in scratch):
        ax.plot(
            ratios,
            scratch,
            "v-.",
            color="#2ca02c",
            linewidth=1.5,
            markersize=5,
            label="Small TM Trained from Scratch",
        )
    ax.plot(
        ratios,
        rand_reinit,
        "x:",
        color="#8c564b",
        linewidth=1.4,
        markersize=5,
        label="Same Mask + Random TA State Re-init",
    )
    ax.plot(
        ratios,
        rand_mask,
        "+:",
        color="#7f7f7f",
        linewidth=1.4,
        markersize=6,
        label="Random Mask + Rewind + Retrain",
    )

    ax.set_xlabel("Retained Active Clauses (% of Original Bank)")
    ax.set_ylabel("MNIST Test Accuracy (%)")
    ax.set_title("TM Lottery Ticket Hypothesis: Accuracy vs. Clause Sparsity")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_iterative_lottery_tickets.{ext}")
    plt.close(fig)


def _plot_disjoint_tickets(disj_res: Dict[str, Any], out_dir: Path) -> None:
    tickets = disj_res["disjoint_tickets"]
    dense_acc = disj_res["dense_baseline"]["test_accuracy"]

    labels = [
        f"{t['ticket_id']}\n({t['active_clauses']} cls,\n{t['model_size_kb']:.0f} KB)"
        for t in tickets
    ]
    oneshot_accs = [t["oneshot_renorm_test_acc"] for t in tickets]
    retrain_accs = [t["winning_ticket_test_acc"] for t in tickets]
    ensemble_accs = [t["cumulative_ensemble_test_acc"] for t in tickets]

    x = np.arange(len(tickets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars1 = ax.bar(
        x - width / 2,
        oneshot_accs,
        width,
        color="#6baed6",
        edgecolor="#2171b5",
        label="Disjoint Ticket (One-Shot, No Retraining)",
    )
    bars2 = ax.bar(
        x + width / 2,
        retrain_accs,
        width,
        color="#de2d26",
        edgecolor="#a50f15",
        label="Disjoint Winning Ticket (Rewind + Retrain)",
    )
    ax.plot(
        x,
        ensemble_accs,
        "o-",
        color="#2ca02c",
        linewidth=2.2,
        markersize=7,
        label="Cumulative Multi-Ticket Ensemble (Tickets 1..m)",
    )
    ax.axhline(
        dense_acc,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Full Dense TM ({dense_acc:.2f}%)",
    )

    for b in bars2:
        h = b.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    all_vals = oneshot_accs + retrain_accs + ensemble_accs + [dense_acc]
    y_min = max(0.0, min(all_vals) - 3.5)
    y_max = min(100.0, max(all_vals) + 1.8)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MNIST Test Accuracy (%)")
    ax.set_title(
        f"Multiple Mutually Disjoint Winning Tickets (0% Overlap, {disj_res['ticket_ratio']*100:.0f}% Size Each)"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_multiple_disjoint_tickets.{ext}")
    plt.close(fig)


def _plot_method_comparison(comp_res: Dict[str, Any], out_dir: Path) -> None:
    rows = comp_res["method_comparison"]
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for idx, m_name in enumerate(methods):
        m_rows = [r for r in rows if r["method"] == m_name]
        m_rows.sort(key=lambda x: x["budget_ratio"], reverse=True)
        budgets = [r["budget_ratio"] * 100.0 for r in m_rows]
        os_acc = [r["oneshot_best_test_acc"] for r in m_rows]
        lth_acc = [r["lottery_ticket_retrain_test_acc"] for r in m_rows]
        mk = markers[idx % len(markers)]
        lw = 2.2 if m_name.startswith("TM-Ticket") else 1.4

        axes[0].plot(budgets, os_acc, marker=mk, linewidth=lw, label=m_name)
        axes[1].plot(budgets, lth_acc, marker=mk, linewidth=lw, label=m_name)

    axes[0].set_title("(a) One-Shot Post-Training Sanitization")
    axes[1].set_title("(b) Lottery-Ticket Rewind + Retrain")

    for ax in axes:
        ax.set_xlabel("Retained Clause Budget (%)")
        ax.invert_xaxis()
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_ylabel("MNIST Test Accuracy (%)")
    axes[1].legend(loc="lower left", fontsize=7.8, frameon=True)
    fig.suptitle("Clause Optimization Strategies: One-Shot vs. Lottery-Ticket Retraining")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig3_optimization_methods_comparison.{ext}")
    plt.close(fig)
