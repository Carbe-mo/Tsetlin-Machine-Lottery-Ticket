"""
Parameter-Free, Data-Driven Lottery Ticket Discovery & Multi-Method Benchmark (`tm_ticket/lottery.py`).

All ticket sizes, pruning schedules, stopping criteria, and disjoint ticket counts
are derived automatically from the data (empirical Margin Contribution Scores on D_val)
without manual thresholds (theta, L_min, tau, alpha, or fixed budget grids):

  1. `discover_multiple_disjoint_tickets`:
     Uses the data-driven 2nd-order Kneedle capacity `N_core(k, p)` and automatic ticket
     count `M_auto = floor(median(N_pos / N_core))` to extract M_auto strictly disjoint
     (0% overlap) winning tickets under BOTH Sequential Peeling and Balanced Snake-Draft,
     retraining each for 50 epochs via physically compacted submodels.
  2. `run_iterative_lottery_ticket`:
     Evaluates the parameter-free data-driven capacity hierarchy:
       - Round 0: Stage 1 + Margin-Certified Lossless Bound (sum |w|C(x) < mu(x)/2)
       - Round 1: Positive-Contribution Pool (MCS* > 0)
       - Round 2: Shannon Perplexity Capacity (N_eff = exp(H))
       - Round 3: 1st-Order Kneedle / Above-Mean Capacity (s >= mean(s))
       - Round 4: Herfindahl Participation Ratio (N_PR = (sum s)^2 / sum s^2)
       - Round 5: 2nd-Order Recursive Kneedle Core Ticket
       - Round 6: 3rd-Order Recursive Kneedle Minimal Backbone
     Automatically selects the 1-SE and 95%-CI Winning Tickets using validation
     standard error sigma_val = sqrt(a_val * (1 - a_val) / |D_val|).
  3. `compare_optimization_methods`:
     Benchmarks all 8 clause optimization criteria head-to-head at the data-driven
     capacities (`Kneedle_AboveMean`, `Participation_Ratio`, `Recursive_Core`, `Minimal_Backbone`),
     reporting both One-Shot and 50-epoch Retrained accuracies.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import PrunableTMClassifier, TMCheckpoint
from .sanitization import (
    compute_certified_lossless_mask,
    compute_clause_scores,
    compute_data_driven_capacities,
    compute_model_stats,
    length_based_filter,
    partition_disjoint_tickets_Auto,
    rule_based_canonicalization,
    select_ticket_mask,
)

# Worker-process global dataset cache to avoid re-serializing datasets per task
_WORKER_DATA: Dict[str, np.ndarray] = {}


def _init_worker_data(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> None:
    global _WORKER_DATA
    _WORKER_DATA = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,
    }


def _parallel_retrain_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single 50-epoch ticket retraining job in a CPU worker process
    using a physically compacted `PrunableTMClassifier` (`build_compact_submodel`),
    which runs 3x-10x faster than training a zero-masked full bank.
    """
    X_train = _WORKER_DATA["X_train"]
    Y_train = _WORKER_DATA["Y_train"]
    X_val = _WORKER_DATA["X_val"]
    Y_val = _WORKER_DATA["Y_val"]
    X_test = _WORKER_DATA["X_test"]
    Y_test = _WORKER_DATA["Y_test"]

    mode = job["mode"]
    retrain_epochs = job["retrain_epochs"]

    if mode in ("checkpoint", "random_reinit"):
        ckpt: TMCheckpoint = job["checkpoint"]
        mask: np.ndarray = job["mask"]
        rand_seed = job.get("random_reinit_seed") if mode == "random_reinit" else None

        sub_tm = PrunableTMClassifier.build_compact_submodel(
            X_sample=X_train[:20],
            Y_sample=Y_train[:20],
            checkpoint=ckpt,
            retention_mask=mask,
            base_clauses=job["number_of_clauses"],
            base_T=job["T"],
            s=job["s"],
            weighted_clauses=job["weighted_clauses"],
            max_included_literals=job.get("max_included_literals"),
            platform="CPU",
            seed=job["seed"],
            random_reinit_seed=rand_seed,
        )
    elif mode == "scratch":
        sub_tm = PrunableTMClassifier(
            number_of_clauses=job["number_of_clauses"],
            T=job["T"],
            s=job["s"],
            platform="CPU",
            weighted_clauses=job["weighted_clauses"],
            max_included_literals=job.get("max_included_literals"),
            seed=job["seed"],
        )
        sub_tm.init(X_train[:20], Y_train[:20])
    else:
        raise ValueError(f"Unknown worker job mode: {mode}")

    best_val, best_test, best_scores = _retrain_and_eval(
        sub_tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
    )

    return {
        "job_id": job["job_id"],
        "best_val": best_val,
        "best_test": best_test,
        "best_scores": best_scores if job.get("return_scores", False) else None,
        "compact_clauses_per_class": sub_tm.number_of_clauses,
    }


def train_dense_tm_with_checkpoints(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    number_of_clauses: int = 2000,
    T: int = 1500,
    s: float = 5.0,
    epochs: int = 50,
    rewind_epoch: int = 1,
    platform: str = "CPU",
    weighted_clauses: bool = True,
    max_included_literals: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[PrunableTMClassifier, Dict[str, TMCheckpoint], List[Dict[str, Any]]]:
    tm = PrunableTMClassifier(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        platform=platform,
        weighted_clauses=weighted_clauses,
        max_included_literals=max_included_literals,
        seed=seed,
    )
    tm.init(X_train, Y_train)

    checkpoints: Dict[str, TMCheckpoint] = {}
    checkpoints["init"] = tm.get_checkpoint(epoch=0)

    history: List[Dict[str, Any]] = []
    best_val_acc = -1.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tm.fit(X_train, Y_train)
        elapsed = time.time() - t0

        val_acc = tm.evaluate_accuracy(X_val, Y_val)
        test_acc = tm.evaluate_accuracy(X_test, Y_test)
        history.append(
            {
                "epoch": ep,
                "val_accuracy": round(val_acc, 3),
                "test_accuracy": round(test_acc, 3),
                "time_sec": round(elapsed, 2),
            }
        )
        if verbose:
            print(
                f"  [Dense Train] Epoch {ep:02d}/{epochs:02d} | "
                f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                f"Time: {elapsed:.2f}s",
                flush=True,
            )

        if ep == rewind_epoch:
            checkpoints["early"] = tm.get_checkpoint(epoch=ep)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoints["final"] = tm.get_checkpoint(epoch=ep)

    if "early" not in checkpoints:
        checkpoints["early"] = tm.get_checkpoint(epoch=epochs)
    if "final" not in checkpoints:
        checkpoints["final"] = tm.get_checkpoint(epoch=epochs)

    return tm, checkpoints, history


def _retrain_and_eval(
    tm: PrunableTMClassifier,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    retrain_epochs: int,
) -> Tuple[float, float, np.ndarray]:
    best_val = tm.evaluate_accuracy(X_val, Y_val)
    best_scores = tm.compute_class_scores(X_test)
    best_test = float(np.mean(np.argmax(best_scores, axis=1) == Y_test) * 100.0)

    for _ in range(retrain_epochs):
        tm.fit(X_train, Y_train)
        val_acc = tm.evaluate_accuracy(X_val, Y_val)
        if val_acc >= best_val:
            best_val = val_acc
            best_scores = tm.compute_class_scores(X_test)
            best_test = float(np.mean(np.argmax(best_scores, axis=1) == Y_test) * 100.0)

    return round(best_val, 3), round(best_test, 3), best_scores


def discover_multiple_disjoint_tickets(
    tm: PrunableTMClassifier,
    checkpoints: Dict[str, TMCheckpoint],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    retrain_epochs: int = 50,
    scoring_method: str = "mcs_hybrid",
    executor: Optional[ProcessPoolExecutor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Parameter-Free Discovery of Multiple Mutually Disjoint Winning Tickets:
      - Derives per-class, per-polarity core ticket capacity `N_core(k, p)` (2nd-order
        Kneedle of positive MCS* scores) and automatic number of disjoint tickets
        `M_auto = floor(median(N_pos / N_core))` directly from the validation data.
      - Evaluates BOTH parameter-free partitioning strategies:
        1. 'peeling'     (Sequential top-down disjoint peeling: Ticket #1, #2, ..., #M_auto)
        2. 'snake_draft' (Round-robin snake-draft partitioning into M_auto balanced tickets)
      - Retrains each disjoint ticket for `retrain_epochs` (50 epochs) using physically
        compacted submodels.
    """
    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    dense_val_acc = tm.evaluate_accuracy(X_val, Y_val)
    dense_test_acc = tm.evaluate_accuracy(X_test, Y_test)
    dense_stats = compute_model_stats(tm)

    # Stage 1 Lossless Canonicalization (no manual L_min filter)
    clean_mask, stage1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)

    C_val = tm.evaluate_clause_outputs(X_val)
    all_scores = compute_clause_scores(
        tm, X_val, Y_val, base_mask=clean_mask, clause_outputs=C_val
    )
    primary_scores = all_scores[scoring_method]

    cap_info = compute_data_driven_capacities(
        scores=primary_scores,
        base_mask=clean_mask,
        val_accuracy_pct=dense_val_acc,
        n_val_samples=X_val.shape[0],
    )
    num_tickets = cap_info["m_auto_disjoint_tickets"]
    core_template = cap_info["masks"]["kneedle_2_core"]
    auto_ratio = round(
        float(np.sum(core_template)) / float(dense_stats.total_clauses), 4
    )

    if verbose:
        print(
            f"\n=== Parameter-Free Disjoint Ticket Discovery ==="
            f"\n  Data-Driven Core Ticket Size: {int(np.sum(core_template))}/{dense_stats.total_clauses} clauses "
            f"({auto_ratio*100:.1f}% of bank) | Auto Ticket Count M_auto = {num_tickets} "
            f"| Retrain = {retrain_epochs} epochs",
            flush=True,
        )

    all_modes_results: Dict[str, List[Dict[str, Any]]] = {}

    for part_mode in ("peeling", "snake_draft"):
        if verbose:
            print(f"\n  --- Partitioning Mode: {part_mode.upper()} (0% Overlap) ---", flush=True)

        ticket_masks = partition_disjoint_tickets_Auto(
            scores=primary_scores,
            base_mask=clean_mask,
            target_mask_template=core_template,
            num_tickets=num_tickets,
            mode=part_mode,
        )

        # Verify strict mutual disjointness across all M_auto tickets
        used_so_far = np.zeros_like(clean_mask, dtype=np.int32)
        oneshot_meta: List[Dict[str, Any]] = []

        for idx, t_mask in enumerate(ticket_masks):
            t_idx = idx + 1
            overlap = int(np.sum(t_mask & used_so_far))
            assert overlap == 0, f"Overlap {overlap} detected in {part_mode} ticket {t_idx}!"
            used_so_far = used_so_far | t_mask

            tm.rewind_to_checkpoint(
                ckpt_clean_final, retention_mask=t_mask, rescale_T_mode="proportional"
            )
            os_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=False)
            os_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)
            tm.update_vote_renormalization()
            os_renorm_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=True)
            os_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)
            t_stats = compute_model_stats(tm, mask=t_mask)

            oneshot_meta.append(
                {
                    "t_idx": t_idx,
                    "t_mask": t_mask,
                    "t_stats": t_stats,
                    "overlap": overlap,
                    "oneshot_val": os_val,
                    "oneshot_test": os_test,
                    "oneshot_renorm_val": os_renorm_val,
                    "oneshot_renorm_test": os_renorm_test,
                }
            )

        jobs = []
        for meta in oneshot_meta:
            t_idx = meta["t_idx"]
            t_mask = meta["t_mask"]
            jobs.append(
                {
                    "job_id": f"{part_mode}_ft_{t_idx}",
                    "mode": "checkpoint",
                    "checkpoint": ckpt_clean_final,
                    "mask": t_mask,
                    "number_of_clauses": tm.number_of_clauses,
                    "T": tm.initial_T,
                    "s": tm.s,
                    "weighted_clauses": tm.weighted_clauses,
                    "max_included_literals": tm.max_included_literals,
                    "seed": tm.seed,
                    "retrain_epochs": retrain_epochs,
                    "return_scores": True,
                }
            )
            jobs.append(
                {
                    "job_id": f"{part_mode}_rw_{t_idx}",
                    "mode": "checkpoint",
                    "checkpoint": checkpoints["early"],
                    "mask": t_mask,
                    "number_of_clauses": tm.number_of_clauses,
                    "T": tm.initial_T,
                    "s": tm.s,
                    "weighted_clauses": tm.weighted_clauses,
                    "max_included_literals": tm.max_included_literals,
                    "seed": tm.seed,
                    "retrain_epochs": retrain_epochs,
                    "return_scores": True,
                }
            )

        retrain_results: Dict[str, Dict[str, Any]] = {}
        if executor is not None:
            for res in executor.map(_parallel_retrain_worker, jobs):
                retrain_results[res["job_id"]] = res
        else:
            _init_worker_data(X_train, Y_train, X_val, Y_val, X_test, Y_test)
            for j_spec in jobs:
                res = _parallel_retrain_worker(j_spec)
                retrain_results[res["job_id"]] = res

        mode_report: List[Dict[str, Any]] = []
        ensemble_test_scores = np.zeros(
            (X_test.shape[0], tm.number_of_classes), dtype=np.float64
        )

        for meta in oneshot_meta:
            t_idx = meta["t_idx"]
            t_stats = meta["t_stats"]
            ft_res = retrain_results[f"{part_mode}_ft_{t_idx}"]
            rw_res = retrain_results[f"{part_mode}_rw_{t_idx}"]

            ft_val, ft_test, ft_scores = (
                ft_res["best_val"],
                ft_res["best_test"],
                ft_res["best_scores"],
            )
            rw_val, rw_test, rw_scores = (
                rw_res["best_val"],
                rw_res["best_test"],
                rw_res["best_scores"],
            )

            if rw_val >= ft_val:
                best_ticket_val, best_ticket_test, best_ticket_scores = (
                    rw_val,
                    rw_test,
                    rw_scores,
                )
                best_regime = "rewind_early"
            else:
                best_ticket_val, best_ticket_test, best_ticket_scores = (
                    ft_val,
                    ft_test,
                    ft_scores,
                )
                best_regime = "finetune"

            norm_factor = np.std(best_ticket_scores)
            if norm_factor > 0:
                ensemble_test_scores += best_ticket_scores / norm_factor
            else:
                ensemble_test_scores += best_ticket_scores

            ensemble_test_acc = float(
                np.mean(np.argmax(ensemble_test_scores, axis=1) == Y_test) * 100.0
            )

            label_prefix = "Peel" if part_mode == "peeling" else "Snake"
            ticket_entry = {
                "partition_mode": part_mode,
                "ticket_id": f"{label_prefix}-Ticket #{t_idx}",
                "ticket_index": t_idx,
                "retention_ratio": round(
                    t_stats.active_clauses / float(dense_stats.total_clauses), 4
                ),
                "active_clauses": t_stats.active_clauses,
                "active_clauses_per_class": t_stats.active_per_class,
                "model_size_kb": t_stats.model_size_kb,
                "compression_ratio": round(
                    dense_stats.model_size_kb / max(t_stats.model_size_kb, 1e-6), 2
                ),
                "mean_literal_length": t_stats.mean_literal_length,
                "overlap_with_previous_tickets": meta["overlap"],
                "oneshot_val_acc": round(meta["oneshot_val"], 3),
                "oneshot_test_acc": round(meta["oneshot_test"], 3),
                "oneshot_renorm_val_acc": round(meta["oneshot_renorm_val"], 3),
                "oneshot_renorm_test_acc": round(meta["oneshot_renorm_test"], 3),
                "rewind_early_val_acc": rw_val,
                "rewind_early_test_acc": rw_test,
                "finetune_val_acc": ft_val,
                "finetune_test_acc": ft_test,
                "winning_ticket_val_acc": best_ticket_val,
                "winning_ticket_test_acc": best_ticket_test,
                "best_regime": best_regime,
                "cumulative_ensemble_test_acc": round(ensemble_test_acc, 3),
            }
            mode_report.append(ticket_entry)

            if verbose:
                print(
                    f"    [{ticket_entry['ticket_id']}] Clauses: {t_stats.active_clauses} "
                    f"({t_stats.model_size_kb:.1f} KB, {ticket_entry['compression_ratio']:.2f}x, Overlap=0) | "
                    f"One-Shot: {meta['oneshot_renorm_test']:.2f}% | "
                    f"Rewind(e_r): {rw_test:.2f}% | "
                    f"FineTune: {ft_test:.2f}% | "
                    f"Winning Acc: {best_ticket_test:.2f}% | "
                    f"Ensemble(1..{t_idx}): {ensemble_test_acc:.2f}%",
                    flush=True,
                )

        all_modes_results[part_mode] = mode_report

    return {
        "dense_baseline": {
            "total_clauses": dense_stats.total_clauses,
            "model_size_kb": dense_stats.model_size_kb,
            "val_accuracy": round(dense_val_acc, 3),
            "test_accuracy": round(dense_test_acc, 3),
            "stage1_removed": stage1_counts,
            "short_removed": 0,
            "tau_data": all_scores["tau_data"],
            "e_val": all_scores["e_val"],
            "sigma_val_pct": cap_info["sigma_val_pct"],
            "ci95_val_pct": cap_info["ci95_val_pct"],
        },
        "ticket_ratio": auto_ratio,
        "num_disjoint_tickets": num_tickets,
        "scoring_method": scoring_method,
        "disjoint_tickets": all_modes_results["peeling"],
        "snake_draft_tickets": all_modes_results["snake_draft"],
    }


def run_iterative_lottery_ticket(
    tm: PrunableTMClassifier,
    checkpoints: Dict[str, TMCheckpoint],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    retrain_epochs: int = 50,
    scoring_method: str = "mcs_hybrid",
    run_scratch_baselines: bool = True,
    executor: Optional[ProcessPoolExecutor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs the 100% Parameter-Free, Data-Driven Lottery-Ticket Pruning Hierarchy:
      - Round 0: Stage 1 + Margin-Certified Lossless Bound (sum |w|C(x) < mu(x)/2, 0 flips)
      - Round 1..6: Data-driven capacity levels (Positive MCS*, Shannon Perplexity N_eff,
        1st-Order Kneedle, Herfindahl Participation Ratio N_PR, 2nd-Order Recursive Core,
        3rd-Order Minimal Backbone).
      - Applies the 1-SE Validation Stopping Rule (Acc_val >= Acc_val_dense - sigma_val)
        to automatically flag the parameter-free optimal ticket.
    """
    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    dense_val_acc = tm.evaluate_accuracy(X_val, Y_val)
    dense_test_acc = tm.evaluate_accuracy(X_test, Y_test)
    dense_stats = compute_model_stats(tm)

    stage1_mask, stage1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)

    C_val_dense = tm.evaluate_clause_outputs(X_val)
    scores_dict = compute_clause_scores(
        tm, X_val, Y_val, base_mask=stage1_mask, clause_outputs=C_val_dense
    )
    primary_scores = scores_dict[scoring_method]

    # Round 0: Margin-Certified Lossless Mask
    cert_mask, cert_removed = compute_certified_lossless_mask(
        tm=tm,
        base_mask=stage1_mask,
        scores=primary_scores,
        margins=scores_dict["margins"],
        clause_outputs=C_val_dense,
    )

    cap_info = compute_data_driven_capacities(
        scores=primary_scores,
        base_mask=cert_mask,
        val_accuracy_pct=dense_val_acc,
        n_val_samples=X_val.shape[0],
    )
    sigma_val = cap_info["sigma_val_pct"]
    ci95_val = cap_info["ci95_val_pct"]
    ordered_levels = cap_info["ordered_levels"]
    level_masks = cap_info["masks"]

    if verbose:
        print(
            f"\n=== Parameter-Free Data-Driven Lottery-Ticket Hierarchy (Retrain={retrain_epochs} epochs) ==="
            f"\n  Dense Val Acc: {dense_val_acc:.2f}% | 1-SE Tolerance (sigma_val): ±{sigma_val:.2f}% "
            f"(1-SE Threshold: {dense_val_acc - sigma_val:.2f}%)",
            flush=True,
        )

    rounds_report: List[Dict[str, Any]] = []
    rng = np.random.RandomState(777)

    # Evaluate Round 0: Stage 1 + Margin-Certified Lossless Bound
    tm.rewind_to_checkpoint(
        ckpt_clean_final, retention_mask=cert_mask, rescale_T_mode="fixed"
    )
    r0_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=False)
    r0_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)
    r0_stats = compute_model_stats(tm, mask=cert_mask, stage1_counts=stage1_counts)
    r0_ratio = round(r0_stats.active_clauses / float(dense_stats.total_clauses), 4)

    rounds_report.append(
        {
            "round": 0,
            "level_name": "R0_Certified_Lossless (Stage1 + mu(x)/2 Bound)",
            "target_ratio": r0_ratio,
            "actual_retention_ratio": r0_ratio,
            "active_clauses": r0_stats.active_clauses,
            "model_size_kb": r0_stats.model_size_kb,
            "compression_ratio": round(
                dense_stats.model_size_kb / max(r0_stats.model_size_kb, 1e-6), 2
            ),
            "mean_literal_length": r0_stats.mean_literal_length,
            "oneshot_val_acc": round(r0_val, 3),
            "oneshot_test_acc": round(r0_test, 3),
            "oneshot_renorm_test_acc": round(r0_test, 3),
            "rewind_early_val_acc": round(r0_val, 3),
            "rewind_early_test_acc": round(r0_test, 3),
            "rewind_init_test_acc": round(dense_test_acc, 3),
            "finetune_val_acc": round(r0_val, 3),
            "finetune_test_acc": round(r0_test, 3),
            "random_reinit_test_acc": round(dense_test_acc, 3),
            "random_mask_retrain_test_acc": round(dense_test_acc, 3),
            "scratch_small_test_acc": round(dense_test_acc, 3),
            "winning_ticket_val_acc": round(r0_val, 3),
            "winning_ticket_test_acc": round(r0_test, 3),
            "passes_1se_rule": True,
            "passes_95ci_rule": True,
        }
    )
    if verbose:
        print(
            f"  [Round 0 | R0_Certified_Lossless] Active: {r0_stats.active_clauses}/{dense_stats.total_clauses} "
            f"({r0_stats.model_size_kb:.1f} KB, Certified Removed={cert_removed}) | "
            f"Val Acc: {r0_val:.2f}% | Test Acc: {r0_test:.2f}% (0 Flips Guaranteed)",
            flush=True,
        )

    # Evaluate each data-driven capacity level (Round 1 .. Round 6)
    for r_idx, (level_label, level_key) in enumerate(ordered_levels, start=1):
        ticket_mask = level_masks[level_key]
        stats = compute_model_stats(tm, mask=ticket_mask)
        ratio = round(stats.active_clauses / float(dense_stats.total_clauses), 4)

        tm.rewind_to_checkpoint(
            ckpt_clean_final, retention_mask=ticket_mask, rescale_T_mode="proportional"
        )
        oneshot_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=False)
        oneshot_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)

        tm.update_vote_renormalization()
        oneshot_renorm_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=True)
        oneshot_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)

        rand_scores = rng.rand(tm.number_of_classes, tm.number_of_clauses)
        rand_mask = select_ticket_mask(
            scores=rand_scores,
            base_mask=np.ones_like(ticket_mask),
            target_retention_ratio=ratio,
            class_balanced=True,
        )

        n_small = max(4, int(round(ratio * tm.number_of_clauses)))
        if n_small % 2 == 1:
            n_small += 1
        t_small = max(1, int(round(tm.initial_T * ratio)))

        round_jobs = [
            {
                "job_id": "finetune",
                "mode": "checkpoint",
                "checkpoint": ckpt_clean_final,
                "mask": ticket_mask,
                "number_of_clauses": tm.number_of_clauses,
                "T": tm.initial_T,
                "s": tm.s,
                "weighted_clauses": tm.weighted_clauses,
                "max_included_literals": tm.max_included_literals,
                "seed": tm.seed,
                "retrain_epochs": retrain_epochs,
            },
            {
                "job_id": "rewind_early",
                "mode": "checkpoint",
                "checkpoint": checkpoints["early"],
                "mask": ticket_mask,
                "number_of_clauses": tm.number_of_clauses,
                "T": tm.initial_T,
                "s": tm.s,
                "weighted_clauses": tm.weighted_clauses,
                "max_included_literals": tm.max_included_literals,
                "seed": tm.seed,
                "retrain_epochs": retrain_epochs,
            },
            {
                "job_id": "rewind_init",
                "mode": "checkpoint",
                "checkpoint": checkpoints["init"],
                "mask": ticket_mask,
                "number_of_clauses": tm.number_of_clauses,
                "T": tm.initial_T,
                "s": tm.s,
                "weighted_clauses": tm.weighted_clauses,
                "max_included_literals": tm.max_included_literals,
                "seed": tm.seed,
                "retrain_epochs": retrain_epochs,
            },
            {
                "job_id": "random_reinit",
                "mode": "random_reinit",
                "checkpoint": checkpoints["init"],
                "mask": ticket_mask,
                "random_reinit_seed": 1000 + r_idx,
                "number_of_clauses": tm.number_of_clauses,
                "T": tm.initial_T,
                "s": tm.s,
                "weighted_clauses": tm.weighted_clauses,
                "max_included_literals": tm.max_included_literals,
                "seed": tm.seed,
                "retrain_epochs": retrain_epochs,
            },
            {
                "job_id": "random_mask",
                "mode": "checkpoint",
                "checkpoint": checkpoints["early"],
                "mask": rand_mask,
                "number_of_clauses": tm.number_of_clauses,
                "T": tm.initial_T,
                "s": tm.s,
                "weighted_clauses": tm.weighted_clauses,
                "max_included_literals": tm.max_included_literals,
                "seed": tm.seed,
                "retrain_epochs": retrain_epochs,
            },
        ]
        if run_scratch_baselines:
            round_jobs.append(
                {
                    "job_id": "scratch",
                    "mode": "scratch",
                    "number_of_clauses": n_small,
                    "T": t_small,
                    "s": tm.s,
                    "weighted_clauses": tm.weighted_clauses,
                    "max_included_literals": tm.max_included_literals,
                    "seed": tm.seed,
                    "retrain_epochs": retrain_epochs,
                }
            )

        if executor is not None:
            res_map = {
                res["job_id"]: res
                for res in executor.map(_parallel_retrain_worker, round_jobs)
            }
        else:
            _init_worker_data(X_train, Y_train, X_val, Y_val, X_test, Y_test)
            res_map = {
                j_spec["job_id"]: _parallel_retrain_worker(j_spec)
                for j_spec in round_jobs
            }

        ft_val, ft_test = res_map["finetune"]["best_val"], res_map["finetune"]["best_test"]
        rw_early_val, rw_early_test = (
            res_map["rewind_early"]["best_val"],
            res_map["rewind_early"]["best_test"],
        )
        rw_init_test = res_map["rewind_init"]["best_test"]
        rand_state_test = res_map["random_reinit"]["best_test"]
        rand_mask_test = res_map["random_mask"]["best_test"]
        scratch_test = res_map["scratch"]["best_test"] if run_scratch_baselines else 0.0

        if rw_early_val >= ft_val:
            winning_val, winning_test = rw_early_val, rw_early_test
        else:
            winning_val, winning_test = ft_val, ft_test

        passes_1se = bool(winning_val >= (dense_val_acc - sigma_val))
        passes_95ci = bool(winning_val >= (dense_val_acc - ci95_val))

        round_entry = {
            "round": r_idx,
            "level_name": level_label,
            "target_ratio": ratio,
            "actual_retention_ratio": ratio,
            "active_clauses": stats.active_clauses,
            "model_size_kb": stats.model_size_kb,
            "compression_ratio": round(
                dense_stats.model_size_kb / max(stats.model_size_kb, 1e-6), 2
            ),
            "mean_literal_length": stats.mean_literal_length,
            "oneshot_val_acc": round(oneshot_val, 3),
            "oneshot_test_acc": round(oneshot_test, 3),
            "oneshot_renorm_test_acc": round(oneshot_renorm_test, 3),
            "rewind_early_val_acc": rw_early_val,
            "rewind_early_test_acc": rw_early_test,
            "rewind_init_test_acc": rw_init_test,
            "finetune_val_acc": ft_val,
            "finetune_test_acc": ft_test,
            "random_reinit_test_acc": rand_state_test,
            "random_mask_retrain_test_acc": rand_mask_test,
            "scratch_small_test_acc": scratch_test,
            "winning_ticket_val_acc": winning_val,
            "winning_ticket_test_acc": winning_test,
            "passes_1se_rule": passes_1se,
            "passes_95ci_rule": passes_95ci,
        }
        rounds_report.append(round_entry)

        status_tag = "1-SE PASS" if passes_1se else ("95%-CI PASS" if passes_95ci else "BELOW 95%-CI")
        if verbose:
            print(
                f"  [Round {r_idx} | {level_label:<42}] Keep {ratio*100:4.1f}% "
                f"({stats.active_clauses:5d} cls, {stats.model_size_kb:6.1f} KB) | "
                f"OneShot: {oneshot_renorm_test:.2f}% | Rewind(e_r): {rw_early_test:.2f}% | "
                f"Rewind(0): {rw_init_test:.2f}% | FineTune: {ft_test:.2f}% | "
                f"RandMask: {rand_mask_test:.2f}% | Scratch: {scratch_test:.2f}% [{status_tag}]",
                flush=True,
            )

    # Identify the smallest data-driven ticket passing the 1-SE rule (and 95%-CI rule)
    passing_1se = [r for r in rounds_report if r["passes_1se_rule"]]
    optimal_1se_ticket = min(passing_1se, key=lambda r: r["active_clauses"])
    passing_95ci = [r for r in rounds_report if r["passes_95ci_rule"]]
    optimal_95ci_ticket = min(passing_95ci, key=lambda r: r["active_clauses"])

    return {
        "dense_baseline": {
            "total_clauses": dense_stats.total_clauses,
            "model_size_kb": dense_stats.model_size_kb,
            "val_accuracy": round(dense_val_acc, 3),
            "test_accuracy": round(dense_test_acc, 3),
            "stage1_removed": stage1_counts,
            "certified_lossless_removed": cert_removed,
            "sigma_val_pct": sigma_val,
            "ci95_val_pct": ci95_val,
        },
        "optimal_1se_ticket": optimal_1se_ticket,
        "optimal_95ci_ticket": optimal_95ci_ticket,
        "scoring_method": scoring_method,
        "iterative_tickets": rounds_report,
    }


def compare_optimization_methods(
    tm: PrunableTMClassifier,
    checkpoints: Dict[str, TMCheckpoint],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    retrain_epochs: int = 50,
    executor: Optional[ProcessPoolExecutor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmarks all clause optimization techniques head-to-head at the 4 parameter-free,
    data-driven capacities derived from the empirical Margin Contribution Score distribution:
      - Capacity 1: `L3_Participation_Ratio` (N_PR)
      - Capacity 2: `L4_Kneedle_AboveMean` (1st-order Kneedle)
      - Capacity 3: `L5_Recursive_Core` (2nd-order Kneedle)
      - Capacity 4: `L6_Minimal_Backbone` (3rd-order Kneedle)
    """
    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    dense_val_acc = tm.evaluate_accuracy(X_val, Y_val)
    stage1_mask, _ = rule_based_canonicalization(tm, merge_duplicates=True)
    mask_l2, _ = length_based_filter(tm, stage1_mask, min_literals=2)

    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)
    C_val = tm.evaluate_clause_outputs(X_val)

    scores_s1 = compute_clause_scores(
        tm, X_val, Y_val, base_mask=stage1_mask, clause_outputs=C_val
    )
    scores_l2 = compute_clause_scores(
        tm, X_val, Y_val, base_mask=mask_l2, clause_outputs=C_val
    )

    cap_info = compute_data_driven_capacities(
        scores=scores_s1["mcs_hybrid"],
        base_mask=stage1_mask,
        val_accuracy_pct=dense_val_acc,
        n_val_samples=X_val.shape[0],
    )

    total_cls = float(tm.number_of_classes * tm.number_of_clauses)
    data_driven_budgets = []
    for key_name in ("participation_pr", "kneedle_1", "kneedle_2_core", "kneedle_3_min"):
        r_val = round(float(np.sum(cap_info["masks"][key_name])) / total_cls, 4)
        if r_val not in data_driven_budgets:
            data_driven_budgets.append(r_val)
    data_driven_budgets.sort(reverse=True)

    if verbose:
        print(
            f"\n=== Comparing Clause Optimization Methods at Data-Driven Capacities "
            f"({[round(b*100, 1) for b in data_driven_budgets]}%, Retrain={retrain_epochs} epochs) ===",
            flush=True,
        )

    methods_spec = [
        ("TM-Ticket (Parameter-Free MCS*)", scores_s1["mcs_hybrid"], stage1_mask, True),
        ("MCS-Boundary (Margin Score)", scores_s1["mcs"], stage1_mask, True),
        ("Discriminative-Margin (w*[rho+ - rho-])", scores_s1["discriminative"], stage1_mask, True),
        ("Vote-Impact (|w|*rho)", scores_s1["vote_impact"], stage1_mask, True),
        ("Paper Stage 1+3 (Firing Rate, Global)", scores_s1["firing_rate"], stage1_mask, False),
        ("Paper Stage 1+2+3 (L_min=2 + Firing)", scores_l2["firing_rate"], mask_l2, False),
        ("Weight-Magnitude (|w|)", scores_s1["weight_mag"], stage1_mask, True),
        ("Random-Pruning (Baseline)", scores_s1["random"], stage1_mask, True),
    ]

    pending_entries: List[Dict[str, Any]] = []
    parallel_jobs: List[Dict[str, Any]] = []

    for ratio in data_driven_budgets:
        for m_idx, (method_name, score_arr, b_mask, balanced) in enumerate(methods_spec):
            mask = select_ticket_mask(
                scores=score_arr,
                base_mask=b_mask,
                target_retention_ratio=ratio,
                class_balanced=balanced,
                prune_negative_scores=False,
            )

            tm.rewind_to_checkpoint(
                ckpt_clean_final, retention_mask=mask, rescale_T_mode="proportional"
            )
            os_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)
            tm.update_vote_renormalization()
            os_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)
            best_os = os_renorm_test if balanced else os_test

            stats = compute_model_stats(tm, mask=mask)
            job_id = f"comp_{ratio}_{m_idx}"

            pending_entries.append(
                {
                    "job_id": job_id,
                    "budget_ratio": ratio,
                    "method": method_name,
                    "mask": mask,
                    "active_clauses": stats.active_clauses,
                    "model_size_kb": stats.model_size_kb,
                    "mean_literal_length": stats.mean_literal_length,
                    "oneshot_raw_test_acc": round(os_test, 3),
                    "oneshot_renorm_test_acc": round(os_renorm_test, 3),
                    "oneshot_best_test_acc": round(best_os, 3),
                }
            )
            parallel_jobs.append(
                {
                    "job_id": job_id,
                    "mode": "checkpoint",
                    "checkpoint": checkpoints["early"],
                    "mask": mask,
                    "number_of_clauses": tm.number_of_clauses,
                    "T": tm.initial_T,
                    "s": tm.s,
                    "weighted_clauses": tm.weighted_clauses,
                    "max_included_literals": tm.max_included_literals,
                    "seed": tm.seed,
                    "retrain_epochs": retrain_epochs,
                }
            )

    res_map: Dict[str, Dict[str, Any]] = {}
    if executor is not None:
        for res in executor.map(_parallel_retrain_worker, parallel_jobs):
            res_map[res["job_id"]] = res
    else:
        _init_worker_data(X_train, Y_train, X_val, Y_val, X_test, Y_test)
        for j_spec in parallel_jobs:
            res = _parallel_retrain_worker(j_spec)
            res_map[res["job_id"]] = res

    comparison_results: List[Dict[str, Any]] = []
    current_printed_ratio = None

    for item in pending_entries:
        ratio = item["budget_ratio"]
        if verbose and ratio != current_printed_ratio:
            print(
                f"\n  --- Data-Driven Clause Capacity: {ratio*100:.1f}% ({item['active_clauses']} clauses) ---",
                flush=True,
            )
            current_printed_ratio = ratio

        lth_test = res_map[item["job_id"]]["best_test"]

        entry = {
            "budget_ratio": ratio,
            "method": item["method"],
            "active_clauses": item["active_clauses"],
            "model_size_kb": item["model_size_kb"],
            "mean_literal_length": item["mean_literal_length"],
            "oneshot_raw_test_acc": item["oneshot_raw_test_acc"],
            "oneshot_renorm_test_acc": item["oneshot_renorm_test_acc"],
            "oneshot_best_test_acc": item["oneshot_best_test_acc"],
            "lottery_ticket_retrain_test_acc": lth_test,
        }
        comparison_results.append(entry)

        if verbose:
            print(
                f"    {entry['method']:<38} | Clauses: {entry['active_clauses']:5d} | "
                f"One-Shot: {entry['oneshot_best_test_acc']:6.2f}% | LTH-Retrain: {lth_test:6.2f}%",
                flush=True,
            )

    return {
        "data_driven_budgets": data_driven_budgets,
        "method_comparison": comparison_results,
    }
