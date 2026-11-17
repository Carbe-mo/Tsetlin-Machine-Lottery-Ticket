"""
Lottery Ticket Discovery & Multi-Method Optimization Benchmark (`tm_ticket/lottery.py`).

Implements three core experimental engines on top of `tmu`:
  1. `run_iterative_lottery_ticket`:
     Iterative Magnitude/Margin Pruning (IMP-TM) across a schedule of sparsity ratios,
     comparing One-Shot, Rewind-Early (Theta_{e_r}), Rewind-Init (Theta_0), Fine-Tune,
     Random-Mask Ticket, Random-State Reinit, and Scratch-Small TM.
  2. `discover_multiple_disjoint_tickets`:
     Proves that a SINGLE trained Tsetlin Machine contains MULTIPLE mutually disjoint
     winning lottery tickets (T_1, T_2, ..., T_M with T_a intersect T_b = empty set),
     reporting each ticket's standalone accuracy, structural stats, pairwise Jaccard
     overlap (0.0), and multi-ticket ensemble accuracy.
  3. `compare_optimization_methods`:
     Evaluates all clause optimization criteria head-to-head (MCS-Hybrid, Pure MCS,
     Discriminative Margin, Vote Impact, Firing-Rate Sanitization, Length+Firing
     Sanitization, Weight Magnitude, and Random Pruning) in both One-Shot and
     Lottery-Ticket Rewind+Retrain regimes.

Supports multi-core parallel retraining (`num_workers > 1`) via `ProcessPoolExecutor`
so 50-epoch retraining across dozens of tickets runs concurrently on multi-core servers.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import PrunableTMClassifier, TMCheckpoint
from .sanitization import (
    compute_clause_scores,
    compute_model_stats,
    length_based_filter,
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
    Executes a single ticket retraining job in a CPU worker process.
    """
    X_train = _WORKER_DATA["X_train"]
    Y_train = _WORKER_DATA["Y_train"]
    X_val = _WORKER_DATA["X_val"]
    Y_val = _WORKER_DATA["Y_val"]
    X_test = _WORKER_DATA["X_test"]
    Y_test = _WORKER_DATA["Y_test"]

    tm = PrunableTMClassifier(
        number_of_clauses=job["number_of_clauses"],
        T=job["T"],
        s=job["s"],
        platform="CPU",
        weighted_clauses=job["weighted_clauses"],
        max_included_literals=job.get("max_included_literals"),
        seed=job["seed"],
    )
    tm.init(X_train, Y_train)

    mode = job["mode"]
    mask = job.get("mask")
    retrain_epochs = job["retrain_epochs"]

    if mode == "checkpoint":
        ckpt: TMCheckpoint = job["checkpoint"]
        tm.rewind_to_checkpoint(
            ckpt, retention_mask=mask, rescale_T_mode="proportional"
        )
    elif mode == "random_reinit":
        tm.reinitialize_random_states(
            retention_mask=mask,
            seed=job["random_reinit_seed"],
            init_include_prob=0.02,
            rescale_T_mode="proportional",
        )
    elif mode == "scratch":
        pass
    else:
        raise ValueError(f"Unknown worker job mode: {mode}")

    best_val, best_test, best_scores = _retrain_and_eval(
        tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
    )
    final_ckpt = tm.get_checkpoint(epoch=retrain_epochs) if job.get("return_ckpt", False) else None

    return {
        "job_id": job["job_id"],
        "best_val": best_val,
        "best_test": best_test,
        "best_scores": best_scores if job.get("return_scores", False) else None,
        "final_ckpt": final_ckpt,
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
    """
    Initializes and trains a dense PrunableTMClassifier for `epochs` epochs,
    capturing TA state & weight checkpoints at:
      - 'init' (epoch 0, before any training updates)
      - 'early' (after `rewind_epoch` epochs of training)
      - 'final' (after `epochs` epochs of training, plus best-validation checkpoint)
    """
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
    """
    Retrains `tm` (with its active `retention_mask`) for `retrain_epochs` epochs,
    tracking the best validation checkpoint, and returns (best_val_acc, test_acc_at_best_val, test_scores).
    """
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
    num_tickets: int = 4,
    ticket_ratio: float = 0.20,
    retrain_epochs: int = 50,
    min_literals: int = 1,
    scoring_method: str = "mcs_hybrid",
    executor: Optional[ProcessPoolExecutor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Discovers `num_tickets` MUTUALLY DISJOINT winning lottery tickets from a single
    trained Tsetlin Machine (Ticket #1, Ticket #2, ..., Ticket #M), where each ticket
    retains `ticket_ratio` (e.g. 20%) of the clauses per class and has ZERO overlap
    with any previously extracted ticket (T_a intersect T_b = empty set).
    """
    if verbose:
        print(
            f"\n=== Discovering {num_tickets} Mutually Disjoint Winning Tickets "
            f"(each {ticket_ratio*100:.0f}% of bank, Retrain={retrain_epochs} epochs) ===",
            flush=True,
        )

    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    dense_val_acc = tm.evaluate_accuracy(X_val, Y_val)
    dense_test_acc = tm.evaluate_accuracy(X_test, Y_test)
    dense_stats = compute_model_stats(tm)

    stage1_mask, stage1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    if min_literals > 0:
        clean_mask, short_rem = length_based_filter(
            tm, stage1_mask, min_literals=min_literals
        )
    else:
        clean_mask, short_rem = stage1_mask, 0

    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)

    C_val = tm.evaluate_clause_outputs(X_val)
    all_scores = compute_clause_scores(
        tm, X_val, Y_val, base_mask=clean_mask, clause_outputs=C_val
    )
    primary_scores = all_scores[scoring_method]

    used_clauses_mask = np.zeros_like(clean_mask, dtype=np.int32)
    ticket_masks: List[np.ndarray] = []
    oneshot_meta: List[Dict[str, Any]] = []

    for t_idx in range(1, num_tickets + 1):
        t_mask = select_ticket_mask(
            scores=primary_scores,
            base_mask=clean_mask,
            target_retention_ratio=ticket_ratio,
            class_balanced=True,
            prune_negative_scores=False,
            exclude_mask=used_clauses_mask,
        )
        overlap_with_previous = int(np.sum(t_mask & used_clauses_mask))
        assert overlap_with_previous == 0, "Disjoint ticket masks must have 0 overlap!"
        used_clauses_mask = used_clauses_mask | t_mask
        ticket_masks.append(t_mask)

        tm.rewind_to_checkpoint(
            ckpt_clean_final, retention_mask=t_mask, rescale_T_mode="proportional"
        )
        oneshot_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=False)
        oneshot_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)

        tm.update_vote_renormalization()
        oneshot_renorm_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=True)
        oneshot_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)
        t_stats = compute_model_stats(tm, mask=t_mask)

        oneshot_meta.append(
            {
                "t_idx": t_idx,
                "t_mask": t_mask,
                "t_stats": t_stats,
                "overlap": overlap_with_previous,
                "oneshot_val": oneshot_val,
                "oneshot_test": oneshot_test,
                "oneshot_renorm_val": oneshot_renorm_val,
                "oneshot_renorm_test": oneshot_renorm_test,
            }
        )

    # Dispatch all 2 * num_tickets retraining runs in parallel if executor is provided
    retrain_results: Dict[str, Dict[str, Any]] = {}
    if executor is not None:
        jobs = []
        for meta in oneshot_meta:
            t_idx = meta["t_idx"]
            t_mask = meta["t_mask"]
            jobs.append(
                {
                    "job_id": f"ft_{t_idx}",
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
                    "job_id": f"rw_{t_idx}",
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
        for res in executor.map(_parallel_retrain_worker, jobs):
            retrain_results[res["job_id"]] = res

    tickets_report: List[Dict[str, Any]] = []
    ensemble_test_scores = np.zeros((X_test.shape[0], tm.number_of_classes), dtype=np.float64)

    for meta in oneshot_meta:
        t_idx = meta["t_idx"]
        t_mask = meta["t_mask"]
        t_stats = meta["t_stats"]

        if executor is not None:
            ft_res = retrain_results[f"ft_{t_idx}"]
            rw_res = retrain_results[f"rw_{t_idx}"]
            ft_val, ft_test, ft_scores = ft_res["best_val"], ft_res["best_test"], ft_res["best_scores"]
            rw_val, rw_test, rw_scores = rw_res["best_val"], rw_res["best_test"], rw_res["best_scores"]
        else:
            tm.rewind_to_checkpoint(
                ckpt_clean_final, retention_mask=t_mask, rescale_T_mode="proportional"
            )
            ft_val, ft_test, ft_scores = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )
            tm.rewind_to_checkpoint(
                checkpoints["early"], retention_mask=t_mask, rescale_T_mode="proportional"
            )
            rw_val, rw_test, rw_scores = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )

        if rw_val >= ft_val:
            best_ticket_val, best_ticket_test, best_ticket_scores = rw_val, rw_test, rw_scores
            best_regime = "rewind_early"
        else:
            best_ticket_val, best_ticket_test, best_ticket_scores = ft_val, ft_test, ft_scores
            best_regime = "finetune"

        norm_factor = np.std(best_ticket_scores)
        if norm_factor > 0:
            ensemble_test_scores += best_ticket_scores / norm_factor
        else:
            ensemble_test_scores += best_ticket_scores

        ensemble_test_acc = float(
            np.mean(np.argmax(ensemble_test_scores, axis=1) == Y_test) * 100.0
        )

        ticket_entry = {
            "ticket_id": f"Ticket #{t_idx}",
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
        tickets_report.append(ticket_entry)

        if verbose:
            print(
                f"  [{ticket_entry['ticket_id']}] Clauses: {t_stats.active_clauses} "
                f"({t_stats.model_size_kb:.1f} KB, Overlap={meta['overlap']}) | "
                f"One-Shot: {meta['oneshot_renorm_test']:.2f}% | "
                f"Rewind-Retrain: {rw_test:.2f}% | "
                f"FineTune: {ft_test:.2f}% | "
                f"Winning Ticket Acc: {best_ticket_test:.2f}% | "
                f"Ensemble(1..{t_idx}): {ensemble_test_acc:.2f}%",
                flush=True,
            )

    pairwise_jaccard = np.zeros((num_tickets, num_tickets), dtype=np.float64)
    for i in range(num_tickets):
        for j in range(num_tickets):
            inter = np.sum(ticket_masks[i] & ticket_masks[j])
            union = np.sum(ticket_masks[i] | ticket_masks[j])
            pairwise_jaccard[i, j] = float(inter) / float(max(union, 1))

    return {
        "dense_baseline": {
            "total_clauses": dense_stats.total_clauses,
            "model_size_kb": dense_stats.model_size_kb,
            "val_accuracy": round(dense_val_acc, 3),
            "test_accuracy": round(dense_test_acc, 3),
            "stage1_removed": stage1_counts,
            "short_removed": short_rem,
        },
        "ticket_ratio": ticket_ratio,
        "num_disjoint_tickets": num_tickets,
        "scoring_method": scoring_method,
        "pairwise_jaccard_matrix": pairwise_jaccard.tolist(),
        "disjoint_tickets": tickets_report,
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
    retention_schedule: Optional[List[float]] = None,
    retrain_epochs: int = 50,
    min_literals: int = 1,
    scoring_method: str = "mcs_hybrid",
    run_scratch_baselines: bool = True,
    executor: Optional[ProcessPoolExecutor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs the full Iterative Lottery-Ticket Loop (IMP-TM) across a schedule of
    clause retention ratios (e.g., [1.0, 0.80, 0.60, 0.40, 0.30, 0.20, 0.15, 0.10]).
    """
    if retention_schedule is None:
        retention_schedule = [1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]

    if verbose:
        print(
            f"\n=== Running Iterative Lottery-Ticket Loop "
            f"(Schedule: {retention_schedule}, Retrain={retrain_epochs} epochs) ===",
            flush=True,
        )

    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    dense_val_acc = tm.evaluate_accuracy(X_val, Y_val)
    dense_test_acc = tm.evaluate_accuracy(X_test, Y_test)
    dense_stats = compute_model_stats(tm)

    # Apply Stage 1 Lossless Canonicalization first
    stage1_mask, stage1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    if min_literals > 0:
        current_mask, short_rem = length_based_filter(
            tm, stage1_mask, min_literals=min_literals
        )
    else:
        current_mask, short_rem = stage1_mask, 0

    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)
    rounds_report: List[Dict[str, Any]] = []
    rng = np.random.RandomState(777)

    for r_idx, ratio in enumerate(retention_schedule):
        if ratio >= 1.0:
            # Round 0: Stage 1 Lossless Canonicalization
            ticket_mask = stage1_mask.copy()
            harmful_count = 0
        else:
            harmful_count = 0
            C_val_curr = tm.evaluate_clause_outputs(X_val)
            scores_curr = compute_clause_scores(
                tm, X_val, Y_val, base_mask=current_mask, clause_outputs=C_val_curr
            )[scoring_method]
            ticket_mask = select_ticket_mask(
                scores=scores_curr,
                base_mask=current_mask,
                target_retention_ratio=ratio,
                class_balanced=True,
                prune_negative_scores=False,
            )
            current_mask = ticket_mask.copy()

        # 1. One-Shot Evaluation
        tm.rewind_to_checkpoint(
            ckpt_clean_final, retention_mask=ticket_mask, rescale_T_mode="proportional"
        )
        oneshot_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=False)
        oneshot_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)

        tm.update_vote_renormalization()
        oneshot_renorm_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=True)
        oneshot_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)

        stats = compute_model_stats(
            tm,
            mask=ticket_mask,
            stage1_counts=stage1_counts if r_idx == 0 else None,
            short_removed=short_rem if r_idx == 0 else 0,
            harmful_removed=harmful_count,
        )

        if ratio >= 1.0:
            round_entry = {
                "round": r_idx,
                "target_ratio": ratio,
                "actual_retention_ratio": round(
                    stats.active_clauses / float(dense_stats.total_clauses), 4
                ),
                "active_clauses": stats.active_clauses,
                "model_size_kb": stats.model_size_kb,
                "mean_literal_length": stats.mean_literal_length,
                "oneshot_val_acc": round(oneshot_val, 3),
                "oneshot_test_acc": round(oneshot_test, 3),
                "oneshot_renorm_test_acc": round(oneshot_renorm_test, 3),
                "rewind_early_test_acc": round(oneshot_test, 3),
                "rewind_init_test_acc": round(dense_test_acc, 3),
                "finetune_test_acc": round(oneshot_test, 3),
                "random_reinit_test_acc": round(dense_test_acc, 3),
                "random_mask_retrain_test_acc": round(dense_test_acc, 3),
                "scratch_small_test_acc": round(dense_test_acc, 3),
                "winning_ticket_test_acc": round(max(oneshot_test, oneshot_renorm_test), 3),
            }
            rounds_report.append(round_entry)
            if verbose:
                print(
                    f"  [Round {r_idx} | Stage 1 Lossless] Active: {stats.active_clauses}/{dense_stats.total_clauses} "
                    f"({stats.model_size_kb:.1f} KB) | Dense Acc: {dense_test_acc:.2f}% -> "
                    f"Sanitized Acc: {oneshot_test:.2f}% (Renorm: {oneshot_renorm_test:.2f}%)",
                    flush=True,
                )
            continue

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

        if executor is not None:
            # Run all 6 retraining regimes for this round concurrently across CPU workers!
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
                    "return_ckpt": True,
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
                    "return_ckpt": True,
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
            res_map = {res["job_id"]: res for res in executor.map(_parallel_retrain_worker, round_jobs)}
            ft_val, ft_test, ckpt_after_ft = (
                res_map["finetune"]["best_val"],
                res_map["finetune"]["best_test"],
                res_map["finetune"]["final_ckpt"],
            )
            rw_early_val, rw_early_test, ckpt_after_rw_early = (
                res_map["rewind_early"]["best_val"],
                res_map["rewind_early"]["best_test"],
                res_map["rewind_early"]["final_ckpt"],
            )
            rw_init_test = res_map["rewind_init"]["best_test"]
            rand_state_test = res_map["random_reinit"]["best_test"]
            rand_mask_test = res_map["random_mask"]["best_test"]
            scratch_test = res_map["scratch"]["best_test"] if run_scratch_baselines else 0.0
        else:
            ft_val, ft_test, _ = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )
            ckpt_after_ft = tm.get_checkpoint(epoch=checkpoints["final"].epoch + retrain_epochs)

            tm.rewind_to_checkpoint(
                checkpoints["early"], retention_mask=ticket_mask, rescale_T_mode="proportional"
            )
            rw_early_val, rw_early_test, _ = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )
            ckpt_after_rw_early = tm.get_checkpoint(epoch=checkpoints["early"].epoch + retrain_epochs)

            tm.rewind_to_checkpoint(
                checkpoints["init"], retention_mask=ticket_mask, rescale_T_mode="proportional"
            )
            _, rw_init_test, _ = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )

            tm.reinitialize_random_states(
                retention_mask=ticket_mask,
                seed=1000 + r_idx,
                init_include_prob=0.02,
                rescale_T_mode="proportional",
            )
            _, rand_state_test, _ = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )

            tm.rewind_to_checkpoint(
                checkpoints["early"], retention_mask=rand_mask, rescale_T_mode="proportional"
            )
            _, rand_mask_test, _ = _retrain_and_eval(
                tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
            )

            scratch_test = 0.0
            if run_scratch_baselines:
                small_tm = PrunableTMClassifier(
                    number_of_clauses=n_small,
                    T=t_small,
                    s=tm.s,
                    platform=tm.platform,
                    weighted_clauses=tm.weighted_clauses,
                    seed=tm.seed,
                )
                small_tm.init(X_train, Y_train)
                _, scratch_test, _ = _retrain_and_eval(
                    small_tm,
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    X_test,
                    Y_test,
                    retrain_epochs=retrain_epochs,
                )
                for k_cls in range(small_tm.number_of_classes):
                    cb_small = small_tm.clause_banks[k_cls]
                    if hasattr(cb_small, "device") and cb_small.device is not None:
                        try:
                            cb_small.device.cleanup()
                        except Exception:
                            pass
                        cb_small.device = None
                del small_tm

        if rw_early_val >= ft_val:
            tm.rewind_to_checkpoint(
                ckpt_after_rw_early, retention_mask=ticket_mask, rescale_T_mode="proportional"
            )
            winning_test = rw_early_test
        else:
            tm.rewind_to_checkpoint(
                ckpt_after_ft, retention_mask=ticket_mask, rescale_T_mode="proportional"
            )
            winning_test = ft_test

        round_entry = {
            "round": r_idx,
            "target_ratio": ratio,
            "actual_retention_ratio": round(
                stats.active_clauses / float(dense_stats.total_clauses), 4
            ),
            "active_clauses": stats.active_clauses,
            "model_size_kb": stats.model_size_kb,
            "mean_literal_length": stats.mean_literal_length,
            "oneshot_val_acc": round(oneshot_val, 3),
            "oneshot_test_acc": round(oneshot_test, 3),
            "oneshot_renorm_test_acc": round(oneshot_renorm_test, 3),
            "rewind_early_test_acc": rw_early_test,
            "rewind_init_test_acc": rw_init_test,
            "finetune_test_acc": ft_test,
            "random_reinit_test_acc": rand_state_test,
            "random_mask_retrain_test_acc": rand_mask_test,
            "scratch_small_test_acc": scratch_test,
            "winning_ticket_test_acc": winning_test,
        }
        rounds_report.append(round_entry)

        if verbose:
            print(
                f"  [Round {r_idx} | Keep {ratio*100:4.1f}% ({stats.active_clauses} cls, {stats.model_size_kb:.1f} KB)] "
                f"OneShot: {oneshot_renorm_test:.2f}% | "
                f"Rewind(e_r): {rw_early_test:.2f}% | "
                f"Rewind(0): {rw_init_test:.2f}% | "
                f"FineTune: {ft_test:.2f}% | "
                f"RandMask: {rand_mask_test:.2f}% | "
                f"Scratch: {scratch_test:.2f}%",
                flush=True,
            )

    return {
        "dense_baseline": {
            "total_clauses": dense_stats.total_clauses,
            "model_size_kb": dense_stats.model_size_kb,
            "val_accuracy": round(dense_val_acc, 3),
            "test_accuracy": round(dense_test_acc, 3),
            "stage1_removed": stage1_counts,
            "short_removed": short_rem,
        },
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
    budget_ratios: Optional[List[float]] = None,
    retrain_epochs: int = 50,
    executor: Optional[ProcessPoolExecutor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmarks all clause optimization techniques head-to-head across multiple
    clause budgets (`budget_ratios`), reporting both One-Shot Post-Training Accuracy
    and Lottery-Ticket (Rewind + Retrain) Accuracy.
    """
    if budget_ratios is None:
        budget_ratios = [0.50, 0.30, 0.20, 0.10]

    if verbose:
        print(
            f"\n=== Comparing All Clause Optimization Techniques "
            f"(Budgets: {budget_ratios}, Retrain={retrain_epochs} epochs) ===",
            flush=True,
        )

    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    stage1_mask, _ = rule_based_canonicalization(tm, merge_duplicates=True)
    mask_l1, _ = length_based_filter(tm, stage1_mask, min_literals=1)
    mask_l2, _ = length_based_filter(tm, stage1_mask, min_literals=2)

    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)
    C_val = tm.evaluate_clause_outputs(X_val)

    scores_l1 = compute_clause_scores(
        tm, X_val, Y_val, base_mask=mask_l1, clause_outputs=C_val
    )
    scores_l2 = compute_clause_scores(
        tm, X_val, Y_val, base_mask=mask_l2, clause_outputs=C_val
    )

    methods_spec = [
        ("TM-Ticket (MCS-Hybrid + C1 + C3)", scores_l1["mcs_hybrid"], mask_l1, True),
        ("MCS-Boundary (Pure Margin Score)", scores_l1["mcs"], mask_l1, True),
        ("Discriminative-Margin (w*[rho+ - rho-])", scores_l1["discriminative"], mask_l1, True),
        ("Vote-Impact (|w|*rho)", scores_l1["vote_impact"], mask_l1, True),
        ("Paper Stage 1+3 (Firing Rate, Global)", scores_l1["firing_rate"], mask_l1, False),
        ("Paper Stage 1+2+3 (L_min=2 + Firing)", scores_l2["firing_rate"], mask_l2, False),
        ("Weight-Magnitude (|w|)", scores_l1["weight_mag"], mask_l1, True),
        ("Random-Pruning (Baseline)", scores_l1["random"], mask_l1, True),
    ]

    pending_entries: List[Dict[str, Any]] = []
    parallel_jobs: List[Dict[str, Any]] = []

    for ratio in budget_ratios:
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

    comparison_results: List[Dict[str, Any]] = []
    current_printed_ratio = None

    for item in pending_entries:
        ratio = item["budget_ratio"]
        if verbose and ratio != current_printed_ratio:
            print(f"\n  --- Target Clause Budget: {ratio*100:.0f}% ---", flush=True)
            current_printed_ratio = ratio

        if executor is not None:
            lth_test = res_map[item["job_id"]]["best_test"]
        else:
            tm.rewind_to_checkpoint(
                checkpoints["early"], retention_mask=item["mask"], rescale_T_mode="proportional"
            )
            _, lth_test, _ = _retrain_and_eval(
                tm,
                X_train,
                Y_train,
                X_val,
                Y_val,
                X_test,
                Y_test,
                retrain_epochs=retrain_epochs,
            )

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

    return {"method_comparison": comparison_results}
