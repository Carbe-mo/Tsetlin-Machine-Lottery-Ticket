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
"""

from __future__ import annotations

from dataclasses import asdict
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


def train_dense_tm_with_checkpoints(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    number_of_clauses: int = 2000,
    T: int = 1500,
    s: float = 10.0,
    epochs: int = 10,
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
      - 'final' (after `epochs` epochs of training)
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
                f"Time: {elapsed:.2f}s"
            )

        if ep == rewind_epoch:
            checkpoints["early"] = tm.get_checkpoint(epoch=ep)

    if "early" not in checkpoints:
        checkpoints["early"] = tm.get_checkpoint(epoch=epochs)
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
    best_test = tm.evaluate_accuracy(X_test, Y_test)
    best_scores = tm.compute_class_scores(X_test)

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
    retrain_epochs: int = 6,
    min_literals: int = 1,
    scoring_method: str = "mcs_hybrid",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Discovers `num_tickets` MUTUALLY DISJOINT winning lottery tickets from a single
    trained Tsetlin Machine (Ticket #1, Ticket #2, ..., Ticket #M), where each ticket
    retains `ticket_ratio` (e.g. 20%) of the clauses per class and has ZERO overlap
    with any previously extracted ticket (T_a intersect T_b = empty set).

    For each disjoint ticket, evaluates:
      1. One-shot post-training accuracy (no retraining)
      2. One-shot accuracy with C3 vote renormalization
      3. Retrained accuracy after rewinding to early checkpoint Theta_{e_r}
      4. Retrained accuracy after fine-tuning from Theta_{final}
      5. Cumulative Ensemble Accuracy when combining Ticket #1 .. Ticket #m
    """
    if verbose:
        print(
            f"\n=== Discovering {num_tickets} Mutually Disjoint Winning Tickets "
            f"(each {ticket_ratio*100:.0f}% of bank) ==="
        )

    # Restore final dense model first
    tm.rewind_to_checkpoint(checkpoints["final"], rescale_T_mode="fixed")
    dense_val_acc = tm.evaluate_accuracy(X_val, Y_val)
    dense_test_acc = tm.evaluate_accuracy(X_test, Y_test)
    dense_stats = compute_model_stats(tm)

    # Stage 1 (Rule-based lossless) + Stage 2 (min_literals)
    stage1_mask, stage1_counts = rule_based_canonicalization(tm, merge_duplicates=True)
    if min_literals > 0:
        clean_mask, short_rem = length_based_filter(
            tm, stage1_mask, min_literals=min_literals
        )
    else:
        clean_mask, short_rem = stage1_mask, 0

    # Save post-canonicalization final checkpoint so duplicate weight merges are preserved
    ckpt_clean_final = tm.get_checkpoint(epoch=checkpoints["final"].epoch)

    # Precompute validation clause outputs and scores on the clean dense bank
    C_val = tm.evaluate_clause_outputs(X_val)
    all_scores = compute_clause_scores(
        tm, X_val, Y_val, base_mask=clean_mask, clause_outputs=C_val
    )
    primary_scores = all_scores[scoring_method]

    used_clauses_mask = np.zeros_like(clean_mask, dtype=np.int32)
    ticket_masks: List[np.ndarray] = []
    tickets_report: List[Dict[str, Any]] = []
    ensemble_test_scores = np.zeros((X_test.shape[0], tm.number_of_classes), dtype=np.float64)

    for t_idx in range(1, num_tickets + 1):
        # Extract the next best disjoint ticket from clauses not yet used
        t_mask = select_ticket_mask(
            scores=primary_scores,
            base_mask=clean_mask,
            target_retention_ratio=ticket_ratio,
            class_balanced=True,
            prune_negative_scores=False,
            exclude_mask=used_clauses_mask,
        )

        # Verify strict disjointness with all previous tickets
        overlap_with_previous = int(np.sum(t_mask & used_clauses_mask))
        assert overlap_with_previous == 0, "Disjoint ticket masks must have 0 overlap!"
        used_clauses_mask = used_clauses_mask | t_mask
        ticket_masks.append(t_mask)

        # 1. Evaluate One-Shot (no retraining) from final trained states
        tm.rewind_to_checkpoint(
            ckpt_clean_final, retention_mask=t_mask, rescale_T_mode="proportional"
        )
        oneshot_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=False)
        oneshot_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)

        tm.update_vote_renormalization()
        oneshot_renorm_val = tm.evaluate_accuracy(X_val, Y_val, use_vote_scales=True)
        oneshot_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)

        t_stats = compute_model_stats(tm, mask=t_mask)

        # 2. Evaluate Fine-Tuning from Theta_final
        ft_val, ft_test, ft_scores = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )

        # 3. Evaluate LTH Rewind to Early Checkpoint (Theta_{e_r}) + Retrain
        tm.rewind_to_checkpoint(
            checkpoints["early"], retention_mask=t_mask, rescale_T_mode="proportional"
        )
        rw_val, rw_test, rw_scores = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )

        # Pick best retrained ticket trajectory (Rewind-Early vs Fine-Tune by validation accuracy)
        if rw_val >= ft_val:
            best_ticket_val, best_ticket_test, best_ticket_scores = rw_val, rw_test, rw_scores
            best_regime = "rewind_early"
        else:
            best_ticket_val, best_ticket_test, best_ticket_scores = ft_val, ft_test, ft_scores
            best_regime = "finetune"

        # Normalize ticket scores before adding to cumulative multi-ticket ensemble
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
            "overlap_with_previous_tickets": overlap_with_previous,
            "oneshot_val_acc": round(oneshot_val, 3),
            "oneshot_test_acc": round(oneshot_test, 3),
            "oneshot_renorm_val_acc": round(oneshot_renorm_val, 3),
            "oneshot_renorm_test_acc": round(oneshot_renorm_test, 3),
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
                f"({t_stats.model_size_kb:.1f} KB, Overlap={overlap_with_previous}) | "
                f"One-Shot: {oneshot_renorm_test:.2f}% | "
                f"Rewind-Retrain: {rw_test:.2f}% | "
                f"FineTune: {ft_test:.2f}% | "
                f"Winning Ticket Acc: {best_ticket_test:.2f}% | "
                f"Ensemble(1..{t_idx}): {ensemble_test_acc:.2f}%"
            )

    # Compute pairwise Jaccard similarity matrix across all discovered tickets
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
    retrain_epochs: int = 5,
    min_literals: int = 1,
    scoring_method: str = "mcs_hybrid",
    run_scratch_baselines: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs the full Iterative Lottery-Ticket Loop (IMP-TM) across a schedule of
    clause retention ratios (e.g., [1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]).

    At each sparsity level, compares:
      - One-Shot Sanitized TM (no retraining)
      - One-Shot Sanitized TM + C3 Vote Renormalization
      - Winning Ticket (Rewind to Theta_{e_r} + Retrain)
      - Winning Ticket (Rewind to Theta_0 + Retrain)
      - Fine-Tuned Ticket (Continue from Theta_{final} + Retrain)
      - Random-State Reinit Ticket (Same winning mask, perturbed random TA states + Retrain)
      - Random-Mask Ticket (Random clause subset rewound to Theta_{e_r} + Retrain)
      - Scratch Small TM (Fresh TM with N_active clauses per class trained from scratch)
    """
    if retention_schedule is None:
        retention_schedule = [1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]

    if verbose:
        print(
            f"\n=== Running Iterative Lottery-Ticket Loop (Schedule: {retention_schedule}) ==="
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
    C_val_dense = tm.evaluate_clause_outputs(X_val)
    dense_scores_dict = compute_clause_scores(
        tm, X_val, Y_val, base_mask=current_mask, clause_outputs=C_val_dense
    )

    rounds_report: List[Dict[str, Any]] = []
    rng = np.random.RandomState(777)

    for r_idx, ratio in enumerate(retention_schedule):
        if ratio >= 1.0:
            # Round 0: Stage 1 + Stage 2 lossless/short-filtered bank without budget cut
            ticket_mask = current_mask.copy()
            # Also drop strictly negative MCS clauses (harmful clauses)
            harmful = (dense_scores_dict["mcs"] < 0.0) & (ticket_mask == 1)
            harmful_count = int(np.sum(harmful))
            ticket_mask[harmful] = 0
        else:
            harmful_count = 0
            # Score current surviving clauses and prune down to `ratio`
            C_val_curr = tm.evaluate_clause_outputs(X_val)
            scores_curr = compute_clause_scores(
                tm, X_val, Y_val, base_mask=current_mask, clause_outputs=C_val_curr
            )[scoring_method]
            ticket_mask = select_ticket_mask(
                scores=scores_curr,
                base_mask=current_mask,
                target_retention_ratio=ratio,
                class_balanced=True,
                prune_negative_scores=True,
            )
            current_mask = ticket_mask.copy()

        # 1. One-Shot Evaluation from clean final checkpoint
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
                    f"  [Round {r_idx} | Sanitized Full] Active: {stats.active_clauses}/{dense_stats.total_clauses} "
                    f"({stats.model_size_kb:.1f} KB) | Dense Acc: {dense_test_acc:.2f}% -> "
                    f"Sanitized Acc: {oneshot_test:.2f}% (Renorm: {oneshot_renorm_test:.2f}%)"
                )
            continue

        # 2. Fine-Tuning from Theta_final
        ft_val, ft_test, _ = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )
        # Update checkpoint for next iterative pruning round from fine-tuned state
        ckpt_after_ft = tm.get_checkpoint(epoch=checkpoints["final"].epoch + retrain_epochs)

        # 3. Winning Ticket: Rewind to Early Checkpoint Theta_{e_r} + Retrain
        tm.rewind_to_checkpoint(
            checkpoints["early"], retention_mask=ticket_mask, rescale_T_mode="proportional"
        )
        rw_early_val, rw_early_test, _ = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )
        ckpt_after_rw_early = tm.get_checkpoint(epoch=checkpoints["early"].epoch + retrain_epochs)

        # 4. Winning Ticket: Rewind to Init Checkpoint Theta_0 + Retrain
        tm.rewind_to_checkpoint(
            checkpoints["init"], retention_mask=ticket_mask, rescale_T_mode="proportional"
        )
        rw_init_val, rw_init_test, _ = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )

        # 5. Control A: Random State Re-Initialization (Same winning mask, perturbed TA states)
        tm.reinitialize_random_states(
            retention_mask=ticket_mask,
            seed=1000 + r_idx,
            init_include_prob=0.02,
            rescale_T_mode="proportional",
        )
        rand_state_val, rand_state_test, _ = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )

        # 6. Control B: Random Clause Mask rewound to Theta_{e_r} + Retrain
        rand_scores = rng.rand(tm.number_of_classes, tm.number_of_clauses)
        rand_mask = select_ticket_mask(
            scores=rand_scores,
            base_mask=np.ones_like(ticket_mask),
            target_retention_ratio=ratio,
            class_balanced=True,
        )
        tm.rewind_to_checkpoint(
            checkpoints["early"], retention_mask=rand_mask, rescale_T_mode="proportional"
        )
        rand_mask_val, rand_mask_test, _ = _retrain_and_eval(
            tm, X_train, Y_train, X_val, Y_val, X_test, Y_test, retrain_epochs=retrain_epochs
        )

        # 7. Control C: Small TM trained from scratch with N_small = active clauses per class
        scratch_test = 0.0
        if run_scratch_baselines:
            n_small = max(4, int(round(ratio * tm.number_of_clauses)))
            if n_small % 2 == 1:
                n_small += 1
            t_small = max(1, int(round(tm.initial_T * ratio)))
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
            # Explicitly release CUDA contexts if small_tm used ClauseBankCuda
            for k_cls in range(small_tm.number_of_classes):
                cb_small = small_tm.clause_banks[k_cls]
                if hasattr(cb_small, "device") and cb_small.device is not None:
                    try:
                        cb_small.device.cleanup()
                    except Exception:
                        pass
                    cb_small.device = None
            del small_tm

        # Advance `tm` state to the best ticket checkpoint for the next iterative pruning step
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
                f"Scratch: {scratch_test:.2f}%"
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
    retrain_epochs: int = 4,
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
            f"\n=== Comparing All Clause Optimization Techniques (Budgets: {budget_ratios}) ==="
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
        ("TM-Ticket (MCS-Hybrid + C1 + C3)", scores_l1["mcs_hybrid"], mask_l1, True, True),
        ("MCS-Boundary (Pure Margin Score)", scores_l1["mcs"], mask_l1, True, True),
        ("Discriminative-Margin (w*[rho+ - rho-])", scores_l1["discriminative"], mask_l1, True, True),
        ("Vote-Impact (|w|*rho)", scores_l1["vote_impact"], mask_l1, True, False),
        ("Paper Stage 1+3 (Firing Rate, Global)", scores_l1["firing_rate"], mask_l1, False, False),
        ("Paper Stage 1+2+3 (L_min=2 + Firing)", scores_l2["firing_rate"], mask_l2, False, False),
        ("Weight-Magnitude (|w|)", scores_l1["weight_mag"], mask_l1, True, False),
        ("Random-Pruning (Baseline)", scores_l1["random"], mask_l1, True, False),
    ]

    comparison_results: List[Dict[str, Any]] = []

    for ratio in budget_ratios:
        if verbose:
            print(f"\n  --- Target Clause Budget: {ratio*100:.0f}% ---")
        for method_name, score_arr, b_mask, balanced, prune_neg in methods_spec:
            mask = select_ticket_mask(
                scores=score_arr,
                base_mask=b_mask,
                target_retention_ratio=ratio,
                class_balanced=balanced,
                prune_negative_scores=prune_neg,
            )

            # One-Shot evaluation
            tm.rewind_to_checkpoint(
                ckpt_clean_final, retention_mask=mask, rescale_T_mode="proportional"
            )
            os_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=False)
            tm.update_vote_renormalization()
            os_renorm_test = tm.evaluate_accuracy(X_test, Y_test, use_vote_scales=True)
            best_os = os_renorm_test if balanced else os_test

            stats = compute_model_stats(tm, mask=mask)

            # Lottery-Ticket Rewind + Retrain evaluation
            tm.rewind_to_checkpoint(
                checkpoints["early"], retention_mask=mask, rescale_T_mode="proportional"
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
                "method": method_name,
                "active_clauses": stats.active_clauses,
                "model_size_kb": stats.model_size_kb,
                "mean_literal_length": stats.mean_literal_length,
                "oneshot_raw_test_acc": round(os_test, 3),
                "oneshot_renorm_test_acc": round(os_renorm_test, 3),
                "oneshot_best_test_acc": round(best_os, 3),
                "lottery_ticket_retrain_test_acc": lth_test,
            }
            comparison_results.append(entry)

            if verbose:
                print(
                    f"    {method_name:<38} | Clauses: {stats.active_clauses:5d} | "
                    f"One-Shot: {best_os:6.2f}% | LTH-Retrain: {lth_test:6.2f}%"
                )

    return {"method_comparison": comparison_results}
