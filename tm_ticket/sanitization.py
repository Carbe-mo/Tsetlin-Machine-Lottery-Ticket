"""
Clause Sanitization & Importance Scoring Module (`tm_ticket/sanitization.py`).

Unifies all post-training and iterative clause optimization techniques:
  1. Stage 1 — Lossless Rule-Based Canonicalization:
     - Empty clause removal (l_{k,j} = 0)
     - Contradictory / tautological clause removal (x_i AND NOT x_i)
     - Encoding-invalid clause removal (thermometer / threshold violations)
     - Duplicate clause detection and lossless weight accumulation (w_1 <- w_1 + w_2)
     - Optional logical subsumption pruning (S_1 subset S_2)
  2. Stage 2 — Rule-Length-Based Filtering:
     - Short-clause filtering (l_{k,j} <= m_min)
     - Optional bloated-clause filtering (l_{k,j} > m_max)
  3. Stage 3 — Unified Clause Importance Scoring:
     - Margin Contribution Score (MCS & MCS-Hybrid) with harmful-clause detection
     - Class-conditional Firing Rate (rho_{k,j}) with optional upper-bound generalist cap
     - Vote Impact (|w_{k,j}| * rho_{k,j})
     - Discriminative Firing Margin (w_{k,j} * (rho_pos - rho_neg))
     - Weight Magnitude (|w_{k,j}|)
     - Random baseline
  4. Stage 4 — Class-Balanced Budget Selection & Model Footprint Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xxhash

from .model import PrunableTMClassifier


@dataclass
class SanitizationStats:
    """Detailed breakdown of clauses removed at each sanitization stage."""
    total_clauses: int
    empty_removed: int
    contradictory_removed: int
    encoding_invalid_removed: int
    duplicates_merged: int
    subsumed_removed: int
    short_removed: int
    harmful_removed: int
    low_score_removed: int
    active_clauses: int
    active_per_class: Dict[int, int]
    mean_literal_length: float
    total_included_literals: int
    model_size_kb: float
    bit_packed_active_kb: float


def rule_based_canonicalization(
    tm: PrunableTMClassifier,
    base_mask: Optional[np.ndarray] = None,
    merge_duplicates: bool = True,
    prune_subsumed: bool = False,
    thermometer_bits_per_feature: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Stage 1: Lossless Rule-Based Clause Sanitization (S_rule^+).
    Modifies duplicate weights in-place when `merge_duplicates=True` and returns
    updated binary retention mask `r^{(1)}` of shape (K, N) plus removal counts.
    """
    K = tm.number_of_classes
    N = tm.number_of_clauses
    A_all = tm.get_all_action_matrices()  # (K, N, 2*d), uint8
    W_all = tm.get_weights_matrix()  # (K, N), int32
    two_d = A_all.shape[2]
    d = two_d // 2

    if base_mask is None:
        mask = np.ones((K, N), dtype=np.int32)
    else:
        mask = base_mask.copy().astype(np.int32)

    counts = {
        "empty": 0,
        "contradictory": 0,
        "encoding_invalid": 0,
        "duplicate": 0,
        "subsumed": 0,
    }

    for k in range(K):
        A_k = A_all[k]  # (N, 2*d)
        pos_lit = A_k[:, :d]  # (N, d)
        neg_lit = A_k[:, d:]  # (N, d)
        lit_counts = A_k.sum(axis=1)  # (N,)

        # 1. Empty clauses: l_{k,j} == 0
        is_empty = (lit_counts == 0) & (mask[k] == 1)
        counts["empty"] += int(np.sum(is_empty))
        mask[k, is_empty] = 0

        # 2. Contradictory / tautological clauses: exists i with x_i=1 and ~x_i=1
        has_contradiction = np.any(pos_lit & neg_lit, axis=1) & (mask[k] == 1)
        counts["contradictory"] += int(np.sum(has_contradiction))
        mask[k, has_contradiction] = 0

        # 3. Encoding-invalid clauses (if thermometer encoding is specified)
        # In thermometer encoding with B bits per feature (b_1 >= b_2 >= ... >= b_B),
        # requiring b_t = 1 and b_{t'} = 0 for t' < t is unsatisfiable.
        if thermometer_bits_per_feature is not None and thermometer_bits_per_feature > 1:
            B = thermometer_bits_per_feature
            n_feat = d // B
            for j in np.where(mask[k] == 1)[0]:
                p_f = pos_lit[j, : n_feat * B].reshape(n_feat, B)
                n_f = neg_lit[j, : n_feat * B].reshape(n_feat, B)
                # Highest index where pos=1 vs lowest index where neg=1
                invalid = False
                for f_idx in range(n_feat):
                    pos_ones = np.where(p_f[f_idx] == 1)[0]
                    neg_ones = np.where(n_f[f_idx] == 1)[0]
                    if pos_ones.size > 0 and neg_ones.size > 0:
                        if np.max(pos_ones) >= np.min(neg_ones):
                            invalid = True
                            break
                if invalid:
                    mask[k, j] = 0
                    counts["encoding_invalid"] += 1

        # 4. Duplicate clause detection & lossless weight accumulation
        seen_hashes: Dict[Tuple[int, int], int] = {}
        active_indices = np.where(mask[k] == 1)[0]
        packed_A = np.packbits(A_k, axis=1, bitorder="little")

        for j in active_indices:
            polarity_half = 0 if j < (N // 2) else 1
            row_hash = xxhash.xxh64(packed_A[j].tobytes()).intdigest()
            sig_key = (polarity_half, row_hash)

            if sig_key in seen_hashes:
                rep_j = seen_hashes[sig_key]
                # Confirm exact byte equality to avoid any hash collision
                if np.array_equal(packed_A[rep_j], packed_A[j]):
                    if merge_duplicates:
                        W_all[k, rep_j] += W_all[k, j]
                        W_all[k, j] = 0
                        if W_all[k, rep_j] == 0:
                            mask[k, rep_j] = 0
                    mask[k, j] = 0
                    counts["duplicate"] += 1
                    continue
            seen_hashes[sig_key] = j

        # 5. Optional logical subsumption pruning within same polarity
        if prune_subsumed:
            for half_start, half_end in [(0, N // 2), (N // 2, N)]:
                half_active = [
                    j for j in range(half_start, half_end) if mask[k, j] == 1
                ]
                # Sort by literal length ascending
                half_active.sort(key=lambda idx: int(lit_counts[idx]))
                for idx_a, j_short in enumerate(half_active):
                    if mask[k, j_short] == 0 or lit_counts[j_short] < 2:
                        continue
                    pack_short = packed_A[j_short]
                    for j_long in half_active[idx_a + 1 :]:
                        if mask[k, j_long] == 0 or lit_counts[j_long] == lit_counts[j_short]:
                            continue
                        # j_short subset of j_long iff (pack_short & ~pack_long) == 0
                        if np.all((pack_short & ~packed_A[j_long]) == 0):
                            mask[k, j_long] = 0
                            counts["subsumed"] += 1

    if merge_duplicates:
        tm.set_weights_matrix(W_all)

    return mask, counts


def length_based_filter(
    tm: PrunableTMClassifier,
    base_mask: np.ndarray,
    min_literals: int = 0,
    max_literals: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Stage 2: Rule-Length-Based Sanitization (S_len).
    Removes clauses with 1 <= l_{k,j} <= min_literals (matching paper's
    r^{(2)} = r^{(1)} * 1[l_{k,j} > m_min]) and optionally l_{k,j} > max_literals.
    """
    mask = base_mask.copy().astype(np.int32)
    lit_counts = tm.get_literal_counts()  # (K, N)

    too_short = (lit_counts <= min_literals) & (mask == 1)
    removed = int(np.sum(too_short))
    mask[too_short] = 0

    if max_literals is not None:
        too_long = (lit_counts > max_literals) & (mask == 1)
        removed += int(np.sum(too_long))
        mask[too_long] = 0

    return mask, removed


def compute_clause_scores(
    tm: PrunableTMClassifier,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
    clause_outputs: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    alpha_disc: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Computes all clause importance scores on reference dataset (X_ref, Y_ref).
    Returns a dictionary mapping method name -> (K, N) float64 score array:
      - 'mcs': Margin Contribution Score (boundary-weighted margin impact)
      - 'mcs_hybrid': Margin Contribution + Discriminative Vote Impact (default TM-Ticket score)
      - 'discriminative': Signed weight * (rho_target - rho_nontarget)
      - 'vote_impact': |w_{k,j}| * rho_{k,j} (class-conditional)
      - 'firing_rate': Class-conditional firing rate rho_{k,j} (paper Stage 3)
      - 'weight_mag': Learned integer weight magnitude |w_{k,j}|
      - 'random': Random baseline score
    """
    if clause_outputs is None:
        clause_outputs = tm.evaluate_clause_outputs(X_ref)  # (K, M, N), uint8

    K, M, N = clause_outputs.shape
    if base_mask is None:
        base_mask = (
            tm.retention_mask
            if tm.retention_mask is not None
            else np.ones((K, N), dtype=np.int32)
        )

    W = tm.get_weights_matrix() * base_mask  # (K, N), int32

    # 1. Compute class vote sums V of shape (M, K)
    V = np.zeros((M, K), dtype=np.float64)
    for k in range(K):
        V[:, k] = clause_outputs[k].astype(np.float64) @ W[k].astype(np.float64)

    # 2. True class score v_y(x) and strongest rival q*(x) = argmax_{q != y} v_q(x)
    sample_idx = np.arange(M)
    v_true = V[sample_idx, Y_ref]  # (M,)

    V_rival = V.copy()
    V_rival[sample_idx, Y_ref] = -np.inf
    q_star = np.argmax(V_rival, axis=1)  # (M,)
    v_rival = V[sample_idx, q_star]  # (M,)

    # Decision margin mu(x) = v_y(x) - v_{q*}(x)
    margins = v_true - v_rival  # (M,)

    # Adaptive temperature tau if not specified: median positive margin (or T)
    if tau is None:
        pos_margins = margins[margins > 0]
        tau = float(np.median(pos_margins)) if pos_margins.size > 0 else float(max(tm.T, 1))
        tau = max(tau, 1.0)

    # Boundary focus weights omega(x) = 1.0 + exp(-max(mu(x), 0) / tau) in [1.0, 2.0]
    # Keeps a 1.0 baseline credit for well-classified samples while upweighting boundary samples 2x
    omega = 1.0 + np.exp(-np.maximum(margins, 0.0) / tau)  # (M,)

    mcs_scores = np.zeros((K, N), dtype=np.float64)
    disc_scores = np.zeros((K, N), dtype=np.float64)
    firing_cond = np.zeros((K, N), dtype=np.float64)
    vote_impact = np.zeros((K, N), dtype=np.float64)
    weight_mag = np.abs(W).astype(np.float64)

    half_N = N // 2

    for k in range(K):
        C_k = clause_outputs[k].astype(np.float64)  # (M, N)
        w_k = W[k].astype(np.float64)  # (N,)

        is_target = (Y_ref == k)
        is_not_target = ~is_target
        is_rival = (q_star == k)

        # Margin coefficient per sample: +1 when k == y, -1 when k == q*(x), 0 otherwise
        margin_sign = is_target.astype(np.float64) - is_rival.astype(np.float64)  # (M,)
        weighted_sign = (omega * margin_sign) / float(M)  # (M,)

        # Vectorized MCS_{k,j} = w_{k,j} * sum_x [ (omega(x) * margin_sign(x) / M) * C_{k,j}(x) ]
        mcs_scores[k] = w_k * (weighted_sign @ C_k)

        # Class-conditional firing rates:
        # Target firing rate on D_k and non-target firing rate on D_{!= k}
        n_pos = max(int(np.sum(is_target)), 1)
        n_neg = max(int(np.sum(is_not_target)), 1)
        rho_target = C_k[is_target].sum(axis=0) / float(n_pos)  # (N,)
        rho_nontarget = C_k[is_not_target].sum(axis=0) / float(n_neg)  # (N,)

        # Paper's Stage 3 class-conditional firing rate (algo.tex Eqs. 4-5):
        # Positive clauses (j < N/2) evaluated on D_k; negative clauses (j >= N/2) on D_{!= k}
        firing_cond[k, :half_N] = rho_target[:half_N]
        firing_cond[k, half_N:] = rho_nontarget[half_N:]

        # Vote impact = |w_{k,j}| * rho_{k,j}
        vote_impact[k] = np.abs(w_k) * firing_cond[k]

        # Discriminative margin contribution:
        # Positive weight clauses should fire on target and NOT on non-target;
        # Negative weight clauses should fire on non-target and NOT on target.
        disc_scores[k] = w_k * (rho_target - rho_nontarget)

    mcs_hybrid = mcs_scores + alpha_disc * disc_scores

    rng = np.random.RandomState(42)
    random_scores = rng.rand(K, N)

    # Mask out already inactive clauses with -inf
    inactive = (base_mask == 0)
    for arr in (
        mcs_scores,
        mcs_hybrid,
        disc_scores,
        firing_cond,
        vote_impact,
        weight_mag,
        random_scores,
    ):
        arr[inactive] = -np.inf

    return {
        "mcs_hybrid": mcs_hybrid,
        "mcs": mcs_scores,
        "discriminative": disc_scores,
        "vote_impact": vote_impact,
        "firing_rate": firing_cond,
        "weight_mag": weight_mag,
        "random": random_scores,
    }


def select_ticket_mask(
    scores: np.ndarray,
    base_mask: np.ndarray,
    target_retention_ratio: float,
    class_balanced: bool = True,
    prune_negative_scores: bool = False,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Constructs a binary ticket mask r of shape (K, N) retaining `target_retention_ratio`
    of the total clause bank N per class (when `class_balanced=True`) or globally.

    If `exclude_mask` is provided (shape (K, N), 1 = already used by previous ticket),
    those clauses are excluded so mutually disjoint tickets can be extracted.
    """
    K, N = scores.shape
    eligible = base_mask.copy().astype(np.int32)
    if exclude_mask is not None:
        eligible = eligible * (1 - exclude_mask.astype(np.int32))

    work_scores = scores.copy()
    work_scores[eligible == 0] = -np.inf

    if prune_negative_scores:
        # Clauses with strictly negative contribution hurt the margin; drop them first
        harmful = (work_scores < 0.0) & (eligible == 1)
        eligible[harmful] = 0
        work_scores[harmful] = -np.inf

    new_mask = np.zeros((K, N), dtype=np.int32)

    if class_balanced:
        target_per_class = max(2, int(round(target_retention_ratio * N)))
        half_N = N // 2
        target_pos = max(1, target_per_class // 2)
        target_neg = max(1, target_per_class - target_pos)

        for k in range(K):
            # Preserve polarity balance between positive and negative clause halves
            pos_elig = np.where(eligible[k, :half_N] == 1)[0]
            neg_elig = np.where(eligible[k, half_N:] == 1)[0] + half_N

            if pos_elig.size > 0:
                k_pos = min(target_pos, pos_elig.size)
                top_pos = pos_elig[np.argsort(work_scores[k, pos_elig])[::-1][:k_pos]]
                new_mask[k, top_pos] = 1

            if neg_elig.size > 0:
                k_neg = min(target_neg, neg_elig.size)
                top_neg = neg_elig[np.argsort(work_scores[k, neg_elig])[::-1][:k_neg]]
                new_mask[k, top_neg] = 1
    else:
        target_total = max(K, int(round(target_retention_ratio * K * N)))
        flat_elig = np.where(eligible.reshape(-1) == 1)[0]
        if flat_elig.size > 0:
            k_tot = min(target_total, flat_elig.size)
            flat_scores = work_scores.reshape(-1)[flat_elig]
            top_flat = flat_elig[np.argsort(flat_scores)[::-1][:k_tot]]
            new_mask.reshape(-1)[top_flat] = 1

    return new_mask


def compute_model_stats(
    tm: PrunableTMClassifier,
    mask: Optional[np.ndarray] = None,
    stage1_counts: Optional[Dict[str, int]] = None,
    short_removed: int = 0,
    harmful_removed: int = 0,
) -> SanitizationStats:
    """
    Computes exact clause counts, literal statistics, and memory footprint (in KB),
    matching the formula in Table III of the Clause Sanitization paper:
        Size (KB) = Active_Clauses * (2 * d) / 8 / 1024.
    """
    if mask is None:
        mask = (
            tm.retention_mask
            if tm.retention_mask is not None
            else np.ones((tm.number_of_classes, tm.number_of_clauses), dtype=np.int32)
        )

    K, N = mask.shape
    lit_counts = tm.get_literal_counts()  # (K, N)
    n_literals = tm.clause_banks[0].number_of_literals  # 2*d (1568 for MNIST)

    active_bool = (mask == 1)
    active_clauses = int(np.sum(active_bool))
    active_lits = lit_counts[active_bool]
    mean_lit_len = float(np.mean(active_lits)) if active_lits.size > 0 else 0.0
    total_inc_lits = int(np.sum(active_lits))

    # Paper's exact formula: 1 bit per literal per active clause
    model_size_kb = (active_clauses * n_literals) / 8.0 / 1024.0
    # Including 16-bit integer weight per active clause
    bit_packed_with_weights_kb = (active_clauses * (n_literals + 16)) / 8.0 / 1024.0

    stage1_counts = stage1_counts or {}
    empty_rem = stage1_counts.get("empty", 0)
    contra_rem = stage1_counts.get("contradictory", 0)
    enc_rem = stage1_counts.get("encoding_invalid", 0)
    dup_rem = stage1_counts.get("duplicate", 0)
    sub_rem = stage1_counts.get("subsumed", 0)

    total_clauses = K * N
    accounted = (
        empty_rem
        + contra_rem
        + enc_rem
        + dup_rem
        + sub_rem
        + short_removed
        + harmful_removed
        + active_clauses
    )
    low_score_rem = max(0, total_clauses - accounted)

    active_per_class = {int(k): int(np.sum(mask[k] == 1)) for k in range(K)}

    return SanitizationStats(
        total_clauses=total_clauses,
        empty_removed=empty_rem,
        contradictory_removed=contra_rem,
        encoding_invalid_removed=enc_rem,
        duplicates_merged=dup_rem,
        subsumed_removed=sub_rem,
        short_removed=short_removed,
        harmful_removed=harmful_removed,
        low_score_removed=low_score_rem,
        active_clauses=active_clauses,
        active_per_class=active_per_class,
        mean_literal_length=round(mean_lit_len, 2),
        total_included_literals=total_inc_lits,
        model_size_kb=round(model_size_kb, 2),
        bit_packed_active_kb=round(bit_packed_with_weights_kb, 2),
    )
