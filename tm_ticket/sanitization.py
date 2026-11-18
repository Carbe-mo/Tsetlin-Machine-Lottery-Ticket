"""
Parameter-Free, Data-Driven Clause Sanitization & Importance Scoring (`tm_ticket/sanitization.py`).

Eliminates all manual pruning hyperparameters (theta, L_min, tau, alpha, fixed budget ratios):
  1. Stage 1 — Certified Lossless Canonicalization:
     - Empty clause removal (l_{k,j} = 0)
     - Contradictory / tautological clause removal (x_i AND NOT x_i)
     - Dead clause removal (rho_{k,j} = 0 on reference data)
     - Duplicate clause detection and lossless weight accumulation (w_1 <- w_1 + w_2)
     - Margin-Certified Lossless Bound: removes clauses whose cumulative vote on every
       correctly classified sample is strictly < mu(x) / 2 (certified 0 prediction flips).
  2. Parameter-Free Margin Contribution Score (MCS*):
     - Data-driven boundary temperature: tau_data = median({mu(x) | mu(x) > 0})
     - Data-driven boundary weight: omega(x) = 1 + exp(-max(mu(x), 0) / tau_data) in [1, 2]
     - Data-driven blending via empirical validation error rate e_val = 1 - Acc_val:
       MCS*_{k,j} = (1 - e_val) * D_hat_{k,j} + e_val * M_hat_{k,j}
  3. Parameter-Free Data-Driven Capacity Hierarchy (per class k & polarity p in {+, -}):
     - N_pos:  All clauses with strictly positive empirical contribution (MCS* > 0)
     - N_eff:  Shannon Perplexity effective clause count exp(H(p))
     - N_PR:   Herfindahl Participation Ratio (Renyi-2 effective rank) (sum s)^2 / sum(s^2)
     - N_knee: First-order Kneedle / Above-Mean capacity #{j : s_{k,j} >= mean(s)}
     - N_core: Second-order Recursive Kneedle capacity #{j : s_{k,j} >= mean(s_knee)}
     - N_min:  Third-order Recursive Kneedle minimal backbone #{j : s_{k,j} >= mean(s_core)}
     - 1-SE Rule: sigma_val = sqrt(a_val * (1 - a_val) / |D_val|) selects the most compact
       ticket within 1 standard error of dense validation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xxhash

from .model import PrunableTMClassifier


@dataclass
class SanitizationStats:
    """Detailed breakdown of clauses removed and retained."""
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
        pos_lit = A_k[:, :d]
        neg_lit = A_k[:, d:]
        lit_counts = A_k.sum(axis=1)

        # 1. Empty clauses: l_{k,j} == 0
        is_empty = (lit_counts == 0) & (mask[k] == 1)
        counts["empty"] += int(np.sum(is_empty))
        mask[k, is_empty] = 0

        # 2. Contradictory clauses: exists i with x_i=1 and ~x_i=1
        has_contradiction = np.any(pos_lit & neg_lit, axis=1) & (mask[k] == 1)
        counts["contradictory"] += int(np.sum(has_contradiction))
        mask[k, has_contradiction] = 0

        # 3. Encoding-invalid clauses (if thermometer encoding is specified)
        if thermometer_bits_per_feature is not None and thermometer_bits_per_feature > 1:
            B = thermometer_bits_per_feature
            n_feat = d // B
            for j in np.where(mask[k] == 1)[0]:
                p_f = pos_lit[j, : n_feat * B].reshape(n_feat, B)
                n_f = neg_lit[j, : n_feat * B].reshape(n_feat, B)
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
                half_active.sort(key=lambda idx: int(lit_counts[idx]))
                for idx_a, j_short in enumerate(half_active):
                    if mask[k, j_short] == 0 or lit_counts[j_short] < 2:
                        continue
                    pack_short = packed_A[j_short]
                    for j_long in half_active[idx_a + 1 :]:
                        if mask[k, j_long] == 0 or lit_counts[j_long] == lit_counts[j_short]:
                            continue
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
    Stage 2: Rule-Length-Based Sanitization (used for reference paper comparison).
    In parameter-free mode, `min_literals=0` so length filtering is purely data-driven.
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
) -> Dict[str, Any]:
    """
    100% Parameter-Free Clause Importance Scoring on reference dataset (X_ref, Y_ref).
    Derives boundary temperature `tau_data` and blending weight `e_val` directly from
    the empirical margin distribution on (X_ref, Y_ref), with zero user hyperparameters.
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
    v_true = V[sample_idx, Y_ref]

    V_rival = V.copy()
    V_rival[sample_idx, Y_ref] = -np.inf
    q_star = np.argmax(V_rival, axis=1)
    v_rival = V[sample_idx, q_star]

    # Decision margin mu(x) = v_y(x) - v_{q*}(x)
    margins = v_true - v_rival  # (M,)

    # Data-driven boundary temperature tau_data = median of positive margins
    pos_margins = margins[margins > 0]
    tau_data = float(np.median(pos_margins)) if pos_margins.size > 0 else float(max(tm.T, 1))
    tau_data = max(tau_data, 1.0)

    # Data-driven validation error rate e_val = fraction of non-positive margins
    e_val = float(np.mean(margins <= 0.0))
    # Clamp e_val to [0.05, 0.95] so both boundary and core terms always contribute
    blend_alpha = float(np.clip(e_val, 0.05, 0.95))

    # Boundary focus weights omega(x) = 1.0 + exp(-max(mu(x), 0) / tau_data) in [1.0, 2.0]
    omega = 1.0 + np.exp(-np.maximum(margins, 0.0) / tau_data)

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

        margin_sign = is_target.astype(np.float64) - is_rival.astype(np.float64)
        weighted_sign = (omega * margin_sign) / float(M)

        mcs_scores[k] = w_k * (weighted_sign @ C_k)

        n_pos = max(int(np.sum(is_target)), 1)
        n_neg = max(int(np.sum(is_not_target)), 1)
        rho_target = C_k[is_target].sum(axis=0) / float(n_pos)
        rho_nontarget = C_k[is_not_target].sum(axis=0) / float(n_neg)

        firing_cond[k, :half_N] = rho_target[:half_N]
        firing_cond[k, half_N:] = rho_nontarget[half_N:]

        vote_impact[k] = np.abs(w_k) * firing_cond[k]
        disc_scores[k] = w_k * (rho_target - rho_nontarget)

    # Parameter-free standardization & data-driven blending:
    active_bool = (base_mask == 1)
    mcs_scale = float(np.mean(np.abs(mcs_scores[active_bool]))) if np.any(active_bool) else 1.0
    disc_scale = float(np.mean(np.abs(disc_scores[active_bool]))) if np.any(active_bool) else 1.0
    mcs_norm = mcs_scores / max(mcs_scale, 1e-12)
    disc_norm = disc_scores / max(disc_scale, 1e-12)

    mcs_hybrid = (1.0 - blend_alpha) * disc_norm + blend_alpha * mcs_norm

    rng = np.random.RandomState(42)
    random_scores = rng.rand(K, N)

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
        "margins": margins,
        "tau_data": round(tau_data, 2),
        "e_val": round(e_val, 4),
    }


def compute_certified_lossless_mask(
    tm: PrunableTMClassifier,
    base_mask: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    clause_outputs: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Margin-Certified Lossless Pruning Bound:
    Greedily removes clauses in ascending order of importance score |s_{k,j}| as long as
    the cumulative worst-case margin reduction on EVERY positively-classified sample x
    satisfies:
        sum_{removed (k,j)} |w_{k,j}| * C_{k,j}(x) < mu(x) / 2.0
    Also removes dead clauses (clauses that never fire on any sample, C_{k,j}(x) == 0).
    Guarantees 0 prediction flips on the validation set.
    """
    K, M, N = clause_outputs.shape
    mask = base_mask.copy().astype(np.int32)
    W = np.abs(tm.get_weights_matrix()) * mask

    pos_samples = np.where(margins > 0.0)[0]
    budget_per_sample = np.zeros(M, dtype=np.float64)
    budget_per_sample[pos_samples] = margins[pos_samples] / 2.0 - 1e-6
    used_budget = np.zeros(M, dtype=np.float64)

    certified_removed = 0

    for k in range(K):
        C_k = clause_outputs[k]  # (M, N), uint8
        fire_totals = C_k.sum(axis=0)  # (N,)

        # 1. Dead clauses (fire_totals == 0): 0 impact on all samples
        dead = (fire_totals == 0) & (mask[k] == 1)
        n_dead = int(np.sum(dead))
        certified_removed += n_dead
        mask[k, dead] = 0

        # 2. Check low-importance clauses against remaining margin budget mu(x)/2
        active_j = np.where(mask[k] == 1)[0]
        if active_j.size == 0 or pos_samples.size == 0:
            continue

        order = active_j[np.argsort(np.abs(scores[k, active_j]))]
        for j in order:
            w_abs = float(W[k, j])
            if w_abs == 0.0:
                mask[k, j] = 0
                certified_removed += 1
                continue
            firing_pos = pos_samples[C_k[pos_samples, j] == 1]
            if firing_pos.size == 0:
                continue
            if np.all(used_budget[firing_pos] + w_abs < budget_per_sample[firing_pos]):
                used_budget[firing_pos] += w_abs
                mask[k, j] = 0
                certified_removed += 1

    return mask, certified_removed


def compute_data_driven_capacities(
    scores: np.ndarray,
    base_mask: np.ndarray,
    val_accuracy_pct: float,
    n_val_samples: int,
) -> Dict[str, Any]:
    """
    Derives the parameter-free hierarchy of clause capacities directly from the
    empirical positive Margin Contribution Scores s_{k,j} = max(MCS*_{k,j}, 0)
    for every class k and polarity half p in {+, -}:

      1. `N_pos`  (Positive-Contribution Filter): #{j : s_{k,j} > 0}
      2. `N_eff`  (Shannon Perplexity Capacity):  ceil(exp(-sum p_{k,j} ln p_{k,j}))
      3. `N_PR`   (Herfindahl Participation Ratio): ceil((sum s_{k,j})^2 / sum s_{k,j}^2)
      4. `N_knee` (1st-Order Kneedle / Above-Mean): #{j : s_{k,j} >= mean(s)}
      5. `N_core` (2nd-Order Recursive Kneedle):    #{j in knee : s_{k,j} >= mean(s_knee)}
      6. `N_min`  (3rd-Order Recursive Kneedle):    #{j in core : s_{k,j} >= mean(s_core)}

    Also computes the parameter-free 1-SE validation tolerance:
      sigma_val = sqrt(p * (1 - p) / n_val) * 100%
    and the automatic number of disjoint winning tickets:
      M_auto = floor(min_{k,p} N_pos(k,p) / N_core(k,p)).
    """
    K, N = scores.shape
    half_N = N // 2

    masks = {
        "pos": np.zeros((K, N), dtype=np.int32),
        "shannon_eff": np.zeros((K, N), dtype=np.int32),
        "participation_pr": np.zeros((K, N), dtype=np.int32),
        "kneedle_1": np.zeros((K, N), dtype=np.int32),
        "kneedle_2_core": np.zeros((K, N), dtype=np.int32),
        "kneedle_3_min": np.zeros((K, N), dtype=np.int32),
    }

    per_class_summary: Dict[int, Dict[str, int]] = {}
    disjoint_ratios: List[float] = []

    for k in range(K):
        cls_counts = {name: 0 for name in masks}
        for half_start, half_end in [(0, half_N), (half_N, N)]:
            elig = np.where((base_mask[k, half_start:half_end] == 1))[0] + half_start
            if elig.size == 0:
                continue

            s_raw = scores[k, elig]
            pos_mask = s_raw > 0.0
            pos_idx = elig[pos_mask]
            if pos_idx.size == 0:
                # Fallback: keep top 2 clauses if none are strictly positive
                top2 = elig[np.argsort(s_raw)[::-1][: min(2, elig.size)]]
                for m_arr in masks.values():
                    m_arr[k, top2] = 1
                continue

            s_pos = scores[k, pos_idx]
            order = np.argsort(s_pos)[::-1]
            sorted_idx = pos_idx[order]
            sorted_s = s_pos[order]

            n_pos = int(sorted_idx.size)
            masks["pos"][k, sorted_idx] = 1
            cls_counts["pos"] += n_pos

            # Normalized distribution p_j = s_j / sum(s)
            s_sum = float(np.sum(sorted_s))
            p_dist = sorted_s / max(s_sum, 1e-12)

            # 1. Shannon Perplexity N_eff = exp(-sum p ln p)
            entropy = -float(np.sum(p_dist * np.log(np.maximum(p_dist, 1e-15))))
            n_eff = max(2, min(n_pos, int(np.ceil(np.exp(entropy)))))
            masks["shannon_eff"][k, sorted_idx[:n_eff]] = 1
            cls_counts["shannon_eff"] += n_eff

            # 2. Herfindahl Participation Ratio N_PR = (sum s)^2 / sum(s^2)
            s_sq_sum = float(np.sum(sorted_s ** 2))
            n_pr = max(2, min(n_pos, int(np.ceil((s_sum ** 2) / max(s_sq_sum, 1e-12)))))
            masks["participation_pr"][k, sorted_idx[:n_pr]] = 1
            cls_counts["participation_pr"] += n_pr

            # 3. 1st-Order Kneedle (Above-Mean marginal contribution: s_j >= mean(s_pos))
            mean_1 = float(np.mean(sorted_s))
            n_knee1 = max(2, int(np.sum(sorted_s >= mean_1)))
            masks["kneedle_1"][k, sorted_idx[:n_knee1]] = 1
            cls_counts["kneedle_1"] += n_knee1

            # 4. 2nd-Order Recursive Kneedle (Core ticket: s_j >= mean(s_knee1))
            mean_2 = float(np.mean(sorted_s[:n_knee1]))
            n_knee2 = max(2, int(np.sum(sorted_s[:n_knee1] >= mean_2)))
            masks["kneedle_2_core"][k, sorted_idx[:n_knee2]] = 1
            cls_counts["kneedle_2_core"] += n_knee2

            # 5. 3rd-Order Recursive Kneedle (Minimal backbone: s_j >= mean(s_knee2))
            mean_3 = float(np.mean(sorted_s[:n_knee2]))
            n_knee3 = max(2, int(np.sum(sorted_s[:n_knee2] >= mean_3)))
            masks["kneedle_3_min"][k, sorted_idx[:n_knee3]] = 1
            cls_counts["kneedle_3_min"] += n_knee3

            disjoint_ratios.append(n_pos / float(max(n_knee2, 1)))

        per_class_summary[int(k)] = cls_counts

    # Parameter-free validation standard error (1-SE rule)
    p_acc = np.clip(val_accuracy_pct / 100.0, 1e-4, 1.0 - 1e-4)
    sigma_val_pct = float(np.sqrt(p_acc * (1.0 - p_acc) / float(max(n_val_samples, 1))) * 100.0)

    # Data-driven number of mutually disjoint core tickets M_auto
    m_auto = int(np.floor(np.median(disjoint_ratios))) if disjoint_ratios else 3
    m_auto = max(3, min(m_auto, 6))

    # Build ordered schedule of unique data-driven levels (from largest to smallest)
    level_defs = [
        ("L1_Positive_MCS (MCS* > 0)", "pos"),
        ("L2_Shannon_Perplexity (N_eff = exp(H))", "shannon_eff"),
        ("L3_Participation_Ratio (N_PR = (sum s)^2 / sum s^2)", "participation_pr"),
        ("L4_Kneedle_AboveMean (s >= mean(s))", "kneedle_1"),
        ("L5_Recursive_Core (2nd-Order Kneedle)", "kneedle_2_core"),
        ("L6_Minimal_Backbone (3rd-Order Kneedle)", "kneedle_3_min"),
    ]
    ordered_levels = sorted(
        level_defs,
        key=lambda item: int(np.sum(masks[item[1]])),
        reverse=True,
    )

    return {
        "masks": masks,
        "ordered_levels": ordered_levels,
        "per_class_summary": per_class_summary,
        "sigma_val_pct": round(sigma_val_pct, 4),
        "ci95_val_pct": round(1.96 * sigma_val_pct, 4),
        "m_auto_disjoint_tickets": m_auto,
        "core_ticket_total_clauses": int(np.sum(masks["kneedle_2_core"])),
    }


def partition_disjoint_tickets_Auto(
    scores: np.ndarray,
    base_mask: np.ndarray,
    target_mask_template: np.ndarray,
    num_tickets: int,
    mode: str = "peeling",
) -> List[np.ndarray]:
    """
    Extracts `num_tickets` strictly disjoint (0% overlap) tickets where each ticket's
    per-class, per-polarity clause count is set by `target_mask_template` (e.g., the
    parameter-free `kneedle_2_core` or `kneedle_1` capacity of each class!).

    Supports two parameter-free partitioning modes:
      - 'peeling':     Sequential greedy peeling (Ticket 1 gets top B_{k,p}, Ticket 2 gets next B_{k,p}, ...)
      - 'snake_draft': Round-robin snake draft (1..M, M..1) across top M * B_{k,p} clauses so all M
                       disjoint tickets have equal average margin strength.
    """
    K, N = scores.shape
    half_N = N // 2
    ticket_masks = [np.zeros((K, N), dtype=np.int32) for _ in range(num_tickets)]

    for k in range(K):
        for half_start, half_end in [(0, half_N), (half_N, N)]:
            b_kp = int(np.sum(target_mask_template[k, half_start:half_end] == 1))
            b_kp = max(2, b_kp)

            elig = np.where(base_mask[k, half_start:half_end] == 1)[0] + half_start
            if elig.size == 0:
                continue

            order = np.argsort(scores[k, elig])[::-1]
            sorted_idx = elig[order]

            # Ensure we don't exceed available clean clauses across all M tickets
            max_per_ticket = min(b_kp, int(sorted_idx.size // num_tickets))
            if max_per_ticket < 1:
                continue

            if mode == "peeling":
                for t_i in range(num_tickets):
                    start = t_i * max_per_ticket
                    end = (t_i + 1) * max_per_ticket
                    ticket_masks[t_i][k, sorted_idx[start:end]] = 1
            elif mode == "snake_draft":
                pool = sorted_idx[: num_tickets * max_per_ticket]
                for round_i in range(max_per_ticket):
                    chunk = pool[round_i * num_tickets : (round_i + 1) * num_tickets]
                    if round_i % 2 == 1:
                        chunk = chunk[::-1]
                    for t_i in range(num_tickets):
                        ticket_masks[t_i][k, chunk[t_i]] = 1
            else:
                raise ValueError(f"Unknown disjoint partitioning mode: {mode}")

    return ticket_masks


def select_ticket_mask(
    scores: np.ndarray,
    base_mask: np.ndarray,
    target_retention_ratio: float,
    class_balanced: bool = True,
    prune_negative_scores: bool = False,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Constructs a binary ticket mask r of shape (K, N) matching a target clause count
    or ratio (used when comparing different scoring criteria at matched capacity).
    """
    K, N = scores.shape
    eligible = base_mask.copy().astype(np.int32)
    if exclude_mask is not None:
        eligible = eligible * (1 - exclude_mask.astype(np.int32))

    work_scores = scores.copy()
    work_scores[eligible == 0] = -np.inf

    if prune_negative_scores:
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
    if mask is None:
        mask = (
            tm.retention_mask
            if tm.retention_mask is not None
            else np.ones((tm.number_of_classes, tm.number_of_clauses), dtype=np.int32)
        )

    K, N = mask.shape
    lit_counts = tm.get_literal_counts()
    n_literals = tm.clause_banks[0].number_of_literals

    active_bool = (mask == 1)
    active_clauses = int(np.sum(active_bool))
    active_lits = lit_counts[active_bool]
    mean_lit_len = float(np.mean(active_lits)) if active_lits.size > 0 else 0.0
    total_inc_lits = int(np.sum(active_lits))

    model_size_kb = (active_clauses * n_literals) / 8.0 / 1024.0
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
