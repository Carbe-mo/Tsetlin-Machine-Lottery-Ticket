"""
PrunableTMClassifier: Subclasses tmu.models.classification.vanilla_classifier.TMClassifier
to add:
  1. Fast vectorized extraction of TA action bit-matrices, literal counts, and weights.
  2. Exact TA-state & weight checkpointing and selective rewinding (CPU and CUDA).
  3. Clause retention masking (`retention_mask`) that freezes pruned clauses during
     both inference and retraining (Type I/II/III feedback and weight updates).
  4. Class-score renormalization and voting threshold (T) rescaling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tmu.clause_bank.clause_bank import ClauseBank
from tmu.models.classification.vanilla_classifier import TMClassifier


def _patch_tmu_numpy2_compat() -> None:
    """
    Patches tmu's ClauseBank.initialize_clauses and ImplClauseBankCUDA._cffi_init
    where `np.uint32(~0)` raises OverflowError under NumPy 2.0+ (NEP 50).
    Uses `np.uint32(0xFFFFFFFF)` which works identically on NumPy 1.x and 2.x.
    """
    all_ones = np.uint32(0xFFFFFFFF)

    def _initialize_clauses_compat(self):
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="c",
        )
        self.clause_bank[:, :, 0 : self.number_of_state_bits_ta - 1] = all_ones
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = np.uint32(0)
        self.clause_bank = np.ascontiguousarray(
            self.clause_bank.reshape(
                (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta)
            )
        )
        self.actions = np.ascontiguousarray(np.zeros(self.number_of_ta_chunks, dtype=np.uint32))
        self.clause_bank_ind = np.empty(
            (self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ind),
            dtype=np.uint32,
        )
        self.clause_bank_ind[:, :, :] = all_ones
        self.clause_bank_ind = np.ascontiguousarray(
            self.clause_bank_ind.reshape(
                (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ind)
            )
        )
        self.incremental_clause_evaluation_initialized = False

    ClauseBank.initialize_clauses = _initialize_clauses_compat

    try:
        from tmu.clause_bank.clause_bank_cuda import ImplClauseBankCUDA, cuda_installed
        if cuda_installed:
            import pycuda.driver as cuda

            def _cuda_cffi_init_compat(self):
                self.clause_bank = np.empty(
                    shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
                    dtype=np.uint32,
                    order="c",
                )
                self.clause_bank[:, :, 0 : self.number_of_state_bits_ta - 1] = all_ones
                self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = np.uint32(0)
                self.clause_bank = self.clause_bank.reshape(
                    (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta),
                    order="c",
                )
                self.clause_bank_gpu = self._profiler.profile(cuda.mem_alloc, self.clause_bank.nbytes)
                self._profiler.profile(cuda.memcpy_htod, self.clause_bank_gpu, self.clause_bank)
                self.clause_bank_synchronized = True

            ImplClauseBankCUDA._cffi_init = _cuda_cffi_init_compat
    except Exception:
        pass


_patch_tmu_numpy2_compat()


@dataclass
class TMCheckpoint:
    """Stores a complete snapshot of TA states, weights, and hyperparameters."""
    epoch: int
    clause_banks_state: Dict[int, np.ndarray]
    clause_banks_ind_state: Dict[int, np.ndarray]
    weights: Dict[int, np.ndarray]
    T: int
    retention_mask: np.ndarray


class PrunableTMClassifier(TMClassifier):
    """
    A Tsetlin Machine classifier supporting in-place clause sanitization,
    selective clause freezing during retraining, and TA-state rewinding
    for Lottery Ticket Hypothesis experiments.
    """

    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        platform: str = "CPU",
        weighted_clauses: bool = True,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        # Gracefully fall back to CPU if CUDA is requested but PyCUDA is not available
        if platform in ("GPU", "CUDA"):
            try:
                from tmu.clause_bank.clause_bank_cuda import cuda_installed
                if not cuda_installed:
                    platform = "CPU"
            except Exception:
                platform = "CPU"

        super().__init__(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            platform=platform,
            weighted_clauses=weighted_clauses,
            seed=seed,
            **kwargs,
        )
        self.initial_T: int = int(T)
        self.retention_mask: Optional[np.ndarray] = None  # shape (K, N), dtype int32
        self.class_vote_scales: Optional[np.ndarray] = None  # shape (K,), dtype float64

    def init(self, X: np.ndarray, Y: np.ndarray):
        was_initialized = self.initialized
        super().init(X, Y)
        if not was_initialized:
            self.retention_mask = np.ones(
                (self.number_of_classes, self.number_of_clauses), dtype=np.int32
            )
            self.class_vote_scales = np.ones(self.number_of_classes, dtype=np.float64)

    # ------------------------------------------------------------------
    # Override active-clause mechanism so pruned clauses are truly frozen
    # ------------------------------------------------------------------
    def mechanism_clause_active(self) -> np.ndarray:
        base_active = super().mechanism_clause_active()
        if self.retention_mask is not None:
            return (base_active * self.retention_mask).astype(np.int32)
        return base_active

    # ------------------------------------------------------------------
    # Synchronization helpers for CPU / CUDA clause banks
    # ------------------------------------------------------------------
    def _get_raw_clause_bank_array(self, k: int) -> np.ndarray:
        cb = self.clause_banks[k]
        if hasattr(cb, "clause_bank"):
            return cb.clause_bank
        if hasattr(cb, "host") and hasattr(cb.host, "clause_bank"):
            return cb.host.clause_bank
        raise AttributeError(f"Cannot locate clause_bank array on {type(cb).__name__}")

    def _sync_from_device(self, k: int) -> None:
        cb = self.clause_banks[k]
        if hasattr(cb, "synchronize_clause_bank"):
            cb.synchronize_clause_bank()

    def _sync_to_device(self, k: int) -> None:
        cb = self.clause_banks[k]
        if hasattr(cb, "device") and cb.device is not None and hasattr(cb, "host"):
            cb.device.upload_clause_bank(cb.host.clause_bank)
        elif hasattr(cb, "clause_bank_gpu") and cb.clause_bank_gpu is not None:
            import pycuda.driver as cuda
            cuda.memcpy_htod(cb.clause_bank_gpu, cb.clause_bank)
            cb.clause_bank_synchronized = True
        if hasattr(cb, "incremental_clause_evaluation_initialized"):
            cb.incremental_clause_evaluation_initialized = False

    # ------------------------------------------------------------------
    # Fast vectorized inspection of clauses, literals, and weights
    # ------------------------------------------------------------------
    def get_action_matrix(self, k: int) -> np.ndarray:
        """
        Returns the binary action matrix A_k of shape (N, 2*d) with dtype uint8
        for class k, extracted directly from the MSB bit-plane of the TA states.
        """
        self._sync_from_device(k)
        cb = self.clause_banks[k]
        raw_bank = self._get_raw_clause_bank_array(k)
        N = self.number_of_clauses
        chunks = cb.number_of_ta_chunks
        bits = cb.number_of_state_bits_ta
        n_lit = cb.number_of_literals

        # MSB plane (index bits - 1) holds the action bit (1 = Include, 0 = Exclude)
        bank_3d = raw_bank.reshape((N, chunks, bits))
        msb_chunks = np.ascontiguousarray(bank_3d[:, :, bits - 1], dtype=np.uint32)

        # Unpack 32-bit unsigned ints (little-endian bit order within each uint32)
        unpacked = np.unpackbits(
            msb_chunks.view(np.uint8), axis=1, bitorder="little"
        )
        return unpacked[:, :n_lit]

    def get_all_action_matrices(self) -> np.ndarray:
        """Returns binary include-action tensor of shape (K, N, 2*d), dtype uint8."""
        return np.stack(
            [self.get_action_matrix(k) for k in range(self.number_of_classes)],
            axis=0,
        )

    def get_literal_counts(self) -> np.ndarray:
        """Returns literal count l_{k,j} of shape (K, N), dtype int32."""
        A = self.get_all_action_matrices()
        return A.sum(axis=2, dtype=np.int32)

    def get_weights_matrix(self) -> np.ndarray:
        """Returns signed weight matrix W of shape (K, N), dtype int32."""
        return np.stack(
            [
                self.weight_banks[k].get_weights().copy()
                for k in range(self.number_of_classes)
            ],
            axis=0,
        )

    def set_weights_matrix(self, W: np.ndarray) -> None:
        """Sets signed weights for all classes from a (K, N) int32 array."""
        for k in range(self.number_of_classes):
            wb_arr = self.weight_banks[k].get_weights()
            wb_arr[:] = W[k].astype(np.int32)

    def apply_retention_mask(
        self,
        mask: np.ndarray,
        zero_weights: bool = True,
        renormalize_votes: bool = False,
    ) -> None:
        """
        Applies a (K, N) binary retention mask to the classifier.
        If zero_weights=True, sets w'_{k,j} = r_{k,j} * w_{k,j} in-place.
        """
        assert mask.shape == (self.number_of_classes, self.number_of_clauses)
        self.retention_mask = (mask > 0).astype(np.int32)

        if zero_weights:
            for k in range(self.number_of_classes):
                wb_arr = self.weight_banks[k].get_weights()
                wb_arr[:] = wb_arr * self.retention_mask[k]

        if renormalize_votes:
            self.update_vote_renormalization()
        else:
            self.class_vote_scales = np.ones(self.number_of_classes, dtype=np.float64)

    def update_vote_renormalization(self) -> None:
        """
        Computes per-class vote scaling factors so classes with fewer surviving
        weights are not disadvantaged in argmax_k v_k(x).
        """
        W = self.get_weights_matrix() * self.retention_mask
        abs_sums = np.abs(W).sum(axis=1).astype(np.float64)
        mean_sum = np.mean(abs_sums[abs_sums > 0]) if np.any(abs_sums > 0) else 1.0
        scales = np.ones(self.number_of_classes, dtype=np.float64)
        for k in range(self.number_of_classes):
            if abs_sums[k] > 0:
                scales[k] = mean_sum / abs_sums[k]
        self.class_vote_scales = scales

    def rescale_threshold(self, mode: str = "proportional") -> int:
        """
        Updates the voting threshold T to match the active clause bank size.
          - 'proportional': T' = max(1, round(T_0 * min_k N_active_k / N))
          - 'sanitization_paper': T' = max(1, floor(0.75 * min_k N_active_k))
          - 'fixed': T' = T_0
        """
        if self.retention_mask is None or mode == "fixed":
            self.T = int(self.initial_T)
            return self.T

        active_per_class = self.retention_mask.sum(axis=1)
        min_active = int(np.min(active_per_class))

        if mode == "proportional":
            ratio = min_active / float(self.number_of_clauses)
            self.T = max(1, int(round(self.initial_T * ratio)))
        elif mode == "sanitization_paper":
            self.T = max(1, int(np.floor(0.75 * min_active)))
        else:
            raise ValueError(f"Unknown threshold rescaling mode: {mode}")
        return self.T

    # ------------------------------------------------------------------
    # Checkpointing and Rewinding for Lottery Ticket Experiments
    # ------------------------------------------------------------------
    def get_checkpoint(self, epoch: int = 0) -> TMCheckpoint:
        """Captures a deep snapshot of all TA states, weights, and mask."""
        cb_states = {}
        cb_ind_states = {}
        weights = {}
        for k in range(self.number_of_classes):
            self._sync_from_device(k)
            cb = self.clause_banks[k]
            raw_bank = self._get_raw_clause_bank_array(k)
            cb_states[k] = raw_bank.copy()
            cb_ind_states[k] = (
                cb.clause_bank_ind.copy()
                if hasattr(cb, "clause_bank_ind")
                else np.empty(0, dtype=np.uint32)
            )
            weights[k] = self.weight_banks[k].get_weights().copy()

        return TMCheckpoint(
            epoch=epoch,
            clause_banks_state=cb_states,
            clause_banks_ind_state=cb_ind_states,
            weights=weights,
            T=int(self.T),
            retention_mask=self.retention_mask.copy(),
        )

    def rewind_to_checkpoint(
        self,
        checkpoint: TMCheckpoint,
        retention_mask: Optional[np.ndarray] = None,
        rewind_weights: bool = True,
        rescale_T_mode: str = "proportional",
    ) -> None:
        """
        Rewinds the TA states (and optionally weights) of the TM to `checkpoint`,
        then applies `retention_mask` so pruned clauses are zeroed and frozen.
        """
        for k in range(self.number_of_classes):
            cb = self.clause_banks[k]
            raw_bank = self._get_raw_clause_bank_array(k)
            raw_bank[:] = checkpoint.clause_banks_state[k]
            if (
                hasattr(cb, "clause_bank_ind")
                and checkpoint.clause_banks_ind_state[k].size > 0
            ):
                cb.clause_bank_ind[:] = checkpoint.clause_banks_ind_state[k]
            self._sync_to_device(k)

            if rewind_weights:
                self.weight_banks[k].get_weights()[:] = checkpoint.weights[k]

        if retention_mask is not None:
            self.apply_retention_mask(retention_mask, zero_weights=True)
        else:
            self.apply_retention_mask(checkpoint.retention_mask, zero_weights=True)

        self.rescale_threshold(mode=rescale_T_mode)

    def reinitialize_random_states(
        self,
        retention_mask: np.ndarray,
        seed: int = 12345,
        init_include_prob: float = 0.01,
        rescale_T_mode: str = "proportional",
    ) -> None:
        """
        Control condition for LTH: keeps the same clause budget / mask structure
        (`retention_mask`), but re-initializes the surviving clauses' TA states to
        fresh random boundary states (state N_states for Exclude or N_states+1 for
        Include with small probability `init_include_prob`) and resets weights to ±1.
        """
        rng = np.random.RandomState(seed)
        N = self.number_of_clauses

        for k in range(self.number_of_classes):
            cb = self.clause_banks[k]
            raw_bank = self._get_raw_clause_bank_array(k)
            chunks = cb.number_of_ta_chunks
            bits = cb.number_of_state_bits_ta
            n_lit = cb.number_of_literals

            # Standard TM boundary state: Exclude at state 2^(bits-1) -> bits 0..bits-2 = 1, MSB = 0
            bank_3d = np.empty((N, chunks, bits), dtype=np.uint32, order="c")
            bank_3d[:, :, 0 : bits - 1] = np.uint32(0xFFFFFFFF)
            bank_3d[:, :, bits - 1] = np.uint32(0)

            # Randomly Perturb initial boundary states (Exclude state 127 vs 128 or sparse Include)
            if init_include_prob > 0:
                inc_bits = (rng.rand(N, chunks * 32) < init_include_prob).astype(np.uint8)
                inc_bits[:, n_lit:] = 0
                packed_inc = np.packbits(inc_bits, axis=1, bitorder="little").view(np.uint32)
                # For included literals: state 2^(bits-1)+1 -> bits 0..bits-2 = 0, MSB = 1
                bank_3d[:, :, 0 : bits - 1] = np.bitwise_not(packed_inc)[:, :, None]
                bank_3d[:, :, bits - 1] = packed_inc
            else:
                # Randomize lowest bit of Exclude state near boundary so TAs have fresh trajectory
                jitter = rng.randint(0, 2, size=(N, chunks), dtype=np.uint32)
                bank_3d[:, :, 0] ^= jitter

            raw_bank[:] = bank_3d.reshape(-1)
            self._sync_to_device(k)

            # Reset weights to +1 (first half) and -1 (second half), masked by retention_mask
            default_w = np.concatenate(
                [
                    np.ones(N // 2, dtype=np.int32),
                    -np.ones(N // 2, dtype=np.int32),
                ]
            )
            self.weight_banks[k].get_weights()[:] = default_w * retention_mask[k]

        self.apply_retention_mask(retention_mask, zero_weights=True)
        self.rescale_threshold(mode=rescale_T_mode)

    # ------------------------------------------------------------------
    # Fast Clause Output Evaluation & Prediction
    # ------------------------------------------------------------------
    def evaluate_clause_outputs(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates all clauses on dataset X (shape [M, d], uint32).
        Returns C of shape (K, M, N) with dtype uint8, where C[k, m, j] in {0, 1}.

        Uses fast vectorized bit-packed matrix operations that match tmu's exact
        predict semantics (empty clauses output 0 during inference).
        """
        M, d = X.shape
        K = self.number_of_classes
        N = self.number_of_clauses

        # Build literal matrix L of shape (M, 2*d): [x_1..x_d, ~x_1..~x_d]
        X_bin = (X > 0).astype(np.uint8)
        L_bin = np.concatenate([X_bin, 1 - X_bin], axis=1)  # (M, 2*d)
        # Bit-pack negated literals (~L_bin) so a clause fails iff (A & ~L) != 0
        not_L_packed = np.packbits(1 - L_bin, axis=1, bitorder="little")  # (M, n_bytes)

        C_out = np.zeros((K, M, N), dtype=np.uint8)

        for k in range(K):
            A_k = self.get_action_matrix(k)  # (N, 2*d), uint8
            lit_counts = A_k.sum(axis=1)  # (N,)
            non_empty = np.where(lit_counts > 0)[0]
            if non_empty.size == 0:
                continue

            A_ne = A_k[non_empty]  # (N_ne, 2*d)
            A_packed = np.packbits(A_ne, axis=1, bitorder="little")  # (N_ne, n_bytes)

            # A clause j fires on sample m iff for all byte chunks b:
            # (not_L_packed[m, b] & A_packed[j, b]) == 0
            # Using float32 matrix multiplication on unpacked binary is even faster on CPU/BLAS:
            # Number of violated literals = (1 - L_bin) @ A_ne.T == 0
            violated = (1 - L_bin).astype(np.float32) @ A_ne.astype(np.float32).T
            C_out[k, :, non_empty] = (violated == 0.0).astype(np.uint8).T

        return C_out

    def compute_class_scores(
        self,
        X: np.ndarray,
        clause_outputs: Optional[np.ndarray] = None,
        use_vote_scales: bool = True,
    ) -> np.ndarray:
        """
        Computes class vote sums V of shape (M, K), dtype float64.
        """
        if clause_outputs is None:
            clause_outputs = self.evaluate_clause_outputs(X)  # (K, M, N)

        K, M, N = clause_outputs.shape
        W = self.get_weights_matrix()
        if self.retention_mask is not None:
            W = W * self.retention_mask

        scores = np.zeros((M, K), dtype=np.float64)
        for k in range(K):
            scores[:, k] = clause_outputs[k].astype(np.float64) @ W[k].astype(np.float64)
            if use_vote_scales and self.class_vote_scales is not None:
                scores[:, k] *= self.class_vote_scales[k]

        return scores

    def evaluate_accuracy(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clause_outputs: Optional[np.ndarray] = None,
        use_vote_scales: bool = False,
    ) -> float:
        """Returns classification accuracy in percentage (0.0 to 100.0)."""
        scores = self.compute_class_scores(
            X, clause_outputs=clause_outputs, use_vote_scales=use_vote_scales
        )
        preds = np.argmax(scores, axis=1)
        return float(np.mean(preds == Y) * 100.0)
