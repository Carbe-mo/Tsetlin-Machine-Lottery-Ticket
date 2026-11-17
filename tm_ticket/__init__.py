"""
TM-Ticket: Margin-Aware Clause Sanitization & Lottery Ticket Discovery for Tsetlin Machines.
Built on top of `tmu` (Tsetlin Machine Unified).
"""

from .model import PrunableTMClassifier, TMCheckpoint
from .sanitization import (
    rule_based_canonicalization,
    length_based_filter,
    compute_clause_scores,
    select_ticket_mask,
    compute_model_stats,
)
from .lottery import (
    run_iterative_lottery_ticket,
    discover_multiple_disjoint_tickets,
    compare_optimization_methods,
)

__all__ = [
    "PrunableTMClassifier",
    "TMCheckpoint",
    "rule_based_canonicalization",
    "length_based_filter",
    "compute_clause_scores",
    "select_ticket_mask",
    "compute_model_stats",
    "run_iterative_lottery_ticket",
    "discover_multiple_disjoint_tickets",
    "compare_optimization_methods",
]
