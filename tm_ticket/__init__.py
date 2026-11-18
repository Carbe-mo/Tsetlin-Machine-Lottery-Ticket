"""
TM-Ticket: Parameter-Free, Data-Driven Clause Sanitization & Lottery Ticket Discovery
for Tsetlin Machines. Built on top of `tmu` (Tsetlin Machine Unified).
"""

from .model import PrunableTMClassifier, TMCheckpoint
from .sanitization import (
    rule_based_canonicalization,
    length_based_filter,
    compute_clause_scores,
    compute_certified_lossless_mask,
    compute_data_driven_capacities,
    partition_disjoint_tickets_Auto,
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
    "compute_certified_lossless_mask",
    "compute_data_driven_capacities",
    "partition_disjoint_tickets_Auto",
    "select_ticket_mask",
    "compute_model_stats",
    "run_iterative_lottery_ticket",
    "discover_multiple_disjoint_tickets",
    "compare_optimization_methods",
]
