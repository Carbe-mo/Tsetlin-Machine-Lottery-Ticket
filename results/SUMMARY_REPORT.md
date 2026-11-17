# TM-Ticket: Multiple Lottery Tickets & Clause Optimization Results

## 1. Configuration & Dense Tsetlin Machine Baseline
- **Dataset:** `MNIST` (Train: 12000, Val: 3000, Test: 5000)
- **Clauses per class ($N$):** 400 (Total clauses: 4000)
- **Hyperparameters:** $T = 300$, $s = 10.0$, Weighted Clauses = `True`, Platform = `CPU`
- **Dense TM Test Accuracy:** **94.22%** (Val: 93.93%, Footprint: 765.62 KB)
- **Stage 1 Lossless Canonicalization Removed:** Empty = 17, Duplicates Merged = 6, Contradictory = 0, Short ($\ell \le 1$) = 268

---

## 2. Multiple Mutually Disjoint Winning Tickets from One Trained TM (Each 20% of Bank, 0% Overlap)

Every ticket below is extracted from the **same** trained TM with **strictly zero overlap** ($\mathcal{T}_a \cap \mathcal{T}_b = \emptyset$, pairwise Jaccard similarity = `0.000`):

| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\Theta_{e_r}$) + Retrain (%) | Fine-Tune (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ticket #1** | 800 | 153.12 | 5.00x | 0 | 90.38% | 90.18% | 91.36% | **91.36%** | **91.36%** |
| **Ticket #2** | 800 | 153.12 | 5.00x | 0 | 89.64% | 88.86% | 90.56% | **90.56%** | **93.46%** |
| **Ticket #3** | 800 | 153.12 | 5.00x | 0 | 82.04% | 88.06% | 89.90% | **89.90%** | **93.76%** |
| **Ticket #4** | 800 | 153.12 | 5.00x | 0 | 54.78% | 88.56% | 88.34% | **88.34%** | **94.44%** |

---

## 3. Iterative Lottery-Ticket Pruning Spectrum (IMP-TM)

| Round | Retained (%) | Active Clauses | Size (KB) | One-Shot (%) | Rewind $\Theta_{e_r}$ (%) | Rewind $\Theta_0$ (%) | Fine-Tune (%) | Random State Re-init (%) | Random Mask (%) | Scratch Small TM (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.1% | 3124 | 597.95 | 93.58% | **93.72%** | 94.22% | 93.72% | 94.22% | 94.22% | 94.22% |
| 1 | 70.0% | 2800 | 535.94 | 93.98% | **93.32%** | 92.84% | 94.44% | 92.24% | 93.02% | 92.82% |
| 2 | 50.0% | 2000 | 382.81 | 93.80% | **93.28%** | 91.82% | 93.68% | 91.98% | 91.72% | 91.42% |
| 3 | 30.0% | 1200 | 229.69 | 92.54% | **90.92%** | 90.14% | 92.46% | 90.82% | 90.12% | 89.98% |
| 4 | 20.0% | 800 | 153.12 | 91.54% | **88.98%** | 89.08% | 91.56% | 89.18% | 87.62% | 87.28% |
| 5 | 10.0% | 400 | 76.56 | 88.94% | **85.38%** | 84.66% | 88.40% | 83.62% | 84.70% | 84.22% |

---

## 4. Head-to-Head Comparison of All Clause Optimization Techniques

| Budget (%) | Clause Optimization Method | Active Clauses | Size (KB) | Mean Literals | One-Shot Test Acc (%) | Lottery-Ticket Retrained Acc (%) |
|---:|---|---:|---:|---:|---:|---:|
| 50% | TM-Ticket (MCS-Hybrid + C1 + C3) | 2000 | 382.81 | 22.6 | 93.16% | **91.56%** |
| 50% | MCS-Boundary (Pure Margin Score) | 2000 | 382.81 | 23.1 | 93.42% | **92.54%** |
| 50% | Discriminative-Margin (w*[rho+ - rho-]) | 2000 | 382.81 | 22.5 | 93.14% | **92.56%** |
| 50% | Vote-Impact (|w|*rho) | 2000 | 382.81 | 22.3 | 93.38% | **93.00%** |
| 50% | Paper Stage 1+3 (Firing Rate, Global) | 2000 | 382.81 | 19.9 | 92.40% | **92.30%** |
| 50% | Paper Stage 1+2+3 (L_min=2 + Firing) | 2000 | 382.81 | 20.8 | 92.52% | **91.38%** |
| 50% | Weight-Magnitude (|w|) | 2000 | 382.81 | 25.3 | 93.72% | **92.20%** |
| 50% | Random-Pruning (Baseline) | 2000 | 382.81 | 19.9 | 92.56% | **92.26%** |
| 30% | TM-Ticket (MCS-Hybrid + C1 + C3) | 1200 | 229.69 | 23.1 | 91.64% | **91.76%** |
| 30% | MCS-Boundary (Pure Margin Score) | 1200 | 229.69 | 24.6 | 92.30% | **91.52%** |
| 30% | Discriminative-Margin (w*[rho+ - rho-]) | 1200 | 229.69 | 22.9 | 91.46% | **90.42%** |
| 30% | Vote-Impact (|w|*rho) | 1200 | 229.69 | 22.3 | 91.80% | **91.48%** |
| 30% | Paper Stage 1+3 (Firing Rate, Global) | 1200 | 229.69 | 19.7 | 89.22% | **89.86%** |
| 30% | Paper Stage 1+2+3 (L_min=2 + Firing) | 1200 | 229.69 | 20.7 | 89.26% | **89.04%** |
| 30% | Weight-Magnitude (|w|) | 1200 | 229.69 | 28.0 | 92.98% | **91.26%** |
| 30% | Random-Pruning (Baseline) | 1200 | 229.69 | 19.8 | 90.68% | **91.30%** |
| 20% | TM-Ticket (MCS-Hybrid + C1 + C3) | 800 | 153.12 | 23.3 | 90.38% | **90.10%** |
| 20% | MCS-Boundary (Pure Margin Score) | 800 | 153.12 | 25.5 | 91.04% | **89.48%** |
| 20% | Discriminative-Margin (w*[rho+ - rho-]) | 800 | 153.12 | 23.2 | 90.16% | **89.66%** |
| 20% | Vote-Impact (|w|*rho) | 800 | 153.12 | 22.6 | 89.98% | **89.14%** |
| 20% | Paper Stage 1+3 (Firing Rate, Global) | 800 | 153.12 | 19.8 | 87.26% | **87.90%** |
| 20% | Paper Stage 1+2+3 (L_min=2 + Firing) | 800 | 153.12 | 20.8 | 87.34% | **87.62%** |
| 20% | Weight-Magnitude (|w|) | 800 | 153.12 | 30.3 | 90.68% | **89.42%** |
| 20% | Random-Pruning (Baseline) | 800 | 153.12 | 19.9 | 87.44% | **89.54%** |
| 10% | TM-Ticket (MCS-Hybrid + C1 + C3) | 400 | 76.56 | 24.2 | 87.80% | **86.00%** |
| 10% | MCS-Boundary (Pure Margin Score) | 400 | 76.56 | 26.7 | 87.72% | **84.70%** |
| 10% | Discriminative-Margin (w*[rho+ - rho-]) | 400 | 76.56 | 24.3 | 87.60% | **85.08%** |
| 10% | Vote-Impact (|w|*rho) | 400 | 76.56 | 22.4 | 86.44% | **86.02%** |
| 10% | Paper Stage 1+3 (Firing Rate, Global) | 400 | 76.56 | 19.6 | 82.38% | **79.90%** |
| 10% | Paper Stage 1+2+3 (L_min=2 + Firing) | 400 | 76.56 | 20.7 | 82.36% | **81.04%** |
| 10% | Weight-Magnitude (|w|) | 400 | 76.56 | 33.4 | 81.94% | **84.56%** |
| 10% | Random-Pruning (Baseline) | 400 | 76.56 | 19.0 | 76.96% | **86.48%** |
