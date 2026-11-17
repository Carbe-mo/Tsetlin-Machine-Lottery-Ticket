# TM-Ticket: Multiple Lottery Tickets & Clause Optimization Results

## 1. Configuration & Dense Tsetlin Machine Baseline
- **Dataset:** `MNIST` (Train: 30000, Val: 5000, Test: 10000)
- **Clauses per class ($N$):** 2000 (Total clauses: 20000)
- **Hyperparameters:** $T = 1500$, $s = 5.0$, Weighted Clauses = `True`, Platform = `CUDA`
- **Dense TM Test Accuracy:** **97.25%** (Val: 97.28%, Footprint: 3828.12 KB)
- **Stage 1 Lossless Canonicalization Removed:** Empty = 101, Duplicates Merged = 85, Contradictory = 0, Short ($\ell \le 1$) = 880

---

## 2. Multiple Mutually Disjoint Winning Tickets from One Trained TM (Each 20% of Bank, 0% Overlap)

Every ticket below is extracted from the **same** trained TM with **strictly zero overlap** ($\mathcal{T}_a \cap \mathcal{T}_b = \emptyset$, pairwise Jaccard similarity = `0.000`):

| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\Theta_{e_r}$) + Retrain (%) | Fine-Tune (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ticket #1** | 4000 | 765.62 | 5.00x | 0 | 93.45% | 95.43% | 96.30% | **96.30%** | **96.30%** |
| **Ticket #2** | 4000 | 765.62 | 5.00x | 0 | 93.68% | 95.49% | 96.16% | **96.16%** | **96.90%** |
| **Ticket #3** | 4000 | 765.62 | 5.00x | 0 | 83.73% | 95.24% | 95.91% | **95.91%** | **97.01%** |
| **Ticket #4** | 4000 | 765.62 | 5.00x | 0 | 25.89% | 95.51% | 95.54% | **95.54%** | **97.14%** |

---

## 3. Iterative Lottery-Ticket Pruning Spectrum (IMP-TM)

| Round | Retained (%) | Active Clauses | Size (KB) | One-Shot (%) | Rewind $\Theta_{e_r}$ (%) | Rewind $\Theta_0$ (%) | Fine-Tune (%) | Random State Re-init (%) | Random Mask (%) | Scratch Small TM (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64.6% | 12926 | 2474.12 | 80.56% | **83.47%** | 97.25% | 83.47% | 97.25% | 97.25% | 97.25% |
| 1 | 74.7% | 14930 | 2857.70 | 96.66% | **96.64%** | 96.29% | 97.51% | 96.42% | 96.47% | 96.24% |
| 2 | 48.6% | 9723 | 1861.04 | 95.09% | **96.38%** | 95.57% | 96.96% | 96.04% | 96.32% | 96.17% |
| 3 | 33.8% | 6753 | 1292.57 | 94.76% | **95.78%** | 95.49% | 96.86% | 95.83% | 95.83% | 95.47% |
| 4 | 24.1% | 4827 | 923.92 | 94.52% | **95.65%** | 94.97% | 96.59% | 95.76% | 95.24% | 95.20% |
| 5 | 14.5% | 2909 | 556.80 | 94.48% | **95.17%** | 94.53% | 96.11% | 94.62% | 95.37% | 95.09% |
| 6 | 9.7% | 1934 | 370.18 | 94.12% | **94.52%** | 94.06% | 95.65% | 93.65% | 93.93% | 94.19% |

---

## 4. Head-to-Head Comparison of All Clause Optimization Techniques

| Budget (%) | Clause Optimization Method | Active Clauses | Size (KB) | Mean Literals | One-Shot Test Acc (%) | Lottery-Ticket Retrained Acc (%) |
|---:|---|---:|---:|---:|---:|---:|
| 50% | TM-Ticket (MCS-Hybrid + C1 + C3) | 9909 | 1896.64 | 23.8 | 94.77% | **96.60%** |
| 50% | MCS-Boundary (Pure Margin Score) | 9117 | 1745.05 | 24.1 | 86.70% | **95.84%** |
| 50% | Discriminative-Margin (w*[rho+ - rho-]) | 9913 | 1897.41 | 23.7 | 94.74% | **96.44%** |
| 50% | Vote-Impact (|w|*rho) | 10000 | 1914.06 | 24.6 | 96.51% | **96.24%** |
| 50% | Paper Stage 1+3 (Firing Rate, Global) | 10000 | 1914.06 | 21.7 | 94.14% | **96.39%** |
| 50% | Paper Stage 1+2+3 (L_min=2 + Firing) | 10000 | 1914.06 | 22.6 | 94.35% | **96.00%** |
| 50% | Weight-Magnitude (|w|) | 10000 | 1914.06 | 26.8 | 97.23% | **96.24%** |
| 50% | Random-Pruning (Baseline) | 10000 | 1914.06 | 22.1 | 96.89% | **96.39%** |
| 30% | TM-Ticket (MCS-Hybrid + C1 + C3) | 6000 | 1148.44 | 24.6 | 94.06% | **95.69%** |
| 30% | MCS-Boundary (Pure Margin Score) | 5840 | 1117.81 | 24.5 | 89.77% | **95.87%** |
| 30% | Discriminative-Margin (w*[rho+ - rho-]) | 6000 | 1148.44 | 24.6 | 93.97% | **95.82%** |
| 30% | Vote-Impact (|w|*rho) | 6000 | 1148.44 | 25.1 | 95.35% | **95.96%** |
| 30% | Paper Stage 1+3 (Firing Rate, Global) | 6000 | 1148.44 | 21.8 | 90.95% | **95.91%** |
| 30% | Paper Stage 1+2+3 (L_min=2 + Firing) | 6000 | 1148.44 | 22.9 | 91.24% | **95.67%** |
| 30% | Weight-Magnitude (|w|) | 6000 | 1148.44 | 29.2 | 96.70% | **95.92%** |
| 30% | Random-Pruning (Baseline) | 6000 | 1148.44 | 22.4 | 95.82% | **96.00%** |
| 20% | TM-Ticket (MCS-Hybrid + C1 + C3) | 4000 | 765.62 | 25.3 | 93.45% | **95.45%** |
| 20% | MCS-Boundary (Pure Margin Score) | 4000 | 765.62 | 25.0 | 90.66% | **95.43%** |
| 20% | Discriminative-Margin (w*[rho+ - rho-]) | 4000 | 765.62 | 25.2 | 93.47% | **95.60%** |
| 20% | Vote-Impact (|w|*rho) | 4000 | 765.62 | 25.3 | 94.39% | **95.27%** |
| 20% | Paper Stage 1+3 (Firing Rate, Global) | 4000 | 765.62 | 21.7 | 88.86% | **94.69%** |
| 20% | Paper Stage 1+2+3 (L_min=2 + Firing) | 4000 | 765.62 | 22.8 | 89.02% | **95.32%** |
| 20% | Weight-Magnitude (|w|) | 4000 | 765.62 | 31.0 | 95.68% | **95.50%** |
| 20% | Random-Pruning (Baseline) | 4000 | 765.62 | 22.4 | 93.97% | **95.32%** |
| 10% | TM-Ticket (MCS-Hybrid + C1 + C3) | 2000 | 382.81 | 25.9 | 92.18% | **94.47%** |
| 10% | MCS-Boundary (Pure Margin Score) | 2000 | 382.81 | 25.9 | 90.07% | **95.18%** |
| 10% | Discriminative-Margin (w*[rho+ - rho-]) | 2000 | 382.81 | 25.8 | 92.09% | **94.45%** |
| 10% | Vote-Impact (|w|*rho) | 2000 | 382.81 | 25.5 | 92.20% | **95.10%** |
| 10% | Paper Stage 1+3 (Firing Rate, Global) | 2000 | 382.81 | 20.6 | 82.47% | **93.44%** |
| 10% | Paper Stage 1+2+3 (L_min=2 + Firing) | 2000 | 382.81 | 22.2 | 83.99% | **93.72%** |
| 10% | Weight-Magnitude (|w|) | 2000 | 382.81 | 32.9 | 91.03% | **94.33%** |
| 10% | Random-Pruning (Baseline) | 2000 | 382.81 | 22.4 | 87.51% | **94.29%** |
