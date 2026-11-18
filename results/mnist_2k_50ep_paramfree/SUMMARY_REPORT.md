# Parameter-Free TM-Ticket: Data-Driven Lottery Tickets & Clause Optimization (50 Epochs)

## 1. Configuration & Data-Driven Calibration Summary
- **Dataset:** `MNIST` (Train: 50000, Val: 10000, Test: 10000)
- **Clauses per class ($N$):** 2000 (Total clauses: 20000)
- **Training & Retraining Epochs:** **50 epochs dense + 50 epochs retrain per lottery ticket**
- **Hyperparameters:** $T = 1500$, $s = 5.0$, Weighted Clauses = `True`, Platform = `CUDA`
- **Dense TM Accuracy:** Test = **98.06%**, Val = **97.86%** (Footprint: `3828.12 KB`)
- **Data-Driven Margin Parameters (Zero Manual Tuning):** $\tau_{\mathrm{data}} = 6756.0$, $e_{\mathrm{val}} = 0.0214$, 1-SE Tolerance $\sigma_{\mathrm{val}} = \pm0.14\%$, 95%-CI Tolerance $= \pm0.28\%$
- **Stage 1 + Certified Lossless Removed:** Empty = 70, Duplicates Merged = 52, Contradictory = 0, Margin-Certified Lossless ($\sum |w|C < \mu/2$) = 630
- **1-SE Optimal Parameter-Free Ticket:** `L1_Positive_MCS (MCS* > 0)` (**13737 clauses**, **2629.35 KB**, **1.46x compression**, Test Acc = **97.76%**)
- **95%-CI Optimal Parameter-Free Ticket:** `L2_Shannon_Perplexity (N_eff = exp(H))` (**9110 clauses**, **1743.71 KB**, **2.20x compression**, Test Acc = **97.87%**)

---

## 2. Parameter-Free Mutually Disjoint Winning Tickets ($M_{\mathrm{auto}} = 6$, Auto Core Size = 9.8% Each, 0% Overlap)

### 2A. Sequential Top-Down Peeling Disjoint Tickets ($\mathcal{T}_a \cap \mathcal{T}_b = \emptyset$)

| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\Theta_{e_r}$) + 50-Ep Retrain (%) | Fine-Tune 50-Ep (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Peel-Ticket #1** | 1955 | 374.20 | 10.23x | 0 | 91.43% | 96.24% | 96.40% | **96.24%** | **96.24%** |
| **Peel-Ticket #2** | 1955 | 374.20 | 10.23x | 0 | 91.15% | 96.76% | 96.49% | **96.76%** | **97.21%** |
| **Peel-Ticket #3** | 1955 | 374.20 | 10.23x | 0 | 90.46% | 96.29% | 96.41% | **96.41%** | **97.40%** |
| **Peel-Ticket #4** | 1955 | 374.20 | 10.23x | 0 | 85.78% | 96.44% | 96.61% | **96.61%** | **97.63%** |
| **Peel-Ticket #5** | 1955 | 374.20 | 10.23x | 0 | 75.80% | 96.35% | 96.33% | **96.33%** | **97.62%** |
| **Peel-Ticket #6** | 1955 | 374.20 | 10.23x | 0 | 50.99% | 96.30% | 96.67% | **96.30%** | **97.73%** |

### 2B. Balanced Snake-Draft Disjoint Tickets ($\mathcal{T}_a \cap \mathcal{T}_b = \emptyset$)

| Ticket ID | Active Clauses | Size (KB) | Compression | Overlap | One-Shot Acc (%) | Rewind ($\Theta_{e_r}$) + 50-Ep Retrain (%) | Fine-Tune 50-Ep (%) | Winning Ticket Acc (%) | Cumulative Ensemble Acc (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Snake-Ticket #1** | 1955 | 374.20 | 10.23x | 0 | 90.79% | 96.38% | 96.27% | **96.27%** | **96.27%** |
| **Snake-Ticket #2** | 1955 | 374.20 | 10.23x | 0 | 90.92% | 96.44% | 96.67% | **96.67%** | **97.35%** |
| **Snake-Ticket #3** | 1955 | 374.20 | 10.23x | 0 | 91.49% | 96.22% | 96.25% | **96.25%** | **97.48%** |
| **Snake-Ticket #4** | 1955 | 374.20 | 10.23x | 0 | 91.85% | 96.28% | 96.37% | **96.37%** | **97.51%** |
| **Snake-Ticket #5** | 1955 | 374.20 | 10.23x | 0 | 91.47% | 96.29% | 96.69% | **96.69%** | **97.73%** |
| **Snake-Ticket #6** | 1955 | 374.20 | 10.23x | 0 | 91.37% | 96.52% | 96.67% | **96.52%** | **97.75%** |

---

## 3. Parameter-Free Data-Driven Lottery-Ticket Hierarchy (50-Epoch Retraining)

| Round | Data-Driven Capacity Level | Retained (%) | Active Clauses | Size (KB) | One-Shot (%) | Rewind $\Theta_{e_r}$ (%) | Rewind $\Theta_0$ (%) | Fine-Tune (%) | Random State (%) | Random Mask (%) | Scratch TM (%) | 1-SE Rule |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `R0_Certified_Lossless (Stage1 + mu(x)/2 Bound)` | 96.2% | 19248 | 3684.19 | 98.07% | **98.07%** | 98.06% | **98.07%** | 98.06% | 98.06% | 98.06% | PASS |
| 1 | `L1_Positive_MCS (MCS* > 0)` | 68.7% | 13737 | 2629.35 | 95.21% | **97.71%** | 97.74% | **97.76%** | 97.71% | 98.02% | 97.68% | PASS |
| 2 | `L2_Shannon_Perplexity (N_eff = exp(H))` | 45.6% | 9110 | 1743.71 | 94.27% | **97.76%** | 97.59% | **97.87%** | 97.60% | 97.47% | 97.71% | 95%-CI |
| 3 | `L3_Participation_Ratio (N_PR = (sum s)^2 / sum s^2)` | 35.5% | 7094 | 1357.84 | 93.74% | **97.68%** | 97.32% | **97.52%** | 97.53% | 97.60% | 97.52% | Below |
| 4 | `L4_Kneedle_AboveMean (s >= mean(s))` | 25.9% | 5182 | 991.87 | 93.29% | **97.57%** | 97.25% | **97.46%** | 97.23% | 97.54% | 97.32% | Below |
| 5 | `L5_Recursive_Core (2nd-Order Kneedle)` | 9.6% | 1916 | 366.73 | 91.40% | **96.56%** | 96.45% | **96.39%** | 96.47% | 96.68% | 96.44% | Below |
| 6 | `L6_Minimal_Backbone (3rd-Order Kneedle)` | 3.6% | 722 | 138.20 | 89.48% | **94.36%** | 94.00% | **94.87%** | 94.02% | 94.53% | 94.37% | Below |

---

## 4. Head-to-Head Clause Optimization Comparison at Data-Driven Capacities (50-Epoch Retraining)

| Data-Driven Budget (%) | Clause Optimization Method | Active Clauses | Size (KB) | Mean Literals | One-Shot Test Acc (%) | 50-Ep Lottery-Ticket Retrained Acc (%) |
|---:|---|---:|---:|---:|---:|---:|
| 35.9% | TM-Ticket (Parameter-Free MCS*) | 7190 | 1376.21 | 28.1 | 94.64% | **97.51%** |
| 35.9% | MCS-Boundary (Margin Score) | 7190 | 1376.21 | 28.4 | 93.47% | **97.60%** |
| 35.9% | Discriminative-Margin (w*[rho+ - rho-]) | 7190 | 1376.21 | 28.0 | 94.60% | **97.53%** |
| 35.9% | Vote-Impact (|w|*rho) | 7190 | 1376.21 | 28.1 | 96.14% | **97.72%** |
| 35.9% | Paper Stage 1+3 (Firing Rate, Global) | 7188 | 1375.83 | 24.9 | 92.39% | **97.61%** |
| 35.9% | Paper Stage 1+2+3 (L_min=2 + Firing) | 7188 | 1375.83 | 26.8 | 92.33% | **97.29%** |
| 35.9% | Weight-Magnitude (|w|) | 7190 | 1376.21 | 32.5 | 97.16% | **97.54%** |
| 35.9% | Random-Pruning (Baseline) | 7190 | 1376.21 | 26.7 | 96.17% | **97.75%** |
| 26.6% | TM-Ticket (Parameter-Free MCS*) | 5310 | 1016.37 | 28.1 | 94.11% | **97.42%** |
| 26.6% | MCS-Boundary (Margin Score) | 5310 | 1016.37 | 28.5 | 92.90% | **97.45%** |
| 26.6% | Discriminative-Margin (w*[rho+ - rho-]) | 5310 | 1016.37 | 28.1 | 94.14% | **97.39%** |
| 26.6% | Vote-Impact (|w|*rho) | 5310 | 1016.37 | 27.6 | 95.24% | **97.40%** |
| 26.6% | Paper Stage 1+3 (Firing Rate, Global) | 5312 | 1016.75 | 24.4 | 90.74% | **97.33%** |
| 26.6% | Paper Stage 1+2+3 (L_min=2 + Firing) | 5312 | 1016.75 | 26.5 | 90.48% | **97.30%** |
| 26.6% | Weight-Magnitude (|w|) | 5310 | 1016.37 | 33.5 | 96.42% | **97.31%** |
| 26.6% | Random-Pruning (Baseline) | 5310 | 1016.37 | 27.0 | 94.35% | **97.26%** |
| 9.8% | TM-Ticket (Parameter-Free MCS*) | 1960 | 375.16 | 28.2 | 92.30% | **96.70%** |
| 9.8% | MCS-Boundary (Margin Score) | 1960 | 375.16 | 28.8 | 91.18% | **96.77%** |
| 9.8% | Discriminative-Margin (w*[rho+ - rho-]) | 1960 | 375.16 | 28.0 | 92.30% | **96.79%** |
| 9.8% | Vote-Impact (|w|*rho) | 1960 | 375.16 | 26.3 | 92.60% | **96.46%** |
| 9.8% | Paper Stage 1+3 (Firing Rate, Global) | 1956 | 374.39 | 23.1 | 85.38% | **95.81%** |
| 9.8% | Paper Stage 1+2+3 (L_min=2 + Firing) | 1956 | 374.39 | 25.4 | 85.07% | **95.85%** |
| 9.8% | Weight-Magnitude (|w|) | 1960 | 375.16 | 35.6 | 91.60% | **96.25%** |
| 9.8% | Random-Pruning (Baseline) | 1960 | 375.16 | 26.7 | 86.02% | **96.74%** |
| 3.7% | TM-Ticket (Parameter-Free MCS*) | 730 | 139.73 | 29.4 | 89.84% | **94.86%** |
| 3.7% | MCS-Boundary (Margin Score) | 730 | 139.73 | 30.4 | 88.66% | **94.92%** |
| 3.7% | Discriminative-Margin (w*[rho+ - rho-]) | 730 | 139.73 | 29.4 | 89.63% | **94.59%** |
| 3.7% | Vote-Impact (|w|*rho) | 730 | 139.73 | 26.7 | 88.77% | **94.85%** |
| 3.7% | Paper Stage 1+3 (Firing Rate, Global) | 734 | 140.49 | 21.1 | 71.71% | **91.56%** |
| 3.7% | Paper Stage 1+2+3 (L_min=2 + Firing) | 734 | 140.49 | 24.5 | 71.94% | **90.90%** |
| 3.7% | Weight-Magnitude (|w|) | 730 | 139.73 | 37.9 | 80.35% | **94.38%** |
| 3.7% | Random-Pruning (Baseline) | 730 | 139.73 | 24.6 | 65.29% | **94.52%** |
