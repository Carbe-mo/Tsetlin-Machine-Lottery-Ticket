# TM-Ticket: Margin-Aware Clause Sanitization & Multiple Lottery Ticket Discovery for Tsetlin Machines

Built on [`tmu`](https://github.com/cair/tmu) (Tsetlin Machine Unified), **TM-Ticket** unifies post-training clause sanitization with a **Lottery Ticket Hypothesis (LTH)** iterative prune–rewind–retrain framework for Tsetlin Machines.

---

## Key Features & Contributions

1. **Stage 1 — Extended Lossless Canonicalization ($\mathcal{S}_{\mathrm{rule}}^{+}$):**
   - Removes **empty clauses** ($\ell_{k,j} = 0$), **contradictory / tautological clauses** ($x_i \land \neg x_i$), and **encoding-invalid clauses** (thermometer threshold violations).
   - Detects **duplicate clauses** ($S_{k,j_1} = S_{k,j_2}$) via fast 64-bit hashing and **losslessly accumulates their weights** ($w_{k,j_1} \leftarrow w_{k,j_1} + w_{k,j_2}$).
   - Optional **logical subsumption pruning** ($S_{k,j_1} \subset S_{k,j_2}$).

2. **Stage 2 & 3 — Margin Contribution Score ($\mathrm{MCS}$ & $\mathrm{MCS\text{-}Hybrid}$):**
   - Scores every clause by its first-order contribution to the classification margin $\mu(\mathbf{x}) = v_y(\mathbf{x}) - \max_{q \neq y} v_q(\mathbf{x})$, weighted toward boundary samples via $\omega(\mathbf{x}) = \exp(-\max(\mu(\mathbf{x}), 0)/\tau)$.
   - Automatically identifies **harmful clauses** ($\mathrm{MCS}_{k,j} < 0$) whose removal improves generalization.
   - Benchmarks head-to-head against all clause optimization criteria:
     - `MCS-Hybrid` (Proposed Margin + Discriminative Score)
     - `MCS-Boundary` (Pure Margin Score)
     - `Discriminative-Margin` ($w_{k,j}(\rho_{k,j}^+ - \rho_{k,j}^-)$)
     - `Vote-Impact` ($|w_{k,j}| \cdot \rho_{k,j}$)
     - `Firing-Rate Sanitization` (Reference Paper Stage 1 + 3)
     - `Length + Firing Sanitization` (Reference Paper Stage 1 + 2 + 3, $L_{\min}=2$)
     - `Weight-Magnitude` ($|w_{k,j}|$)
     - `Random Pruning`

3. **Stage 4 — Class-Balanced Quotas & Vote Renormalization:**
   - Prevents hard classes from collapsing under global thresholds and renormalizes class vote sums post-pruning.

4. **Multiple Mutually Disjoint Winning Lottery Tickets & Iterative Pruning (IMP-TM):**
   - **Multiple Disjoint Tickets (`discover_multiple_disjoint_tickets`):** Extracts $M$ mutually exclusive winning tickets ($\mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_M$ with $\mathcal{T}_a \cap \mathcal{T}_b = \emptyset$, Jaccard overlap = `0.000`) from a **single** trained TM, demonstrating that multiple disjoint sub-banks can independently match dense accuracy after rewinding to $\Theta_{e_r}$ and retraining—and that their ensemble surpasses the full model.
   - **Iterative Lottery-Ticket Loop (`run_iterative_lottery_ticket`):** Prunes progressively from 100% down to 10% active clauses, comparing **Rewind to Early Epoch ($\Theta_{e_r}$)**, **Rewind to Initialization ($\Theta_0$)**, **Fine-Tuning ($\Theta_{\mathrm{final}}$)**, **One-Shot Sanitization**, **Random State Re-initialization**, **Random Mask**, and **Small TM Trained from Scratch**.

---

## Repository Structure

```text
.
├── tm_ticket/
│   ├── __init__.py
│   ├── model.py          # PrunableTMClassifier(TMClassifier): fast bit-matrix extraction, masking, TA state rewinding
│   ├── sanitization.py   # Rule canonicalization, length filter, MCS & all clause importance scorers
│   ├── lottery.py        # Disjoint multi-ticket discovery, IMP-TM loop, and multi-method benchmark
│   └── plotting.py       # Generates publication-ready PDF & PNG figures
├── run_experiments.py    # CLI runner with 'fast', 'standard', and 'full' (coder.uia.no) presets
├── requirements.txt
└── results/              # Generated JSON, CSV tables, PDF/PNG plots, and SUMMARY_REPORT.md
```

---

## Running on `coder.uia.no` (or Local Machine)

### 1. Setup

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_NAME>
pip install -r requirements.txt
```
*(If running with GPU acceleration on `coder.uia.no`, ensure `pycuda` is installed: `pip install pycuda`.)*

### 2. Run Presets

#### A. Full Publication Suite on `coder.uia.no` (2,000 clauses/class, full 60K MNIST, CUDA or CPU)
```bash
# With CUDA GPU:
python run_experiments.py --preset full --platform CUDA --clauses 2000 --T 1500 --s 10.0 --epochs 25 --retrain-epochs 10

# Or on multi-core CPU:
python run_experiments.py --preset full --platform CPU --clauses 2000 --T 1500 --s 10.0 --epochs 25 --retrain-epochs 10
```

#### B. Standard Benchmark (1,000 clauses/class)
```bash
python run_experiments.py --preset standard
```

#### C. Fast Verification (~2 minutes on CPU)
```bash
python run_experiments.py --preset fast
```

#### D. Run on Fashion-MNIST or Kuzushiji-MNIST
```bash
python run_experiments.py --preset full --dataset fmnist --platform CUDA
python run_experiments.py --preset full --dataset kmnist --platform CUDA
```

---

## Outputs Produced in `results/`

- `SUMMARY_REPORT.md`: Complete markdown tables of all discovered lottery tickets, their accuracies, sizes (KB), overlaps, and method comparisons.
- `disjoint_lottery_tickets.csv`: Standalone & ensemble accuracies of Ticket #1, Ticket #2, Ticket #3, Ticket #4 (0% overlap).
- `iterative_lottery_tickets.csv`: Accuracy across sparsity levels (100% $\rightarrow$ 10%) for all rewinding and baseline regimes.
- `optimization_methods_comparison.csv`: Head-to-head comparison of all 8 clause optimization criteria.
- `fig1_iterative_lottery_tickets.pdf` / `.png`
- `fig2_multiple_disjoint_tickets.pdf` / `.png`
- `fig3_optimization_methods_comparison.pdf` / `.png`
