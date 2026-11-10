# Advances in Deep Reinforcement Learning: From Distributional QR-DQN to Epistemic Uncertainty Ensembles

**Research Team & Authors:**  
- **Pranay Gupta** · Roll No: `102309025`  
- **Idhant Mehta** · Roll No: `102323064`
- **Nakul Makol** · Roll No: `102486010`

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete research codebase, empirical benchmark pipelines, and publication monograph for an international conference-level investigation into value-based Deep Reinforcement Learning (DRL).

Grounded in contemporary literature (*NeurIPS / ICML / ICLR / AAAI*), the project establishes a rigorous three-tiered methodological advancement evaluated across classical continuous control benchmarks (`CartPole-v1` and `Acrobot-v1`) under strict multi-seed protocols ($N=5$) with stratified 95% bootstrap confidence intervals and Mann-Whitney U hypothesis testing.

---

## 🔬 Research Overview & Three-Tiered Methodology

```
                                  ┌────────────────────────────────────────┐
                                  │       Full Unified Apex Agent          │
                                  │   (Distributional + CURL + Ensemble)   │
                                  └───────────────────▲────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
        ┌──────────────┴──────────────┐                               ┌──────────────┴──────────────┐
        │  Iteration 2: Contrastive   │                               │    Iteration 3: Epistemic   │
        │   Representation (CURL)     │                               │      Ensemble Exploration   │
        │  - Bilinear InfoNCE Loss    │                               │  - K=5 Bootstrapped Heads   │
        │  - Momentum Target Encoder  │                               │  - Randomized Prior Nets    │
        │  - Latent Linear Probing    │                               │  - Online Meta-Gradients    │
        └──────────────▲──────────────┘                               └──────────────▲──────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                       ┌──────────────┴──────────────┐
                                       │    Iteration 1: QR-DQN      │
                                       │  - 51 Quantile Atoms        │
                                       │  - Asymmetric Huber Loss    │
                                       │  - Wang Risk Distortion     │
                                       └──────────────▲──────────────┘
                                                      │
                                       ┌──────────────┴──────────────┐
                                       │     Baseline RL Models      │
                                       │  - Vanilla DQN (2013)       │
                                       │  - Enhanced DQN (Double+PER)│
                                       └─────────────────────────────┘
```

### 1. Iteration 1 — Distributional RL via QR-DQN & Risk-Sensitive Policy Extraction
- **Theoretical Basis:** Rather than estimating scalar expectation $\mathbb{E}[Z(s, a)]$, Quantile Regression DQN (*Dabney et al., ICML 2018*) models the full return distribution $Z(s, a)$ via $N=51$ quantile atoms $\{\theta_1, \dots, \theta_N\}$ at fixed cumulative probabilities $\tau_i = \frac{2i - 1}{2N}$.
- **Loss Formulation:** Asymmetric Quantile Huber Loss:
  $$\mathcal{L}_{QR} = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \left| \tau_i - \mathbb{I}(T\theta_j - \theta_i < 0) \right| \cdot \mathcal{L}_{Huber}^\kappa(T\theta_j - \theta_i)$$
- **Novel Risk Distortion:** Integrates Wang distortion risk-sensitive policy extraction $g(\tau) = \Phi(\Phi^{-1}(\tau) + \xi)$ enabling dynamic interpolation across risk-neutral ($\xi=0$), risk-averse ($\xi=0.75$), and risk-seeking ($\xi=-0.75$) regimes.

### 2. Iteration 2 — Self-Supervised Contrastive Representation Learning (CURL / SPR)
- **Theoretical Basis:** Inspired by CURL (*Srinivas et al., ICML 2020*) and SPR (*Schwarzer et al., ICLR 2021*), state features are regularized via a contrastive auxiliary objective.
- **Architecture:** Momentum target encoder ($\mu=0.01$) with a learnable bilinear similarity metric $W \in \mathbb{R}^{64 \times 64}$:
  $$\mathcal{L}_{InfoNCE} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(q_i^\top W k_i^+ / \tau_{temp})}{\sum_{j=1}^B \exp(q_i^\top W k_j / \tau_{temp})}$$
- **Joint Optimization:** $\mathcal{L}_{total} = \mathcal{L}_{QR} + \lambda_{contrast} \cdot \mathcal{L}_{InfoNCE}$.

### 3. Iteration 3 — Epistemic Uncertainty Ensembles & Meta-Gradient Optimization
- **Theoretical Basis:** Bootstrapped DQN (*Osband et al., NeurIPS 2016, 2018*) maintains $K=5$ independent heads with fixed randomized prior functions to prevent posterior collapse.
- **Intrinsic Curiosity Bonus:** Augmented rewards $r^+(s, a) = r(s, a) + \beta \cdot \sigma_{epistemic}(s, a)$ driven by ensemble standard deviation.
- **Online Meta-Gradients:** Bilevel optimization (*Xu et al., NeurIPS 2018; MacKay et al., ICLR 2019*) adaptively tunes $\beta$ based on outer validation Bellman consistency.
- **Full Unified Apex Agent:** Unifies distributional quantile atoms, contrastive representation learning, and epistemic uncertainty ensembles into an end-to-end architecture.

---

## 📁 Repository Directory Structure

```text
.
├── Enhanced-DQN-for-Stable-and-Sample-Efficient-Control.pdf  # Primary Project Research Report & Presentation
├── README.md                            # Comprehensive documentation & reproduction guide
├── LICENSE                              # MIT License
├── requirements.txt                     # Pinned project dependencies
│
├── reports/                             # Comprehensive Monograph & Historical Reports
│   ├── Final_Conference_Research_Report.pdf  # Publication-grade conference research monograph
│   ├── Project_Report.pdf                   # Semester project baseline report
│   └── REPORT.md                            # Detailed theoretical methodology & oral defense prep
│
├── src/                                 # Modular Reinforcement Learning Core
│   ├── dqn_agent.py                     # Vanilla DQN baseline (Mnih et al., 2013)
│   ├── enhanced_agent.py                # Enhanced DQN (Double Q + PER + Soft Updates)
│   ├── network.py                       # Baseline Multi-Layer Perceptron Q-network
│   ├── replay_buffer.py                 # Sum-Segment Tree Prioritized Replay Buffer
│   ├── distributional_network.py        # 51-Atom Quantile Network & Wang Distortion
│   ├── qr_dqn_agent.py                  # Quantile Regression DQN Agent
│   ├── contrastive_encoder.py           # Momentum Target Encoder & Bilinear InfoNCE Module
│   ├── contrastive_qr_dqn_agent.py      # Unified Contrastive + QR-DQN Agent
│   ├── ensemble_agent.py                # Bootstrapped Ensemble DQN with Prior Networks
│   ├── meta_gradient_optimizer.py       # Online Meta-Gradient Hyperparameter Optimizer
│   ├── unified_agent.py                 # Full Unified Apex Agent (CURL + QR + Ensemble)
│   ├── experiment_worker.py             # Standalone picklable worker for parallel execution
│   └── train.py                         # Single-seed training pipeline
│
├── scripts/                             # Benchmark Execution & Statistical Analysis
│   ├── run_training.py                  # Single-run baseline vs enhanced training
│   ├── run_iteration1_parallel.py       # Multi-seed parallel benchmark for Iteration 1
│   ├── run_iteration2_benchmark.py      # Multi-seed parallel benchmark for Iteration 2
│   ├── run_iteration3_benchmark.py      # Multi-seed parallel benchmark for Iteration 3
│   ├── analyze_iteration1.py            # Statistical analysis & Mann-Whitney U testing (It 1)
│   ├── analyze_iteration2.py            # Statistical analysis & AUC evaluation (It 2)
│   ├── analyze_iteration3.py            # Statistical analysis across full ensemble (It 3)
│   ├── generate_plots.py                # Core reward and loss curve plotting
│   ├── generate_pdf_report.py           # Baseline ReportLab PDF generator
│   ├── generate_pdf_report_iteration1.py# ReportLab publication compiler (It 1)
│   ├── generate_pdf_report_iteration2.py# ReportLab publication compiler (It 2)
│   └── generate_final_conference_paper.py# Final unified conference monograph generator
│
└── results/                             # Benchmark Results & Visualizations
    ├── metrics.json                     # Quantitative baseline metrics
    ├── learning_curves.png              # Learning curve comparison
    ├── iteration1/                      # Iteration 1 multi-seed checkpoints & results
    ├── iteration2/                      # Iteration 2 multi-seed checkpoints & results
    └── iteration3/                      # Iteration 3 multi-seed checkpoints & results
```

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/PranayGupta987/deep-rl-advances.git
cd deep-rl-advances
pip install -r requirements.txt
```

### 2. Multi-Seed Parallel Benchmarking
Execute 5-seed parallel benchmarks across any iteration:
```bash
# Iteration 1: QR-DQN & Risk Distortion Benchmark
python scripts/run_iteration1_parallel.py

# Iteration 2: Self-Supervised Contrastive RL Benchmark
python scripts/run_iteration2_benchmark.py

# Iteration 3: Epistemic Ensemble & Unified Apex Benchmark
python scripts/run_iteration3_benchmark.py
```

### 3. Statistical Analysis & Hypothesis Testing
Evaluate statistical significance with 95% bootstrap confidence intervals (2,000 resamples):
```bash
python scripts/analyze_iteration1.py
python scripts/analyze_iteration2.py
python scripts/analyze_iteration3.py
```

### 4. Build Conference Publication Monograph
Generate the comprehensive two-pass paginated publication PDF:
```bash
python scripts/generate_final_conference_paper.py
```
Output: `reports/Final_Conference_Research_Report.pdf`

---

## 📊 Summary of Benchmark Results

Evaluated over $N=5$ seeds (`42, 100, 2013, 2024, 777`) on 200-episode horizons:

| Environment | Architecture / Variant | Final 50-Ep Return | Std Dev | 95% Bootstrap CI | p-value (vs Enhanced) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CartPole-v1** | Vanilla DQN (2013) | 148.62 | 53.48 | [104.2, 192.1] | $p < 0.05$ |
| **CartPole-v1** | Enhanced DQN (Double+PER) | 135.16 | 45.19 | [102.8, 178.6] | Baseline |
| **CartPole-v1** | **QR-DQN (Risk-Neutral)** | **182.36** | **63.85** | **[133.2, 237.4]** | **$p = 0.038$** |
| **CartPole-v1** | **QR-DQN (Risk-Seeking)** | **182.44** | **63.82** | **[133.4, 237.3]** | **$p = 0.038$** |
| **CartPole-v1** | **Full Unified Apex Agent** | **288.90** | **6.20** | **[285.2, 292.6]** | **$p < 0.001$** |
| **Acrobot-v1** | Vanilla DQN (2013) | -107.75 | 6.84 | [-113.2, -102.8] | $p = 0.042$ |
| **Acrobot-v1** | Enhanced DQN (Double+PER) | -107.17 | 10.05 | [-115.8, -98.6] | Baseline |
| **Acrobot-v1** | **QR-DQN (Risk-Averse)** | **-101.38** | **6.01** | **[-106.8, -96.2]** | **$p = 0.024$** |
| **Acrobot-v1** | **Contrastive QR-DQN** | **-86.50** | **4.80** | **[-89.4, -83.6]** | **$p < 0.01$** |
| **Acrobot-v1** | **Full Unified Apex Agent** | **-76.80** | **3.50** | **[-78.9, -74.7]** | **$p < 0.001$** |

---

## 👥 Project Team & Authors
- **Pranay Gupta** · Roll No: `102309025`
- **Idhant Mehta** · Roll No: `102323064`

---

## 📜 Citation & License
This project is open-source under the [MIT License](LICENSE).
```bibtex
@article{gupta2026deeprladvances,
  title={Advances in Deep Reinforcement Learning: From Distributional QR-DQN to Epistemic Uncertainty Ensembles},
  author={Gupta, Pranay and Mehta, Idhant},
  year={2026}
}
```
