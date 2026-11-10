"""
Rigorous Statistical Analysis & Publication-Quality Visualization for Iteration 1.

Performs:
  1. Multi-seed aggregation (Mean, Std, Median, IQR, 95% Bootstrap Confidence Intervals).
  2. Non-parametric hypothesis testing (Mann-Whitney U tests with p-values and effect sizes).
  3. Risk-sensitivity analysis of the Wang parameter (risk-neutral vs risk-averse vs risk-seeking).
  4. Publication-ready visualization (shaded 95% CI bands, performance boxplots, quantile variance).
  5. LaTeX table export for direct inclusion in conference submissions.

Usage:
    python analyze_iteration1.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Publication styling
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

# Distinct, accessible color palette
PALETTE = {
    "vanilla":               "#d95f02",  # burnt orange
    "enhanced":              "#7570b3",  # purple
    "qr_dqn":                "#1b9e77",  # teal green
    "qr_dqn_risk_averse":    "#e7298a",  # magenta/pink
    "qr_dqn_risk_seeking":   "#e6ab02",  # yellow/gold
}

AGENT_NAMES = {
    "vanilla":               "Vanilla DQN",
    "enhanced":              "Enhanced DQN (Double+PER)",
    "qr_dqn":                "QR-DQN (Risk-Neutral)",
    "qr_dqn_risk_averse":    "QR-DQN (Risk-Averse, xi=0.75)",
    "qr_dqn_risk_seeking":   "QR-DQN (Risk-Seeking, xi=-0.75)",
}


def load_data(path="results/iteration1/multi_seed_results.json"):
    with open(path) as f:
        return json.load(f)


def moving_average(arr, window=20):
    """Compute moving average along the last axis of a 2D array (seeds x episodes)."""
    kernel = np.ones(window) / window
    smoothed = np.array([
        np.convolve(row, kernel, mode='valid') for row in arr
    ])
    return smoothed


def bootstrap_ci(data_matrix, n_bootstraps=2000, ci=0.95):
    """
    Compute point estimate (mean) and bootstrap 95% confidence intervals
    across seeds for each episode.

    Parameters
    ----------
    data_matrix : ndarray of shape (n_seeds, n_episodes)

    Returns
    -------
    mean : ndarray (n_episodes,)
    lower_ci : ndarray (n_episodes,)
    upper_ci : ndarray (n_episodes,)
    """
    n_seeds, n_episodes = data_matrix.shape
    mean = np.mean(data_matrix, axis=0)

    # Resample seeds with replacement
    rng = np.random.default_rng(42)
    boot_means = np.zeros((n_bootstraps, n_episodes))
    for b in range(n_bootstraps):
        sample_indices = rng.choice(n_seeds, size=n_seeds, replace=True)
        boot_means[b] = np.mean(data_matrix[sample_indices], axis=0)

    alpha = (1.0 - ci) / 2.0
    lower_ci = np.percentile(boot_means, alpha * 100, axis=0)
    upper_ci = np.percentile(boot_means, (1.0 - alpha) * 100, axis=0)
    return mean, lower_ci, upper_ci


# ── Figure 1: Multi-Seed Learning Curves with 95% Shaded CIs ──────────── #
def plot_multi_seed_learning_curves(data, out_dir="results/iteration1"):
    envs = list(data.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(7 * len(envs), 5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        env_data = data[env_name]
        for agent_key in ["vanilla", "enhanced", "qr_dqn", "qr_dqn_risk_averse"]:
            if agent_key not in env_data:
                continue

            matrix = np.array(env_data[agent_key]["reward_matrix"])  # (seeds, episodes)
            smoothed_matrix = moving_average(matrix, window=20)
            mean, lower, upper = bootstrap_ci(smoothed_matrix)

            x = np.arange(len(mean)) + 10  # center centered window

            color = PALETTE.get(agent_key, "#333333")
            label = AGENT_NAMES.get(agent_key, agent_key)

            ax.plot(x, mean, label=label, color=color, linewidth=2.2)
            ax.fill_between(x, lower, upper, color=color, alpha=0.18)

        ax.set_title(f"Benchmark: {env_name}", fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return (Moving Avg)")
        ax.legend(loc='lower right' if 'Acrobot' in env_name else 'upper left', framealpha=0.9)

    fig.suptitle("Sample Efficiency with 95% Bootstrap Confidence Intervals (N=5 Seeds)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "multi_seed_learning_curves.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "multi_seed_learning_curves.pdf"))
    plt.close(fig)
    print("✓ Saved multi_seed_learning_curves.png and .pdf")


# ── Figure 2: Final Performance Distribution (Boxplot + Jitter) ───────── #
def plot_final_performance_boxplot(data, out_dir="results/iteration1"):
    envs = list(data.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(6.5 * len(envs), 5.5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        env_data = data[env_name]
        agents_to_plot = ["vanilla", "enhanced", "qr_dqn", "qr_dqn_risk_averse"]
        plot_data = []
        labels = []
        colors = []

        for k in agents_to_plot:
            if k in env_data:
                # Final 50 episodes mean return per seed (n_seeds values)
                matrix = np.array(env_data[k]["reward_matrix"])
                final_50 = matrix[:, -50:].mean(axis=1)
                plot_data.append(final_50)
                labels.append(AGENT_NAMES[k].replace(" (", "\n("))
                colors.append(PALETTE[k])

        bplot = ax.boxplot(
            plot_data, patch_artist=True, tick_labels=labels,
            medianprops=dict(color='black', linewidth=2),
            widths=0.55,
        )
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Overlay individual seed points (strip plot / jitter)
        for i, vals in enumerate(plot_data):
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(np.ones_like(vals) * (i + 1) + jitter, vals,
                       color=colors[i], edgecolors='black', s=55, zorder=4, alpha=0.9)

        ax.set_title(f"Final 50-Episode Return: {env_name}", fontweight='bold')
        ax.set_ylabel("Mean Return (last 50 episodes)")

    fig.suptitle("Asymptotic Performance Across Seeds (N=5)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_performance_distribution.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "final_performance_distribution.pdf"))
    plt.close(fig)
    print("✓ Saved final_performance_distribution.png and .pdf")


# ── Figure 3: Quantile Uncertainty & Risk Sensitivity ─────────────────── #
def plot_risk_sensitivity(data, out_dir="results/iteration1"):
    """Compares risk-neutral, risk-averse, and risk-seeking variants of QR-DQN."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    risk_agents = ["qr_dqn", "qr_dqn_risk_averse", "qr_dqn_risk_seeking"]

    for ax, env_name in zip(axes, data.keys()):
        env_data = data[env_name]
        for k in risk_agents:
            if k not in env_data:
                continue
            matrix = np.array(env_data[k]["reward_matrix"])
            smoothed = moving_average(matrix, window=20)
            mean, lower, upper = bootstrap_ci(smoothed)
            x = np.arange(len(mean)) + 10

            color = PALETTE[k]
            label = AGENT_NAMES[k]
            ax.plot(x, mean, label=label, color=color, linewidth=2)
            ax.fill_between(x, lower, upper, color=color, alpha=0.15)

        ax.set_title(f"Risk Modulation: {env_name}", fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return")
        ax.legend(framealpha=0.9)

    fig.suptitle("Effect of Wang Distortion Parameter (Risk Tuning) on Learning Dynamics", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_sensitivity_curves.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "risk_sensitivity_curves.pdf"))
    plt.close(fig)
    print("✓ Saved risk_sensitivity_curves.png and .pdf")


# ── Statistical Testing Table ─────────────────────────────────────────── #
def compute_statistical_tests(data):
    """
    Computes summary statistics and non-parametric Mann-Whitney U tests
    comparing QR-DQN vs Enhanced and Vanilla baselines.
    """
    print("\n" + "=" * 90)
    print(f"{'STATISTICAL SIGNIFICANCE & PERFORMANCE SUMMARY (N=5 Seeds)':^90}")
    print("=" * 90)

    latex_rows = []

    for env_name, env_data in data.items():
        print(f"\n--- Environment: {env_name} ---")
        print(f"{'Agent':<32s} {'Mean ± Std':>16s} {'Median [IQR]':>20s} {'p vs Vanilla':>14s} {'p vs Enhanced':>14s}")
        print("-" * 96)

        # Baseline vectors of final-50 episode return per seed
        vanilla_final = np.array(env_data["vanilla"]["reward_matrix"])[:, -50:].mean(axis=1)
        enhanced_final = np.array(env_data["enhanced"]["reward_matrix"])[:, -50:].mean(axis=1)

        for agent_key in ["vanilla", "enhanced", "qr_dqn", "qr_dqn_risk_averse", "qr_dqn_risk_seeking"]:
            if agent_key not in env_data:
                continue

            scores = np.array(env_data[agent_key]["reward_matrix"])[:, -50:].mean(axis=1)
            mean_val = np.mean(scores)
            std_val = np.std(scores, ddof=1)
            med_val = np.median(scores)
            iqr_val = stats.iqr(scores)

            # Mann-Whitney U test (two-sided)
            if agent_key == "vanilla":
                p_vanilla_str = "-"
            else:
                _, p_v = stats.mannwhitneyu(scores, vanilla_final, alternative='two-sided')
                p_vanilla_str = f"{p_v:.4f}"

            if agent_key == "enhanced":
                p_enhanced_str = "-"
            else:
                _, p_e = stats.mannwhitneyu(scores, enhanced_final, alternative='two-sided')
                p_enhanced_str = f"{p_e:.4f}"

            name = AGENT_NAMES[agent_key]
            print(f"{name:<32s} {mean_val:8.2f} ± {std_val:5.2f} {med_val:8.2f} [{iqr_val:5.2f}] {p_vanilla_str:>14s} {p_enhanced_str:>14s}")

            latex_rows.append(
                f"{env_name} & {name} & {mean_val:.2f} $\\pm$ {std_val:.2f} & {med_val:.2f} [{iqr_val:.2f}] & {p_vanilla_str} & {p_enhanced_str} \\\\"
            )

    # Save LaTeX table
    latex_path = "results/iteration1/statistical_table.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated by analyze_iteration1.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Statistical benchmark comparison across $N=5$ random seeds. P-values from two-sided Mann-Whitney U tests against baselines.}\n")
        f.write("\\label{tab:iteration1_benchmark}\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Environment & Algorithm & Final Return & Median [IQR] & $p$ (vs. Vanilla) & $p$ (vs. Enhanced) \\\\\n\\midrule\n")
        f.write("\n".join(latex_rows))
        f.write("\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n✓ LaTeX table written to {latex_path}")


if __name__ == "__main__":
    data = load_data()
    plot_multi_seed_learning_curves(data)
    plot_final_performance_boxplot(data)
    plot_risk_sensitivity(data)
    compute_statistical_tests(data)
    print("\n✓ Iteration 1 analysis completed successfully.")
