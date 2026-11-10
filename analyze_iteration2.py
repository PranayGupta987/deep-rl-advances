"""
Rigorous Statistical Analysis & Publication-Quality Visualization for Iteration 2:
Contrastive Representation Learning (CURL / SPR) + QR-DQN.

Performs:
  1. Multi-seed statistical aggregation (Mean, Std, Median, IQR, 95% Bootstrap CIs).
  2. Non-parametric hypothesis testing (Mann-Whitney U tests, p-values, Cohen's d).
  3. Sample efficiency and Area Under Curve (AUC) comparative analysis.
  4. Publication-ready visualization (shaded 95% CI bands, AUC distributions, ablation).
  5. LaTeX table export for direct conference inclusion.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

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

PALETTE = {
    "enhanced":              "#7570b3",  # purple (baseline)
    "qr_dqn":                "#1b9e77",  # teal green (Iteration 1)
    "contrastive_qr_dqn":    "#d95f02",  # bold orange (Iteration 2 - novel)
}

AGENT_NAMES = {
    "enhanced":              "Enhanced DQN (Double+PER)",
    "qr_dqn":                "QR-DQN (Distributional RL alone)",
    "contrastive_qr_dqn":    "Contrastive QR-DQN (CURL + QR-DQN)",
}


def load_data(path="results/iteration2/multi_seed_results.json"):
    with open(path) as f:
        return json.load(f)


def moving_average(arr, window=20):
    kernel = np.ones(window) / window
    return np.array([np.convolve(row, kernel, mode='valid') for row in arr])


def bootstrap_ci(data_matrix, n_bootstraps=2000, ci=0.95):
    n_seeds, n_episodes = data_matrix.shape
    mean = np.mean(data_matrix, axis=0)
    rng = np.random.default_rng(42)
    boot_means = np.zeros((n_bootstraps, n_episodes))
    for b in range(n_bootstraps):
        sample_indices = rng.choice(n_seeds, size=n_seeds, replace=True)
        boot_means[b] = np.mean(data_matrix[sample_indices], axis=0)
    alpha = (1.0 - ci) / 2.0
    lower_ci = np.percentile(boot_means, alpha * 100, axis=0)
    upper_ci = np.percentile(boot_means, (1.0 - alpha) * 100, axis=0)
    return mean, lower_ci, upper_ci


def plot_learning_curves(data, out_dir="results/iteration2"):
    os.makedirs(out_dir, exist_ok=True)
    envs = list(data.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(14, 5.2), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, envs):
        env_data = data[env_name]
        for ag_key in ["enhanced", "qr_dqn", "contrastive_qr_dqn"]:
            if ag_key not in env_data:
                continue
            r_mat = np.array(env_data[ag_key]["reward_matrix"])
            smoothed = moving_average(r_mat, window=15)
            mean, lower, upper = bootstrap_ci(smoothed)
            episodes = np.arange(15, 15 + len(mean))
            color = PALETTE[ag_key]
            lbl = AGENT_NAMES[ag_key]
            ax.plot(episodes, mean, label=lbl, color=color, lw=2.2)
            ax.fill_between(episodes, lower, upper, color=color, alpha=0.18)

        ax.set_title(f"Benchmark: {env_name}", fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Undiscounted Cumulative Return")
        ax.legend(loc="lower right" if "CartPole" in env_name else "lower right")

    plt.suptitle("Iteration 2: Contrastive Representation Learning + QR-DQN vs Baselines\n(Mean ± 95% Bootstrap CI over N=5 Seeds)", fontsize=13, y=1.02)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "fig1_iteration2_learning_curves.png")
    pdf_path = os.path.join(out_dir, "fig1_iteration2_learning_curves.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {png_path}")


def plot_auc_sample_efficiency(data, out_dir="results/iteration2"):
    fig, axes = plt.subplots(1, len(data), figsize=(12, 5))
    if len(data) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, data.keys()):
        env_data = data[env_name]
        agents = ["enhanced", "qr_dqn", "contrastive_qr_dqn"]
        auc_data = []
        for ag in agents:
            r_mat = np.array(env_data[ag]["reward_matrix"])
            auc = np.mean(r_mat, axis=1)  # mean return over all episodes = Area Under Curve normalized
            auc_data.append(auc)

        bp = ax.boxplot(auc_data, labels=[AGENT_NAMES[ag].split(' (')[0] for ag in agents],
                        patch_artist=True, widths=0.55)
        for patch, ag in zip(bp['boxes'], agents):
            patch.set_facecolor(PALETTE[ag])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2.0)

        # Overlay individual seed points
        for i, (ag, vals) in enumerate(zip(agents, auc_data)):
            x = np.random.normal(i + 1, 0.04, size=len(vals))
            ax.scatter(x, vals, color='black', alpha=0.8, zorder=5, s=28)

        ax.set_title(f"Sample Efficiency (AUC): {env_name}", fontweight='bold')
        ax.set_ylabel("Normalized Area Under Curve (AUC)")

    plt.suptitle("Sample Efficiency Distribution Across Seeds (N=5)", fontsize=13, y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, "fig2_sample_efficiency_auc.png")
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {p}")


def perform_statistical_analysis(data, out_dir="results/iteration2"):
    summary = {}
    latex_rows = []

    print("\n" + "=" * 80)
    print(" ITERATION 2: STATISTICAL SIGNIFICANCE & ABLATION RESULTS (N=5 Seeds)")
    print("=" * 80)

    for env_name, env_data in data.items():
        summary[env_name] = {}
        print(f"\n--- Environment: {env_name} ---")

        for ag_key in ["enhanced", "qr_dqn", "contrastive_qr_dqn"]:
            r_mat = np.array(env_data[ag_key]["reward_matrix"])
            final_50 = np.mean(r_mat[:, -50:], axis=1)
            auc = np.mean(r_mat, axis=1)

            mean_val = float(np.mean(final_50))
            std_val = float(np.std(final_50, ddof=1))
            median_val = float(np.median(final_50))
            iqr_val = float(stats.iqr(final_50))

            summary[env_name][ag_key] = {
                "final_50_mean": mean_val,
                "final_50_std": std_val,
                "final_50_median": median_val,
                "final_50_iqr": iqr_val,
                "auc_mean": float(np.mean(auc)),
                "auc_std": float(np.std(auc, ddof=1)),
            }
            print(f"  {AGENT_NAMES[ag_key]:<36s}: Final50 = {mean_val:7.2f} ± {std_val:5.2f} | Median = {median_val:7.2f} (IQR={iqr_val:5.2f}) | AUC = {np.mean(auc):7.2f}")

        # Pairwise Hypothesis Tests
        c_vals = np.mean(np.array(env_data["contrastive_qr_dqn"]["reward_matrix"])[:, -50:], axis=1)
        qr_vals = np.mean(np.array(env_data["qr_dqn"]["reward_matrix"])[:, -50:], axis=1)
        enh_vals = np.mean(np.array(env_data["enhanced"]["reward_matrix"])[:, -50:], axis=1)

        # Contrastive QR-DQN vs QR-DQN
        u1, p1 = stats.mannwhitneyu(c_vals, qr_vals, alternative='two-sided')
        d1 = (np.mean(c_vals) - np.mean(qr_vals)) / np.sqrt((np.var(c_vals) + np.var(qr_vals)) / 2 + 1e-8)

        # Contrastive QR-DQN vs Enhanced DQN
        u2, p2 = stats.mannwhitneyu(c_vals, enh_vals, alternative='two-sided')
        d2 = (np.mean(c_vals) - np.mean(enh_vals)) / np.sqrt((np.var(c_vals) + np.var(enh_vals)) / 2 + 1e-8)

        summary[env_name]["tests"] = {
            "contrastive_vs_qrdqn": {"U": float(u1), "p_val": float(p1), "cohens_d": float(d1)},
            "contrastive_vs_enhanced": {"U": float(u2), "p_val": float(p2), "cohens_d": float(d2)},
        }

        print(f"\n  [Hypothesis Test 1] Contrastive QR-DQN vs QR-DQN: U={u1:.1f}, p={p1:.4f}, Cohen's d={d1:.2f}")
        print(f"  [Hypothesis Test 2] Contrastive QR-DQN vs Enhanced: U={u2:.1f}, p={p2:.4f}, Cohen's d={d2:.2f}")

        # Format LaTeX rows
        for ag_key in ["enhanced", "qr_dqn", "contrastive_qr_dqn"]:
            s = summary[env_name][ag_key]
            bold_start = r"\textbf{" if ag_key == "contrastive_qr_dqn" else ""
            bold_end = "}" if ag_key == "contrastive_qr_dqn" else ""
            latex_rows.append(
                f"{env_name} & {AGENT_NAMES[ag_key].split(' (')[0]} & "
                f"{bold_start}{s['final_50_mean']:.1f} \\pm {s['final_50_std']:.1f}{bold_end} & "
                f"{s['final_50_median']:.1f} ({s['final_50_iqr']:.1f}) & "
                f"{bold_start}{s['auc_mean']:.1f} \\pm {s['auc_std']:.1f}{bold_end} \\\\"
            )

    # Save summary JSON
    with open(os.path.join(out_dir, "statistical_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save LaTeX table
    tex = (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Iteration 2 Benchmark: Self-Supervised Contrastive Representation Learning (CURL + QR-DQN) vs Baselines across $N=5$ Seeds. (Mean $\pm$ Std and Median (IQR) reported).}" "\n"
        r"\label{tab:iteration2_results}" "\n"
        r"\begin{tabular}{llccc}" "\n"
        r"\toprule" "\n"
        r"Environment & Algorithm & Final 50-Ep Return & Median (IQR) & Sample Efficiency (AUC) \\" "\n"
        r"\midrule" "\n" +
        "\n".join(latex_rows) + "\n" +
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table*}" "\n"
    )
    with open(os.path.join(out_dir, "statistical_table_iteration2.tex"), "w") as f:
        f.write(tex)
    print(f"\n[+] Saved LaTeX table to: {os.path.join(out_dir, 'statistical_table_iteration2.tex')}")
    return summary


def main():
    data = load_data()
    plot_learning_curves(data)
    plot_auc_sample_efficiency(data)
    perform_statistical_analysis(data)


if __name__ == "__main__":
    main()
