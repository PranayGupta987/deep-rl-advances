"""
Generate all comparison plots from training metrics.
Usage: python generate_plots.py
Reads results/metrics.json, writes PNG figures into results/.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── styling ──────────────────────────────────────────────────────────── #
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

COLORS = {
    "vanilla":  "#e74c3c",   # red
    "enhanced": "#2ecc71",   # green
}
LABELS = {
    "vanilla":  "Vanilla DQN",
    "enhanced": "Enhanced DQN\n(Double + PER + Soft)",
}


def smooth(data, window=20):
    """Simple moving average for cleaner curves."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_metrics(path="results/metrics.json"):
    with open(path) as f:
        return json.load(f)


# ── plot 1: learning curves (episode reward) ─────────────────────────── #
def plot_learning_curves(data, out_dir="results"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (env_name, agents) in enumerate(data.items()):
        ax = axes[idx]
        for agent_tag in ["vanilla", "enhanced"]:
            rewards = agents[agent_tag]["rewards"]
            raw = np.array(rewards)
            smoothed = smooth(raw, 20)
            x_raw = np.arange(len(raw))
            x_sm  = np.arange(len(smoothed)) + 10  # center the moving avg
            ax.plot(x_raw, raw, alpha=0.15, color=COLORS[agent_tag])
            ax.plot(x_sm, smoothed, linewidth=2, color=COLORS[agent_tag],
                    label=LABELS[agent_tag])
        ax.set_title(env_name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Learning Curves: Vanilla DQN vs. Enhanced DQN", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved learning_curves.png")


# ── plot 2: training loss ─────────────────────────────────────────────── #
def plot_loss_curves(data, out_dir="results"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (env_name, agents) in enumerate(data.items()):
        ax = axes[idx]
        for agent_tag in ["vanilla", "enhanced"]:
            losses = agents[agent_tag]["losses"]
            raw = np.array(losses)
            smoothed = smooth(raw, 20)
            x_sm = np.arange(len(smoothed)) + 10
            ax.plot(x_sm, smoothed, linewidth=2, color=COLORS[agent_tag],
                    label=LABELS[agent_tag])
        ax.set_title(env_name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Training Loss (smoothed)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Training Loss: Vanilla DQN vs. Enhanced DQN", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved loss_curves.png")


# ── plot 3: bar chart – final performance ─────────────────────────────── #
def plot_final_reward_bar(data, out_dir="results"):
    envs = list(data.keys())
    vanilla_avgs = []
    enhanced_avgs = []
    for env_name in envs:
        vanilla_avgs.append(np.mean(data[env_name]["vanilla"]["rewards"][-50:]))
        enhanced_avgs.append(np.mean(data[env_name]["enhanced"]["rewards"][-50:]))

    x = np.arange(len(envs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, vanilla_avgs, width, label="Vanilla DQN",
                color=COLORS["vanilla"], edgecolor='white')
    b2 = ax.bar(x + width/2, enhanced_avgs, width, label="Enhanced DQN",
                color=COLORS["enhanced"], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Average Reward (last 50 episodes)")
    ax.set_title("Final Performance Comparison", fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # annotate bars
    for b in b1:
        ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for b in b2:
        ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_reward_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved final_reward_comparison.png")


# ── plot 4: episode length over training ──────────────────────────────── #
def plot_episode_length(data, out_dir="results"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (env_name, agents) in enumerate(data.items()):
        ax = axes[idx]
        for agent_tag in ["vanilla", "enhanced"]:
            steps = agents[agent_tag]["steps"]
            raw = np.array(steps, dtype=float)
            smoothed = smooth(raw, 20)
            x_sm = np.arange(len(smoothed)) + 10
            ax.plot(x_sm, smoothed, linewidth=2, color=COLORS[agent_tag],
                    label=LABELS[agent_tag])
        ax.set_title(env_name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Length (steps)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Episode Length Over Training", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "episode_length.png"), dpi=150)
    plt.close(fig)
    print("Saved episode_length.png")


# ── plot 5: cumulative reward ─────────────────────────────────────────── #
def plot_cumulative_reward(data, out_dir="results"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (env_name, agents) in enumerate(data.items()):
        ax = axes[idx]
        for agent_tag in ["vanilla", "enhanced"]:
            rewards = agents[agent_tag]["rewards"]
            cum = np.cumsum(rewards)
            ax.plot(cum, linewidth=2, color=COLORS[agent_tag],
                    label=LABELS[agent_tag])
        ax.set_title(env_name)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Cumulative Reward Over Training", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_reward.png"), dpi=150)
    plt.close(fig)
    print("Saved cumulative_reward.png")


# ── summary table ────────────────────────────────────────────────────── #
def print_summary(data):
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for env_name, agents in data.items():
        print(f"\n  Environment: {env_name}")
        print(f"  {'Metric':<30s} {'Vanilla DQN':>15s} {'Enhanced DQN':>15s} {'Improvement':>15s}")
        print("  " + "-"*75)
        for metric_name, key, last_n in [
            ("Final Avg Reward (last 50)", "rewards", 50),
            ("Final Avg Reward (last 20)", "rewards", 20),
        ]:
            v = np.mean(agents["vanilla"][key][-last_n:])
            e = np.mean(agents["enhanced"][key][-last_n:])
            if v != 0:
                imp = ((e - v) / abs(v)) * 100
                imp_str = f"{imp:+.1f}%"
            else:
                imp_str = "N/A"
            print(f"  {metric_name:<30s} {v:>15.1f} {e:>15.1f} {imp_str:>15s}")

        # Training time
        vt = agents["vanilla"]["time"]
        et = agents["enhanced"]["time"]
        print(f"  {'Training Time (s)':<30s} {vt:>15.1f} {et:>15.1f}")
    print()


# ──────────────────────────── main ───────────────────────────────────── #
if __name__ == "__main__":
    data = load_metrics()
    plot_learning_curves(data)
    plot_loss_curves(data)
    plot_final_reward_bar(data)
    plot_episode_length(data)
    plot_cumulative_reward(data)
    print_summary(data)
    print("All plots generated in results/")
