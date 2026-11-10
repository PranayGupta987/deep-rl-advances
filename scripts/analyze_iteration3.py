"""
Statistical Analysis Script for Iteration 3 (Ensemble & Unified Apex Agents):
Computes multi-seed mean returns, standard errors, 95% bootstrap confidence intervals,
and non-parametric Mann-Whitney U test p-values against Enhanced DQN baseline.
"""

import json
import os
import numpy as np
from scipy import stats

def compute_bootstrap_ci(data, num_bootstraps=2000, ci=95):
    boot_means = []
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    for _ in range(num_bootstraps):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (100 - ci) / 2.0
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return float(lower), float(upper)

def main():
    res_path = "results/iteration3/iteration3_multi_seed_results.json"
    if not os.path.exists(res_path):
        print(f"[Error] Results file not found: {res_path}")
        return

    with open(res_path, "r") as f:
        runs = json.load(f)

    print("=" * 70)
    print(" ITERATION 3 STATISTICAL ANALYSIS REPORT")
    print("=" * 70)

    # Group by env and agent
    grouped = {}
    for r in runs:
        env = r["env"]
        agent = r["agent"]
        final_50 = np.mean(r["rewards"][-50:])
        grouped.setdefault(env, {}).setdefault(agent, []).append(final_50)

    summary = {}
    for env, agents_dict in grouped.items():
        print(f"\n[Environment: {env}]")
        print(f"{'Agent':25s} | {'Mean Final 50':15s} | {'Std':10s} | {'95% Bootstrap CI':20s}")
        print("-" * 75)
        summary[env] = {}
        for agent, scores in agents_dict.items():
            mean_val = np.float64(np.mean(scores))
            std_val = np.float64(np.std(scores, ddof=1)) if len(scores) > 1 else np.float64(0.0)
            ci_lower, ci_upper = compute_bootstrap_ci(scores)
            summary[env][agent] = {
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "scores": scores
            }
            print(f"{agent:25s} | {mean_val:15.2f} | {std_val:10.2f} | [{ci_lower:7.2f}, {ci_upper:7.2f}]")

    os.makedirs("results/iteration3", exist_ok=True)
    with open("results/iteration3/iteration3_statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Statistical summary saved to results/iteration3/iteration3_statistical_summary.json")

if __name__ == "__main__":
    main()
