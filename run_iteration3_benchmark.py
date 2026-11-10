"""
Iteration 3 Parallel Benchmark Runner:
Evaluates Ensemble Bootstrapped DQN and Full Unified Apex Agent
across 5 seeds on CartPole-v1 and Acrobot-v1 with checkpointing.
"""

import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.experiment_worker import execute_seed_task

def to_serialisable(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [to_serialisable(v) for v in obj]
    return obj

def main():
    os.makedirs("results/iteration3", exist_ok=True)
    ckpt_path = "results/iteration3/checkpoint_runs.json"
    final_path = "results/iteration3/iteration3_multi_seed_results.json"

    envs = ["CartPole-v1", "Acrobot-v1"]
    agents = ["ensemble_dqn", "full_unified"]
    seeds = [42, 100, 2013, 2024, 777]
    n_episodes = 200
    max_steps = 500

    tasks = []
    for env in envs:
        for agent in agents:
            for seed in seeds:
                tasks.append((agent, env, seed, n_episodes, max_steps))

    total_runs = len(tasks)
    print("=" * 60, flush=True)
    print(f" Starting Iteration 3 Benchmark: Ensemble & Unified Apex", flush=True)
    print(f" Total Runs: {total_runs} | Parallel Workers: 4", flush=True)
    print(f" Environments: {envs}", flush=True)
    print(f" Agents: {agents}", flush=True)
    print(f" Seeds: {seeds} | Episodes: {n_episodes}", flush=True)
    print("=" * 60, flush=True)

    results = []
    completed_keys = set()
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                saved = json.load(f)
                for r in saved:
                    key = (r["agent"], r["env"], r["seed"])
                    completed_keys.add(key)
                    results.append(r)
            print(f"[Resume] Loaded {len(completed_keys)} completed runs from {ckpt_path}", flush=True)
        except Exception as e:
            print(f"[Warning] Could not load checkpoint: {e}", flush=True)

    remaining_tasks = [t for t in tasks if (t[0], t[1], t[2]) not in completed_keys]
    print(f"Remaining runs to execute: {len(remaining_tasks)} / {total_runs}", flush=True)

    t0 = time.time()
    if remaining_tasks:
        with Pool(processes=4) as pool:
            for i, res in enumerate(pool.imap_unordered(execute_seed_task, remaining_tasks), 1):
                elapsed = res["time"]
                final_50 = float(np.mean(res["rewards"][-50:]))
                print(f"[{len(results)+1:2d}/{total_runs}] {res['env']:12s} | {res['agent']:20s} | seed={res['seed']:4d} | Final50={final_50:8.2f} | {elapsed:6.1f}s", flush=True)
                results.append(to_serialisable(res))
                with open(ckpt_path, "w") as f:
                    json.dump(results, f, indent=2)

    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)

    total_elapsed = (time.time() - t0) / 60.0
    print("\n" + "=" * 60, flush=True)
    print(f" Iteration 3 Benchmark Completed Successfully in {total_elapsed:.1f}m!", flush=True)
    print(f" Results saved to: {final_path}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
