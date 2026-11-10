"""
High-Performance Parallel Multi-Seed Benchmark for Iteration 1.

Features:
- Checkpointing after every completed run (resumable)
- Single-thread enforcement per process to prevent OpenMP contention
- Comprehensive multi-seed evaluation across 5 seeds and 5 agents on 2 environments
"""

import sys
import os
import time
import json
import concurrent.futures
import numpy as np
import torch

torch.set_num_threads(1)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.experiment_worker import execute_seed_task


SEEDS = [42, 100, 2013, 2024, 777]
ENVS  = ["CartPole-v1", "Acrobot-v1"]
AGENTS = ["vanilla", "enhanced", "qr_dqn", "qr_dqn_risk_averse", "qr_dqn_risk_seeking"]
N_EP  = 200
MAX_STEPS = 500


def to_serialisable(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [to_serialisable(v) for v in obj]
    return obj


def main():
    os.makedirs("results/iteration1", exist_ok=True)
    t_start = time.time()

    ckpt_path = "results/iteration1/checkpoint_runs.json"
    all_runs = []
    completed_keys = set()
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                all_runs = json.load(f)
            for r in all_runs:
                completed_keys.add((r["agent"], r["env"], r["seed"]))
            print(f"[*] Resuming from checkpoint: {len(all_runs)} runs already completed.", flush=True)
        except Exception as e:
            print(f"[!] Warning reading checkpoint: {e}", flush=True)
            all_runs = []

    tasks = []
    for env_name in ENVS:
        for agent_type in AGENTS:
            for seed in SEEDS:
                if (agent_type, env_name, seed) not in completed_keys:
                    tasks.append((agent_type, env_name, seed, N_EP, MAX_STEPS))

    total_tasks = len(ENVS) * len(AGENTS) * len(SEEDS)
    completed = len(all_runs)
    num_workers = min(8, os.cpu_count() or 4)

    print("============================================================")
    print(" Starting Iteration 1 Parallel Benchmark")
    print(f" Total Runs: {total_tasks} | Remaining: {len(tasks)} | Parallel Workers: {num_workers}")
    print(f" Environments: {ENVS}")
    print(f" Agents: {AGENTS}")
    print(f" Seeds: {SEEDS} | Episodes: {N_EP}")
    print("============================================================\n", flush=True)

    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(execute_seed_task, t): t for t in tasks}

            for future in concurrent.futures.as_completed(futures):
                task_info = futures[future]
                agent_type, env_name, seed, _, _ = task_info
                try:
                    result = future.result()
                    all_runs.append(result)
                    completed += 1
                    avg50 = np.mean(result["rewards"][-50:])
                    elapsed = time.time() - t_start
                    print(
                        f"[{completed:2d}/{total_tasks:2d}] "
                        f"{env_name:12s} | {agent_type:<22s} | seed={seed:4d} | "
                        f"Final50={avg50:7.2f} | {result['time']:5.1f}s | "
                        f"Elapsed: {elapsed/60:.1f}m",
                        flush=True,
                    )
                    # Checkpoint immediately
                    with open(ckpt_path, "w") as f:
                        json.dump([to_serialisable(r) for r in all_runs], f)
                except Exception as e:
                    print(f"[ERROR] Task {task_info} failed: {e}", flush=True)

    # Reconstruct nested results structure
    all_results = {}
    for env_name in ENVS:
        all_results[env_name] = {}
        for agent_type in AGENTS:
            matching_runs = [
                r for r in all_runs
                if r["env"] == env_name and r["agent"] == agent_type
            ]
            matching_runs.sort(key=lambda r: SEEDS.index(r["seed"]))

            reward_matrix = np.array([r["rewards"] for r in matching_runs])
            loss_matrix = np.array([r["losses"] for r in matching_runs])

            all_results[env_name][agent_type] = {
                "runs": matching_runs,
                "reward_matrix": reward_matrix.tolist(),
                "loss_matrix": loss_matrix.tolist(),
                "agent": agent_type,
                "env": env_name,
                "seeds": [r["seed"] for r in matching_runs],
            }

    out_path = "results/iteration1/multi_seed_results.json"
    with open(out_path, "w") as f:
        json.dump(to_serialisable(all_results), f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[SUCCESS] Completed all {completed}/{total_tasks} runs in {total_time/60:.2f} minutes.")
    print(f"Results saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
