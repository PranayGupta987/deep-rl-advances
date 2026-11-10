"""
Training script for comparing Vanilla DQN vs. Enhanced DQN across
multiple OpenAI Gymnasium control environments.
"""
import os
import time
import json
import numpy as np
import gymnasium as torch_gym

from src.dqn_agent import DQNAgent
from src.enhanced_agent import EnhancedDQNAgent


def train_agent(agent_type, env_name, n_episodes=300, max_steps=500, seed=42):
    np.random.seed(seed)
    env = torch_gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if agent_type == 'vanilla':
        agent = DQNAgent(state_dim, n_actions, eps_decay_steps=int(n_episodes * max_steps * 0.6))
    else:
        agent = EnhancedDQNAgent(state_dim, n_actions, eps_decay_steps=int(n_episodes * max_steps * 0.6))

    episode_rewards = []
    episode_losses = []
    episode_steps = []

    start_time = time.time()
    total_steps = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_loss = []
        steps = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                ep_loss.append(loss)

            state = next_state
            ep_reward += reward
            steps += 1
            total_steps += 1

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_losses.append(np.mean(ep_loss) if len(ep_loss) > 0 else 0.0)
        episode_steps.append(steps)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            print(f"[{env_name}] [{agent_type.upper()}] Ep {ep+1}/{n_episodes} | Avg Reward (last 50): {avg_r:.1f} | Eps: {agent.eps:.3f}")

    duration = time.time() - start_time
    env.close()

    return {
        "rewards": episode_rewards,
        "losses": episode_losses,
        "steps": episode_steps,
        "duration": duration,
        "total_steps": total_steps
    }


def main():
    environments = ["CartPole-v1", "Acrobot-v1"]
    n_episodes = 250  # sufficient for solid convergence on classic control
    results = {}

    os.makedirs("results", exist_ok=True)

    for env_name in environments:
        print(f"\n==============================")
        print(f"Training on {env_name}")
        print(f"==============================")

        results[env_name] = {}
        for agent_type in ["vanilla", "enhanced"]:
            print(f"\n-> Starting {agent_type} DQN...")
            res = train_agent(agent_type, env_name, n_episodes=n_episodes)
            results[env_name][agent_type] = res
            print(f"Finished {agent_type} in {res['duration']:.2f}s. Final 50-ep avg reward: {np.mean(res['rewards'][-min(50, len(res['rewards'])):]) :.1f}")

    # Save metrics
    with open("results/metrics.json", "w") as f:
        # Convert numpy floats/ints to python native types for JSON serialization
        json_safe = {}
        for env, agents in results.items():
            json_safe[env] = {}
            for ag, data in agents.items():
                json_safe[env][ag] = {
                    "rewards": [float(r) for r in data["rewards"]],
                    "losses": [float(l) for l in data["losses"]],
                    "steps": [int(s) for s in data["steps"]],
                    "duration": float(data["duration"]),
                    "total_steps": int(data["total_steps"])
                }
        json.dump(json_safe, f, indent=2)
    print("\nSaved all metrics to results/metrics.json successfully!")


if __name__ == "__main__":
    main()
