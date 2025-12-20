import numpy as np
import gymnasium as gym
from collections import deque

import rl_env 

def run_frozen_lake(episodes=500000, max_steps=200, window_size=100, target_success_rate=1.0):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    n_states = env.observation_space.n # Get the number of states
    n_actions = env.action_space.n # Get the number of actions

    q = rl_env.QLearning(n_actions, n_states, 0.1, 0.9)  # smaller alpha for stability on slippery
    eg = rl_env.epsilonGreedy(0.9999, 0.0)  # slow epsilon decay: epsilon = 0.999^episode

    success_history = deque(maxlen=window_size)

    for ep in range(episodes):
        state, _ = env.reset()
        episode_success = 0
        for step in range(max_steps):
            action_space = np.arange(n_actions, dtype=float)
            action = eg.choose_action(action_space, state, q, ep)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            q.update(state, int(action), reward, next_state)
            state = next_state
            if terminated or truncated:
                episode_success = int(reward > 0)
                break
        success_history.append(episode_success)

        if (ep + 1) % window_size == 0:
            if success_history:
                window_rate = sum(success_history) / len(success_history) * 100
            else:
                window_rate = 0.0
            print(f"Episode {ep+1}/{episodes} Success rate (last {window_size}): {window_rate:.1f}%")
            if window_rate >= target_success_rate * 100:
                print(f"Reached {target_success_rate * 100:.1f}% success over the last {window_size} episodes, stopping early.")
                break
    env.close()
    q_table = q.get_q_table()
    print(q_table)
    evaluate_policy(q)


def evaluate_policy(q, episodes=100, max_steps=200):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    q_table = np.array(q.get_q_table())
    successes = 0

    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                successes += int(reward > 0)
                break
    env.close()

    success_rate = successes / episodes * 100
    print(f"Evaluation success rate over {episodes} episodes: {success_rate:.1f}%")

if __name__ == '__main__':
    run_frozen_lake()
