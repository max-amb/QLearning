import numpy as np
import gymnasium as gym
import copy
import random
from collections import deque

import rl_env 

LEARNING_RATE = 100.0
DISCOUNT_REWARD = 0.5
EPSILON_POWER = 0.995
EPSILON_FLOOR = 0.0
MCS_ITERATION_LIMIT = 100

env = gym.make('FrozenLake-v1', is_slippery=True)

def run_frozen_lake(episodes=500000, max_steps=100, window_size=100, target_success_rate=1.0):
    n_states = env.observation_space.n # Get the number of states
    n_actions = env.action_space.n # Get the number of actions
    action_space = np.arange(n_actions, dtype=int) # Returns [0, 1, ..., n_actions-1]

    q = rl_env.QLearning(n_actions, n_states, LEARNING_RATE, DISCOUNT_REWARD)
    eg = rl_env.epsilonGreedy(EPSILON_POWER, EPSILON_FLOOR)
    mcs = rl_env.MonteCarloSearch(MCS_ITERATION_LIMIT, rolloutReward)

    success_history = deque(maxlen=window_size)

    for ep in range(episodes):
        state, _ = env.reset()
        episode_success = 0
        for _ in range(max_steps):
            action = eg.choose_action(action_space, state, q, mcs, ep)
            next_state, reward, terminated, truncated, _ = env.step(action)
            q.update(state, int(action), reward, next_state)
            state = next_state
            if terminated or truncated:
                episode_success = int(float(reward) > 0)
                break
        success_history.append(episode_success)

        if (ep + 1) % window_size == 0:
            if success_history:
                window_rate = sum(success_history) / len(success_history) * 100
            else:
                window_rate = 0.0
            print(f"Episode {ep+1}/{episodes} Success rate (last {window_size}): {window_rate:.1f}%, Epsilon {eg.get_epsilon()}")
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
                successes += int(float(reward) > 0) # If succesfull reward > 0 => True which converts to 1
                break
    env.close()

    success_rate = successes / episodes * 100
    print(f"Evaluation success rate over {episodes} episodes: {success_rate:.1f}%")

def rolloutReward(currentState, currentAction): # Current state action
    saveState = copy.deepcopy(env)
    number_of_actions = env.action_space.n # Get the number of actions
    _, reward, terminated, truncated, _ = saveState.step(currentAction)

    steps = 1
    while not (terminated or truncated):
        _, new_reward, terminated, truncated, _ = saveState.step(random.randrange(0, number_of_actions))
        reward = (float(new_reward)*(DISCOUNT_REWARD ** steps)) + float(reward)
        steps += 1;

    return reward

if __name__ == '__main__':
    run_frozen_lake()
