import cv2
import os

import numpy as np

import random

from .utils import *
from tqdm import tqdm as _tqdm
import pickle 




from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)




def run_classic_q_learning(rob: IRobobo, iterations=2000, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_final=0.05):
    """
    Classic Q-learning using a tabular Q-table.
    Assumes the observation space (sensor_to_vec) can be encoded as a string or tuple key.
    """
    from collections import defaultdict

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    q_table = defaultdict(lambda: np.zeros(8))  # 8 possible actions

    validation_reward = []
    reset_position = rob.get_position()
    reset_orientation = rob.get_orientation()

    for i in tqdm(range(iterations), desc="Classic Q-Learning"):
        if i % 100 == 0:
            teleport(rob, reset_position, reset_orientation)
        if i % int(iterations / 20) == 0:
            teleport(rob, reset_position, reset_orientation)
            validation_reward.append(validate(rob, classic=True, q_table=q_table))
            teleport(rob, reset_position, reset_orientation)

        state = obs_to_key(sensor_to_vec(get_sensor_data(rob)))

        epsilon = get_epsilon(iters_left=iterations - i, total_iters=iterations,
                              epsilon_start=epsilon_start, epsilon_final=epsilon_final)

        if random.random() < epsilon:
            action = random.randint(0, 7)
        else:
            action = np.argmax(q_table[state])

        next_obs, reward = take_action(rob, action)
        next_state = obs_to_key(next_obs)

        # Q-learning update rule
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

    # Save Q-table
    with open('/root/results/classic_q_table.pkl', 'wb') as f:
        pickle.dump(dict(q_table), f)
    print("Classic Q-table saved to classic_q_table.pkl")

    # Save validation rewards
    with open('/root/results/validation_rewards.pkl', 'wb') as f:
        pickle.dump(validation_reward, f)
    print("Validation rewards saved to validation_rewards.pkl")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()



def apply_classic_policy(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    path = "/root/results/classic_q_table.pkl"

    if not os.path.exists(path):
        print("No Q-table found at", path)
        return

    # Load the Q-table
    with open(path, 'rb') as f:
        q_table = pickle.load(f)
    print("Loaded classic Q-table.")

    def obs_to_key(obs):
        return tuple(obs)  # convert array to tuple to use as dict key

    for i in _tqdm(range(10000000)):
        obs = sensor_to_vec(get_sensor_data(rob))
        key = obs_to_key(obs)

        if key in q_table:
            action = np.argmax(q_table[key])
        else:
            action = random.randint(0, 7)  # fallback if state is unseen

        _, _ = take_action(rob, action)  # take action, ignore reward for now

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
