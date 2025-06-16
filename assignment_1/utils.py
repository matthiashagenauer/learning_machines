import cv2
import os

import numpy as np

import random
import torch
#from torch import nn
#import torch.nn.functional as F
#from torch import optim
#from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm as _tqdm
import pickle # Import the pickle 

from typing import List, Tuple


from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    Position,
    Orientation,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


LOWER_THRESHOLD = 25 # for deepQ
HIGHER_THRESHOLD = 35 # for deepQ
THRESHOLD = 40

def teleport(rob, reset_pos, reset_orient):
    #reset_orient = Orientation(1.5239092710256086, 1.2178260440512918, 1.6365592454553204)
    #reset_pos = Position(2.400886486231184, 1.8463276511607962, 0.04071442365134694)
    rob.set_position(reset_pos, reset_orient)

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


def obs_to_key(obs):
        return tuple(obs)  # Convert array to tuple for dict key

def validate(rob:IRobobo, classic=False, q_table=None, q_network=None):
    total_reward = 0
    if classic:
        for i in range(100):
            state = obs_to_key(sensor_to_vec(get_sensor_data(rob)))
            action = np.argmax(q_table[state])
            _, reward = take_action(rob, action)
            total_reward += reward
    else:
        for i in range(100):
            state = np.array(sensor_to_vec(get_sensor_data(rob)))
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  
            with torch.no_grad():
                q_values = q_network(state) 
                action = torch.argmax(q_values).item()  
            state, reward = take_action(rob, action)
            total_reward += reward
            
    return total_reward


def set_thresholds(lower, higher):
    global LOWER_THRESHOLD, HIGHER_THRESHOLD
    LOWER_THRESHOLD = lower
    HIGHER_THRESHOLD = higher

def sensor_to_vec(sensor_data):
    sensor_data = np.asarray(sensor_data)

    result = np.select(
        [sensor_data < LOWER_THRESHOLD, (sensor_data >= LOWER_THRESHOLD) & (sensor_data < HIGHER_THRESHOLD), sensor_data >= HIGHER_THRESHOLD],
        [0, 1, 2]
    )

    #print("Sensor Data:")
    #print(sensor_data)
    #print("After Transition:")
    #print(result)

    return result

"""

def sensor_to_vec(sensor_data, threshold=40):

    sensor_data = np.asarray(sensor_data)
    result = (sensor_data >= threshold).astype(int)
    return result
"""


def get_epsilon(iters_left, total_iters=1000, epsilon_start=1.0, epsilon_final=0.05):
    # k determines how fast we decay
    decay_rate = 5.0  # Higher means faster decay
    
    fraction = iters_left / total_iters
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-decay_rate * (1 - fraction))
    
    return epsilon


def take_action(rob, action):
    move_duration = 150
    # straight
    if action == 0:
        rob.move_blocking(100, 100, move_duration)
    # turn around/backwards / full 180
    elif action == 1:
        rob.move_blocking(20, -20, 2375)
        rob.move_blocking(100, 100, move_duration)
    # left or right  / 90
    elif action == 2:
        rob.move_blocking(20, -20, 1187)
        rob.move_blocking(100, 100, move_duration)
    # left or  right / 90
    elif action == 3:
        rob.move_blocking(-20, 20, 1187)
        rob.move_blocking(100, 100, move_duration)
    # slight left or right / 45
    elif action == 4:
        rob.move_blocking(20, -20, 593)
        rob.move_blocking(100, 100, move_duration)
    # slight left or right / 45
    elif action == 5:
        rob.move_blocking(-20, 20, 593)
        rob.move_blocking(100, 100, move_duration)
    # bit further left or right / 135
    elif action == 6:
        rob.move_blocking(20, -20, 1780)
        rob.move_blocking(100, 100, move_duration)
    # bit further left or right / 135
    elif action == 7:
        rob.move_blocking(20, -20, 1780)
        rob.move_blocking(100, 100, move_duration)
    next_state = sensor_to_vec(get_sensor_data(rob))
    reward = compute_reward(next_state=next_state, action=action)

    return next_state, reward

def compute_reward(next_state, action, prev_state = None, ):
    """
    Reward function for obstacle avoidance.
    More positive reward for keeping a safe distance,
    and negative if the robot is getting too close to obstacles.
    """
    """
    # Encourage fewer "danger" readings
    danger_penalty = np.sum(next_state == 2) * -30
    warning_penalty = np.sum(next_state == 1) * -3
    clear_bonus = np.sum(next_state == 0) * 0

    # Encourage going straight (actions 0 = forward, 2/3 = slight turns)
    forward_bonus = 5 if action == 0 else 0

    # Penalty for spinning (actions 6 and 7 are large turns)
    spin_penalty = -10 if action not in [0] else 0

    total_reward = danger_penalty + clear_bonus + forward_bonus + spin_penalty + warning_penalty
    """
    if next_state[-1] == 1:
        move_towards_target_reward = 1 if (action < 10 or action > -10) else 0
        proximity_reward = next_state[1] # only front sensor
        total_reward = move_towards_target_reward + proximity_reward
    else:
        danger_penalty = np.sum(next_state[:-1] == 2) * -30
        warning_penalty = np.sum(next_state == 1) * -3
        not_looking_at_green_penalty = -10
        total_reward = danger_penalty + warning_penalty + not_looking_at_green_penalty
    
    return float(total_reward)

def get_sensor_data(rob: IRobobo):
    sensor_data = rob.read_irs()
    if isinstance(rob, SimulationRobobo):
        return [sensor_data[i] for i in [7, 4, 5]]
    else:
        return [sensor_data[i] for i in [5, 4, 7]]
    

def detect_green_blocks(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect green blocks in the image and return their bounding boxes.
    
    Args:
        image: Input image in BGR format (from `read_image_front`).
    
    Returns:
        List of (x, y, width, height) bounding boxes for green blocks.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range in HSV (adjust these values)
    lower_green = np.array([35, 50, 50])   # Lower bound for green
    upper_green = np.array([85, 255, 255]) # Upper bound for green
    
    # Create a mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Optional: Remove noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours and return bounding boxes
    min_area = 100  # Minimum area to consider as a block (adjust as needed)
    green_blocks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            green_blocks.append((x, y, w, h))
    
    return green_blocks

def get_state(rob):
    # Get sensor data and convert to vector
    sensor_array = sensor_to_vec(get_sensor_data(rob))
    
    # Detect green blocks - convert to boolean (1 if any blocks found, 0 otherwise)
    green_blocks = detect_green_blocks(rob.read_image_front())
    has_green_block = int(len(green_blocks) > 0)  # 1 if any blocks detected, else 0
    
    # Combine into single state array
    next_state = np.append(sensor_array, has_green_block)
    
    return next_state
    