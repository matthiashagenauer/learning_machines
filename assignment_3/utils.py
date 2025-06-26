import cv2
import os
import time

import numpy as np

from tqdm import tqdm as _tqdm

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
HIGHER_THRESHOLD = 45 # for deepQ
#THRESHOLD = 40

######################## BEGINING Deep Q stuff #########################

def get_epsilon(iters_left, total_iters=1000, epsilon_start=1.0, epsilon_final=0.05):
    # k determines how fast we decay
    decay_rate = 5.0  # Higher means faster decay
    
    fraction = iters_left / total_iters
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-decay_rate * (1 - fraction))
    
    return epsilon


def translate_action(action_idx):

    # straight
    if action_idx == 0:
        return 0
    # turn around/backwards / full 180
    elif action_idx == 1:
        return 20
    # left or right  / 90
    elif action_idx == 2:
        return 40
    # left or  right / 90
    elif action_idx == 3:
        return 60
    # slight left or right / 45
    elif action_idx == 4:
        return -20
    # slight left or right / 45
    elif action_idx == 5:
        return -40
    # bit further left or right / 135
    elif action_idx == 6:
        return -60
    # bit further left or right / 135


######################## END Deep Q stuff #########################



def teleport(rob, reset_pos, reset_orient):
    #reset_orient = Orientation(1.5239092710256086, 1.2178260440512918, 1.6365592454553204)
    #reset_pos = Position(2.400886486231184, 1.8463276511607962, 0.04071442365134694)
    rob.set_position(reset_pos, reset_orient)

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

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

SMALLEST_DISTANCE = np.inf

def compute_reward(next_state, action, detect_red_middle=None, red_in_arms=None, prev_state = None, block_collection=True, distance_to_base=None):
    """
    Reward function for obstacle avoidance.
    More positive reward for keeping a safe distance,
    and negative if the robot is getting too close to obstacles.
    """
    global SMALLEST_DISTANCE
    total_reward = 0
    if block_collection:
        
        forward_bonus = 0
        if detect_red_middle:
            total_reward = 10
            if action == 0:
                forward_bonus = 50

        if red_in_arms:
            total_reward = 1000
        total_reward = total_reward + forward_bonus
    else:
        if not red_in_arms:
            total_reward = total_reward - 500
        
        if (distance_to_base > SMALLEST_DISTANCE) and red_in_arms:
            SMALLEST_DISTANCE = distance_to_base
            total_reward += 10

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

    # save image - not needed - got some already
    #save_dir = "/root/results/"
    #timestamp = int(time.time())
    #cv2.imwrite(f"{save_dir}/{timestamp}_raw.jpg", image)
    
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

def green_area_percentage(image: np.ndarray) -> float:
    """
    Calculate the percentage of the image occupied by green pixels.

    Args:
        image: Input image in BGR format (e.g., from `read_image_front`).

    Returns:
        A float representing the percentage of green area in the image.
    """
    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create a binary mask where green pixels are white
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Count green pixels (non-zero values in mask)
    green_pixels = cv2.countNonZero(mask)

    # Total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate percentage
    green_percentage = (green_pixels / total_pixels) * 100

    return green_percentage


def get_state(rob, block_collection=True):
    # Get sensor data and convert to vector
    sensor_array = sensor_to_vec(get_sensor_data(rob))
    red_in_arms = 0
    red_in_middle = 0
    # Detect green blocks - convert to boolean (1 if any blocks found, 0 otherwise)
    image = rob.read_image_front()

    if is_red_in_arms(image):
        red_in_arms = 1

    next_state = np.append(sensor_array, red_in_arms)

    if block_collection:
        if is_red_in_middle(image):
            red_in_middle = 1
        
        # Combine into single state array
        next_state = np.append(next_state, red_in_middle)
    else:
        green_percentage = green_area_percentage(image)

        next_state = np.append(next_state, green_percentage)

    
    return next_state
    
def is_red_in_lower_middle(image: np.ndarray, save_prefix: str = "output") -> bool:
    """
    Detect if there is any red in the lower-middle part of the image.
    Also saves the original and processed mask images.

    Args:
        image: Input image in BGR format.
        save_prefix: Prefix used for saved image filenames.

    Returns:
        True if red is detected in the lower-middle region, False otherwise.
    """
    time_stamp = time.time()
    # Save original image
   # cv2.imwrite(f"/root/results/{time_stamp}_original.jpg", image)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV (two ranges due to hue wrap-around)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Image dimensions
    h, w, _ = image.shape

    # Define lower-middle region: bottom 1/4 vertically, middle 1/3 horizontally
    start_y = int(h * 0.75)
    end_y = h
    start_x = int(w * 0.33)
    end_x = int(w * 0.66)

    # Crop region of interest (ROI)
    roi = hsv[start_y:end_y, start_x:end_x]

    # Create red masks
    mask1 = cv2.inRange(roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(roi, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Save processed mask image
    #cv2.imwrite(f"/root/results/{time_stamp}_processed_mask.jpg", mask)

    # Check if any red pixels are found
    return np.any(mask > 0)

def is_red_in_middle(image: np.ndarray) -> bool:
    """
    Detect if there is any red in the [3/9, 4/9] horizontal band (middle slice) of the image.

    Args:
        image: Input image in BGR format.

    Returns:
        True if red is detected in the specified middle band, False otherwise.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    time_stamp = time.time()
    # Define red color range in HSV (two parts due to hue wrap-around)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Get image dimensions
    h, w, _ = image.shape

    # Define horizontal band
    start_x = int(w *10/ 21)
    end_x = int(w * 11 / 21)

    # Use full vertical range
    start_y = 0
    end_y = h

    # Crop the region of interest
    roi = hsv[start_y:end_y, start_x:end_x]

    # Create masks for red color
    mask1 = cv2.inRange(roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(roi, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    #cv2.imwrite(f"root/results/{time_stamp}_processed_mask.jpg", mask)

    # Return True if any red pixel is found
    return np.any(mask > 0)


def is_red_in_arms(image: np.ndarray) -> bool:
    """
    Detect if there is any red in the bottom third of the [3/9, 4/9] horizontal slice of the image.

    Args:
        image: Input image in BGR format.

    Returns:
        True if red is detected in the specified bottom-middle region, False otherwise.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV (two parts due to hue wrap-around)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Get image dimensions
    h, w, _ = image.shape

    # Define horizontal band [3/9, 4/9]
    start_x = int(w * 10 / 21)
    end_x = int(w * 11 / 21)

    # Define bottom third vertically
    start_y = int(h * (512-50)/ 512)
    end_y = h

    # Crop the region of interest
    roi = hsv[start_y:end_y, start_x:end_x]

    # Create masks for red color
    mask1 = cv2.inRange(roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    time_stamp = time.time()
   #cv2.imwrite(f"/root/results/{time_stamp}_processed_mask.jpg", red_mask)
   # cv2.imwrite(f"/root/results/{time_stamp}_original.jpg", image)

    # Return True if any red pixel is found
    return np.any(red_mask > 0)

