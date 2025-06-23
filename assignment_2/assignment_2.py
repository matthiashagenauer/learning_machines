import cv2
import os

import numpy as np

import random

import pickle # Import the pickle module
#from .DeepQ import *
#from .ClassicQ import *
from .utils import *
from .coppelia_env import *
from .DeepQ import run_training, apply_policy
from .PPO import train_ppo





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

    
# Controller instance
def controller(rob: IRobobo):
    while True:
        choice = input("Choose mode [deepq / ppo]: ").strip().lower()
        if choice == "deepq":
            run_training(rob)
            break
        elif choice == "ppo":
            train_ppo(rob)
            break
        elif choice == "apply":
            apply_policy(rob)
            break
        else:
            print("Invalid choice. Try again.")