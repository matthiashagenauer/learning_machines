import cv2
import os

import numpy as np

import random

import pickle # Import the pickle module
from .DeepQ import *
from .ClassicQ import *
from .utils import *




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

"""
# if we want to avoid pytorch
def run_training(rob):
    pass
def apply_policy(rob):
    pass
"""
    
# Controller instance
def controller(rob: IRobobo):
    while True:
        choice = input("Choose mode [training / apply_classic / apply / classic]: ").strip().lower()
        if choice == "training":
            run_training(rob)
            break
        elif choice == "apply_classic":
            apply_classic_policy(rob)
            break
        elif "apply" in choice:
            if " " in choice:
                args = choice.split()
                set_thresholds(int(args[1]), int(args[2]))
            
                #global THRESHOLD
                #THRESHOLD = int(args[1])
            apply_policy(rob)
            break
        elif choice == "classic":
            run_classic_q_learning(rob)
            break
        else:
            print("Invalid choice. Try again.")
