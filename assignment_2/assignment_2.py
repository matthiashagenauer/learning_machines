import cv2
import os

import numpy as np

import random

import pickle # Import the pickle module
#from .DeepQ import *
#from .ClassicQ import *
from .utils import *
from .coppelia_env import *




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
    #coppelia_main()
