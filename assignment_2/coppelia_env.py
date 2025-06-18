import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .utils import *
from stable_baselines3.common.env_checker import check_env


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


class Coppelia_env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}


    def __init__(self, rob: IRobobo, grid_size=10, render_mode="console", deepQ = False):
        super(Coppelia_env, self).__init__()
        """
        self.render_mode = render_mode

        # Size of the 1D-grid
        self.grid_size = grid_size
        # Initialize the agent at the right of the grid
        self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(1,), dtype=np.float32
        )
        """
        self.rob = rob
        #self.rob.set_phone_pan()
        #self.rob.set_phone_tilt()
        
        self.observation_space = spaces.Box(low=0, high=2, shape=(4,), dtype=np.int8)   
        self.deepQ = deepQ

        self.action_space = spaces.Box(low=-180, high=180, shape=(), dtype=np.float32)
        if isinstance(self.rob, SimulationRobobo):
            self.reset_position = self.rob.get_position()
            self.reset_orientation = self.rob.get_orientation()


    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()
        #else:
        #    return np.array([0, 0, 0]).astype(np.float32), {} 
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([0, 0, 0, 0]).astype(np.int8), {}  # empty info dict

    def step(self, action):
        """
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        """
        if self.deepQ:
            action = translate_action(action)
        TURN_360 = 4750
        try:
            if action < 0:
                turn_time = (TURN_360 / 360) * (-1) * action
                self.rob.move(-20, 20, int(turn_time)) # for continuous action space turn
            else:
                turn_time = (TURN_360 / 360) * action
                self.rob.move(20, -20, int(turn_time)) # for continuous action space turn
            self.rob.move(100, 100, 250) # then move
        except Exception as e:
            print(e)
        #print(turn_time)
        
        next_state = get_state(self.rob)

        terminated = self.rob.get_nr_food_collected() >= 7
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = compute_reward(next_state=next_state, action=action)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(next_state).astype(np.int8),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        """
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))
        """
        pass

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()

    def get_robot_state(self):
        return get_state(self.rob)
    
    def get_food_collected(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.sleep(.2)
            return self.rob.get_nr_food_collected()


def coppelia_main(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    env = Coppelia_env(rob)
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()