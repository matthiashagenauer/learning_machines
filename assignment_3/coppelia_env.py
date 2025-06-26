import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .utils import *
#from stable_baselines3.common.env_checker import check_env


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


    def reset(self, seed=None, options=None, add_random_perturbation=False):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()
        self.rob.set_phone_tilt(109,100)
        if add_random_perturbation:
            self.rob.move_blocking(60,60, 1000)
            random_integers = np.random.randint(500, 1800, size=3)

            self.rob.move_blocking(20, -20, random_integers[0])
            self.rob.move_blocking(50, 50, random_integers[1])
            self.rob.move_blocking(-20, 20, random_integers[2])


        #else:
        #    return np.array([0, 0, 0]).astype(np.float32), {} 
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([0, 0, 0, 0]).astype(np.int8), {}  # empty info dict

    def step(self, action, block_collection = True):
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
            if action == 0:
                self.rob.move_blocking(105, 100, 250) # then move
            elif action < 0:
                turn_time = (TURN_360 / 360) * (-1) * action
                self.rob.move_blocking(-20, 20, int(turn_time)) # for continuous action space turn
            else:
                turn_time = (TURN_360 / 360) * action
                self.rob.move_blocking(20, -20, int(turn_time)) # for continuous action space turn
            
        except Exception as e:
            print(e)
        
        next_state = get_state(self.rob, block_collection=block_collection)
        #food_collected_after = self.rob.get_nr_food_collected()
        #print(food_collected)
        
        truncated = False  # we do not limit the number of steps here

        image = self.rob.read_image_front()
        
        distance_to_base = self.get_distance_to_base()

        red_in_arms = is_red_in_arms(image)
        reward = 0

        if block_collection:
            detect_red_middle = is_red_in_middle(image)
            terminated = red_in_arms
            reward = compute_reward(next_state=next_state, detect_red_middle=detect_red_middle, red_in_arms=red_in_arms, action=action, block_collection=block_collection)
        else:
            terminated = (not red_in_arms) or (distance_to_base <= 0 and red_in_arms)
            reward = compute_reward(next_state=next_state, red_in_arms=red_in_arms, action=action, block_collection=block_collection, distance_to_base=distance_to_base)


        
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(next_state).astype(np.int8),
            reward,
            terminated,
            truncated,
            info,
        )
    
    def teleport(self, episode, episodes):
        data = [[Orientation(yaw=-1.5680605471049571, pitch=0.8859427569251203, roll=-1.5728157278055066), Position(x=-2.954414489149553, y=-0.010197066939466556, z=0.039775510502622735)],[Orientation(yaw=1.549657091374225, pitch=1.4985915499913811, roll=1.5919674546103197), Position(x=-3.1124296739182897, y=-0.006067762322869174, z=0.03977213855380894)], [Orientation(yaw=1.569279832184099, pitch=-0.1227015533179101, roll=1.5706981205135484), Position(x=-2.5418040956824326, y=0.6303631796844946, z=0.03978579702412296)], [Orientation(yaw=1.5558650304737902, pitch=-1.461930644692143, roll=1.556045822091844), Position(x=-2.4990833978280547, y=0.5988361667980143, z=0.039785167544283)], [Orientation(yaw=1.5564409564033026, pitch=1.4619636052886722, roll=1.5851506925340424), Position(x=-2.588058613029801, y=0.5988525928622892, z=0.03978474608918915)],  [Orientation(yaw=1.5692786375385732, pitch=-0.007752696926199876, roll=1.570880007063897), Position(x=-2.683789766023037, y=1.0682724844481044, z=0.03978536796050946)], [Orientation(yaw=1.5640246136389826, pitch=-1.3451782079478403, roll=1.564281207698291), Position(x=-2.636619905125915, y=1.037013124945882, z=0.03978499886255747)], [Orientation(yaw=1.567962105254265, pitch=0.9938714606594542, roll=1.573264402234514), Position(x=-2.719316190883052, y=1.051396649612378, z=0.03978509604719551)], [Orientation(yaw=1.5691379058064465, pitch=0.4321565045912905, roll=1.5715761080079473), Position(x=-3.399299759906827, y=1.4853870538027008, z=0.039785382163896724)], [Orientation(yaw=-1.569104046571536, pitch=0.47090024488326543, roll=-1.5714777874037436), Position(x=-3.7954756945000945, y=0.7411902518594959, z=0.039784831848629104)], [Orientation(yaw=1.5692731302544858, pitch=-0.1589992764718999, roll=1.5706400056494119), Position(x=-3.845611377487542, y=0.04245635634794696, z=0.039785872833502336)]]
        n_pos = len(data)
        if episode + 1 % episodes/n_pos == 0: 
            self.rob.set_position(data[i][1], data[i][0])
        

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

    def get_robot_state(self, block_collection):
        return get_state(self.rob, block_collection=block_collection)
    
    def get_food_collected(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.sleep(.2)
            return self.rob.get_nr_food_collected()

    def calibrate_with_block_in_arm(self):
        """
        Teleports robot so red block is in its arm and returns relevant sensor readings + camera frame.
        """
        self._teleport_robot_to_grab_position()
        self._step_simulation()  # simulate one step to get readings
        
        sensor_data = self.get_proximity_sensor_readings()
        camera_img = self.get_camera_image()

        return {"proximity": sensor_data,"camera": camera_img}
    
    def get_distance_to_base(self):
        base_position = self.rob.get_base_position()
        robot_position = self.rob.get_position()

        dx = base_position.x - robot_position.x
        dy = base_position.y - robot_position.y
        dz = base_position.z - robot_position.z
        
        return np.sqrt(dx**2 + dy**2 + dz**2)


def _teleport_robot_to_grab_position(self):
    """
    Sets robot and red block to a known configuration where the block is in the arm.
    """
    # Example values, adjust as needed for your scene
    self._set_object_pose(self.robot_handle, [0.1, 0.2, 0.0], [0, 0, 0])
    self._set_object_pose(self.red_block_handle, [0.1, 0.25, 0.02], [0, 0, 0])

def coppelia_main(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    env = Coppelia_env(rob)
    # If the environment don't follow the interface, an error will be thrown
    #check_env(env, warn=True)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()