import cv2

import numpy as np
'''
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
'''


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


def move_robot(rob, wh1=20, wh2=20, time=200):
    rob.move(wh1, wh2, time)
    rob.sleep(1)

def sensor_to_vec(sensor_data, lower_threshold = 30, higher_threshold = 80):
    result = np.select(
    [sensor_data < lower_threshold, (sensor_data >= lower_threshold) & (sensor_data < higher_threshold), sensor_data >= higher_threshold],
    [0, 1, 2]
    )
    return result

def rotate(spin_length, n_degrees):
    return (spin_length / 360) * n_degrees

    
def move(rob, start_direction, start_coord, final_coord, spin_length, y_scale = 1):
    "We take 0 degrees as facing in the positive x direction"
    "Coordinates are np.arrays [x,y]"

    "First we wnat to rotate hubert to face the new coordinate"
    dist = start_coord-final_coord
    if dist[0] == 0:
        if dist[1] > 0:
            rotation = 360 - start_direction + 90
        else:
            rotation  = 360 - start_direction - 90
    else: 
        rotation = 360 - start_direction + np.arctan(dist[1]/dist[0])
    
    rob.move(-20, 20, int(np.floor(rotate(spin_length, (rotation)))))

    "Then he should move to the new coordinate"
    move_robot(rob, time = int(np.floor(np.sqrt(dist[0]**2 + dist[1]**2))))

    "return final direction"
    return np.arctan(dist[1]/dist[0])

def run_all_playground_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    #spin_length = 0
    """
    control = True
    while control:
        
        input_text = input("Enter command: ")
        if input_text == "front":
            rob.move(100, 100, 2000)
            rob.sleep(0.5)
            print(rob.read_irs())
            print()
        elif "move" in input_text: 
            args = input_text.split()
            print(args)
            start_direction = int(args[1])
            start_coord_x = int(args[2])
            start_coord_y = int(args[3])
            start_coord = np.array([start_coord_x, start_coord_y])
            final_coord_x = int(args[4])
            final_coord_y = int(args[5])
            final_coord = np.array([final_coord_x, final_coord_y])
            spin_length = int(args[6])

            move(rob, start_direction, start_coord, final_coord, spin_length)

        elif input_text == "right":
            rob.move(100, 50, 2000)
            rob.sleep(0.5)
            print(rob.read_irs())
            print()
        elif input_text == "left":
            rob.move(50, 100, 2000)
            rob.sleep(0.5)
            print(rob.read_irs())
            print()
        elif input_text == "back":
            rob.move(-100, -100, 2000)
            rob.sleep(0.5)
            print(rob.read_irs())
            print()
        elif input_text == "sense":
            rob.sleep(0.5)
            print(rob.read_irs())
            print()
        elif "turn" in input_text:
            variables = input_text.split()
            rob.move(-int(variables[1]), int(variables[2]), int(variables[3]))
            rob.sleep(0.5)
            print(rob.read_irs())
            print()

        elif "spin_length" in input_text:
            variables = input_text.split(" ")
            print(variables)
            spin_length = int(variables[1])

        
        elif "spin " in input_text:
            variables = input_text.split()
            print(variables)
            spin = rotate(spin_length, int(variables[1]))
            print(spin)
            rob.move(-20, 20, int(spin))
            rob.sleep(0.5)
            print(rob.read_irs())
            print()

        elif "tilt" in input_text:
            rob.set_phone_tilt(109, 50)
            rob.sleep(0.5)
            print(rob.read_irs())
            print()
        elif input_text == "stop":
            control = False

        else:
            
        """
    """
    repeat = True
    rob.sleep(4)
    infra_red_data = rob.read_irs()
    while repeat:
        front_sensors = np.array([infra_red_data[3], infra_red_data[4]])
        while all(val < 45 for val in front_sensors):
                print(infra_red_data)
                rob.move_blocking(100, 100, 200)
                #rob.sleep(0.25)
                infra_red_data = rob.read_irs()
                front_sensors = np.array([infra_red_data[3], infra_red_data[4]])
        print(infra_red_data)
        #n_degrees = np.random.randint(200,6000)
        rob.move_blocking(-20, 20, 2380) 
        rob.sleep(0.25)
        #rob.move_blocking(100, 100, 1000)
        #rob.sleep(0.25)
        infra_red_data = rob.read_irs()
        print(infra_red_data)
    """
    for i in range(4):

        rob.set_phone_pan(123, 100)
        rob.sleep(5)
        rob.set_phone_tilt(109, 100)
        rob.sleep(5)




        
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


'''

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(8, num_hidden)
        self.l2 = nn.Linear(num_hidden, 8)

    def forward(self, x):
        
        # YOUR CODE HERE
        x = torch.relu(self.l1(x))  
        x = self.l2(x)  
        return x
        #raise NotImplementedError

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        # YOUR CODE HERE
        #raise NotImplementedError
        if len(self.memory) >= self.capacity:
            self.memory.pop(0) 
        self.memory.append(transition)

    def sample(self, batch_size):
        # YOUR CODE HERE
        #raise NotImplementedError
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    
    # YOUR CODE HERE
    #raise NotImplementedError
    if it >= 1000:
        epsilon =  0.05
    else:
        epsilon = 1.0 -(it/1000)*(1.0-0.05)
    
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  
        Args:
            obs: current state
        Returns:
            An action (int).
        """
        
        # YOUR CODE HERE
        #raise NotImplementedError
        if random.random() < self.epsilon:
            return random.randint(0, 1) 
        
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  
        with torch.no_grad():
            q_values = self.Q(obs) 
            action = torch.argmax(q_values).item()  

        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1
    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    
    # YOUR CODE HERE
    #raise NotImplementedError

    q_values = Q(states) 
    return q_values.gather(1, actions)  


def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of rewards. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    
    # YOUR CODE HERE
    #raise NotImplementedError

    dones = dones.to(torch.bool)  

    with torch.no_grad():
        next_q_values = Q(next_states).max(dim=1, keepdim=True)[0] 
        targets = rewards + (discount_factor * next_q_values * (~dones))
    
    return targets



def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

def take_action(rob, action):
    if action == 0:
        rob.move(100, 100, 2000)
    elif action == 1:
        rob.move(20, -20, 2375)
        rob.move(100, 100, 2000)
    elif action == 2:
        rob.move(20, -20, 1187)
        rob.move(100, 100, 2000)
    elif action == 3:
        rob.move(-20, 20, 1187)
        rob.move(100, 100, 2000)
    elif action == 4:
        rob.move(20, -20, 593)
        rob.move(100, 100, 2000)
    elif action == 5:
        rob.move(-20, 20, 593)
        rob.move(100, 100, 2000)
    elif action == 6:
        rob.move(20, -20, 2375)
        rob.move(100, 100, 2000)
    elif action == 7:
        rob.move(20, -20, 2375)
        rob.move(100, 100, 2000)
    
    



def run_episodes(rob, train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = sensor_to_vec(rob.rob.read_irs())

        steps = 0
        while True:
            
            epsilon = get_epsilon(global_steps)
            policy.set_epsilon(epsilon) 
            action = policy.sample_action(state) 
            next_state, reward, done, _ = take_action(rob, action) 
            
            memory.push((state, action, reward, next_state, done))  
            
            loss = train(Q, memory, optimizer, batch_size, discount_factor)  
            
            state = next_state  
            steps += 1
            global_steps += 1

            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                break
    return episode_durations

'''