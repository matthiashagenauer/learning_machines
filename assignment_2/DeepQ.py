import cv2
import os

import numpy as np

from .utils import *

import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from .coppelia_env import Coppelia_env

from tqdm import tqdm as _tqdm




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



class QNetwork(nn.Module):
    """
    Very simple NN
    """
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 8)

    def forward(self, x):
        
        x = torch.relu(self.l1(x))  
        x = self.l2(x)  
        return x
  

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
      
        if len(self.memory) >= self.capacity:
            self.memory.pop(0) 
        self.memory.append(transition)

    def sample(self, batch_size):
     
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



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
        
        if random.random() < self.epsilon:
            return random.randint(0, 7) 
        
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
    print(state)
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]
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
    
    return loss.item() 



def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, steps_per_episode):
    writer = SummaryWriter(log_dir="/root/results/tensorboard")  # TensorBoard writer

    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in _tqdm(range(num_episodes)):
        env.reset()
        state = env.get_robot_state()
        
        steps = 0
        for i in _tqdm(range(steps_per_episode)):
            
            epsilon = get_epsilon(global_steps)
            policy.set_epsilon(epsilon) 
            action = policy.sample_action(state)
          
            next_state, reward, done, _, _ = env.step(action) 
            
            memory.push((state, action, reward, next_state, done))  
            
            loss = train(Q, memory, optimizer, batch_size, discount_factor)  

            if loss is not None:
                writer.add_scalar("Loss/step", loss, i)

            writer.add_scalar("Epsilon/step", epsilon, i)
            writer.add_scalar("Reward/step", reward, i)
            
            state = next_state  
            steps += 1
            global_steps += 1

            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                episode_durations.append(steps)
                break

    # Save the Q-network parameters
    torch.save(Q.state_dict(), '/root/results/q_network_params.pth')
    print("Q-network parameters saved to q_network_params.pth")

    writer.close()

    return episode_durations



def run_training(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    memory_size = 200
    num_episodes = 5
    learn_rate = 1e-4
    batch_size = 64
    steps_per_episode = 400
    
    path = "/root/results/"
    Q = QNetwork()

    # Load model parameters if available
    model_path = f'{path}q_network_params.pth'
    if os.path.exists(model_path):
        Q.load_state_dict(torch.load(model_path))
        print("Loaded existing Q-network parameters from", model_path)
    else:
        print("No existing Q-network parameters found. Initializing new model.")


    memory = ReplayMemory(memory_size)
    policy = EpsilonGreedyPolicy(Q, .5)
    env = Coppelia_env(rob, deepQ = True)
    run_episodes(train=train, 
                 env= env, 
                 Q=Q, 
                 policy=policy, 
                 memory=memory, 
                 batch_size=batch_size, 
                 learn_rate=learn_rate, 
                 num_episodes=num_episodes, 
                 steps_per_episode=steps_per_episode,
                 discount_factor=0.99)
        
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


"""
def apply_policy(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    path = "/root/results/"
    Q = QNetwork()

    # Load model parameters if available
    model_path = f'{path}q_network_params.pth'
    if os.path.exists(model_path):
        Q.load_state_dict(torch.load(model_path))
        print("Loaded existing Q-network parameters from", model_path)
    else:
        print("No existing Q-network parameters found. Initializing new model.")
    for i in _tqdm(range(10000000)):
    #while True:
        state = np.array(sensor_to_vec(get_sensor_data(rob)))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  
        with torch.no_grad():
            q_values = Q(state) 
            action = torch.argmax(q_values).item()  
        state, reward = take_action(rob, action)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
"""