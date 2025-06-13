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

from tqdm import tqdm as _tqdm
import pickle # Import the pickle module




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
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
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
    
    # YOUR CODE HERE
    #raise NotImplementedError

    q_values = Q(states) 
    return q_values.gather(1, actions)  


def compute_targets(Q, rewards, next_states, discount_factor):
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

    # removed

    with torch.no_grad():
        next_q_values = Q(next_states).max(dim=1, keepdim=True)[0] 
        #targets = rewards + (discount_factor * next_q_values * (~dones))

        targets = rewards + (discount_factor * next_q_values)
    
    return targets






def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state = zip(*transitions)
    #("States:")
    #print(state)
    #print("Rewards:")
    #print(reward)

    
    # convert to PyTorch and define 
    state = torch.tensor(np.array(state), dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]
    next_state = torch.tensor(np.array(next_state), dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]

    # = torch.tensor(state, dtype=torch.float)
    #action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    #next_state = torch.tensor(next_state, dtype=torch.float)
    #reward = torch.tensor(reward, dtype=torch.float)[:, None]
    #done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())



def run_episodes(rob, train, Q, policy, memory, batch_size = 8, discount_factor=0.95, learn_rate=1e-5, iterations=1000):
    writer = SummaryWriter(log_dir="/root/results/tensorboard")  # TensorBoard writer
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    #global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    state = sensor_to_vec(get_sensor_data(rob))
    steps = 0
    
    validation_reward = []
    reset_position = rob.get_position()
    reset_orientation = rob.get_orientation()

    for i in _tqdm(range(iterations)):
    #while True:
        if i % 100 == 0:
            teleport(rob, reset_position, reset_orientation)
        if i % int(iterations / 20) == 0:
            teleport(rob, reset_position, reset_orientation)
            validation_reward.append(validate(rob, classic=False, q_network=Q))
            teleport(rob, reset_position, reset_orientation)
        #epsilon = get_epsilon(global_steps)
        epsilon = get_epsilon(iters_left=iterations - i, total_iters=iterations)
        policy.set_epsilon(epsilon) 
        action = policy.sample_action(state) 
        #print(f"Action taken: {action}")
        next_state, reward = take_action(rob=rob, action=action) 
        
        memory.push((state, action, reward, next_state))  
        
        loss = train(Q, memory, optimizer, batch_size, discount_factor)  
        
        if loss is not None:
            writer.add_scalar("Loss/step", loss, i)

        writer.add_scalar("Epsilon/step", epsilon, i)
        writer.add_scalar("Reward/step", reward, i)
        
        state = next_state  
        steps += 1
        #global_steps += 1

    # Save the Q-network parameters
    torch.save(Q.state_dict(), '/root/results/q_network_params.pth')
    print("Q-network parameters saved to q_network_params.pth")

    # Save the memory buffer
    with open('/root/results/replay_memory.pkl', 'wb') as f:
        pickle.dump(memory, f)
    print("Replay memory saved to replay_memory.pkl")

    # Save validation rewards
    with open('/root/results/validation_rewards.pkl', 'wb') as f:
        pickle.dump(validation_reward, f)
    print("Validation rewards saved to validation_rewards.pkl")

    writer.close()  # Close the TensorBoard writer


def run_training(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    memory_size = 200
    iterations = 20000
    learn_rate = 1e-4
    batch_size = 64
    
    path = "/root/results/"
    Q = QNetwork()

    # Load model parameters if available
    model_path = f'{path}q_network_params.pth'
    if os.path.exists(model_path):
        Q.load_state_dict(torch.load(model_path))
        print("Loaded existing Q-network parameters from", model_path)
    else:
        print("No existing Q-network parameters found. Initializing new model.")

    # Load replay memory if available
    memory_path = f'{path}replay_memory.pkl'
    if os.path.exists(memory_path):
        with open(memory_path, 'rb') as f:
            memory = pickle.load(f)
        print("Loaded existing replay memory from", memory_path)
    else:
        memory = ReplayMemory(memory_size)
        print("No existing replay memory found. Creating new memory.")

    policy = EpsilonGreedyPolicy(Q, .5)
    run_episodes(rob, train, Q, policy, memory, batch_size=batch_size, learn_rate=learn_rate, iterations=iterations)
        
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()



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
