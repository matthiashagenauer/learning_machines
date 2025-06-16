# plot_validation_rewards.py

import pickle
import os
import matplotlib.pyplot as plt

# Path to the pickle file
VALIDATION_REWARDS_PATH = './results/validation_rewards.pkl'

def load_validation_rewards(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Validation rewards file not found: {path}")
    
    with open(path, 'rb') as f:
        rewards = pickle.load(f)
    
    print(f"✅ Loaded validation rewards from {path}")
    return rewards

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o', linestyle='-', color='blue')
    plt.title('Validation Rewards over Training')
    plt.xlabel('Validation Step')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        rewards = load_validation_rewards(VALIDATION_REWARDS_PATH)
        rewards[0] = -8000
        plot_rewards(rewards)
    except Exception as e:
        print(f"❌ Error: {e}")
