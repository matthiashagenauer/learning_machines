import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv

from .coppelia_env import Coppelia_env

import numpy as np

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

# Constants
LOG_DIR = "/root/results/logs_ppo/"
SAVE_DIR = "/root/results/model_ppo/"
TIMESTEPS = 512 # Adjust based on your needs
EVAL_FREQ = 64 # Evaluate every N timesteps
N_EVAL_EPISODES = 1

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True

def train_ppo(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    # Create directories if they don't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    env = Coppelia_env(rob)
    
    # Wrap environment
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])  # Required for SB3
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=SAVE_DIR,
        name_prefix="ppo_robobo"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )



    print("Initialized model.")
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=16,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    
    random_callback = TensorboardCallback()
    # Train the model
    start_time = time.time()

    print("Starting training.")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, random_callback],
        tb_log_name="ppo", progress_bar = True, log_interval = 32
    )

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the final model
    model.save(os.path.join(SAVE_DIR, "ppo_robobo_final"))
    
    # Close environment
    env.close()
    if isinstance(rob, SimulationRobobo):
        if not rob.is_stopped():
            rob.stop_simulation()

if __name__ == "__main__":
    train_ppo()
