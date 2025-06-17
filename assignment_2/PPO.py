import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .coppelia_env import Coppelia_env

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
TIMESTEPS = 10000 # Adjust based on your needs
EVAL_FREQ = 1000 # Evaluate every N timesteps
N_EVAL_EPISODES = 10

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
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo"
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Save the final model
    model.save(os.path.join(SAVE_DIR, "ppo_robobo_final"))
    
    # Close environment
    env.close()
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

if __name__ == "__main__":
    train_ppo()