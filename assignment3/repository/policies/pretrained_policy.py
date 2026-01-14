import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

VECNORM_PATH = "agents/vec_normalize.pkl"

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_pretrained_policy(model_path="model_final"):
    device = pick_device()

    # Dummy env just to load stats
    dummy_env = DummyVecEnv([lambda: gym.make("highway-fast-v0")])
    vecnorm = VecNormalize.load(VECNORM_PATH, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    model = PPO.load(model_path, device=device)

    def policy(obs, info):
        # obs is single observation from Gymnasium, convert to batch
        obs_batch = np.expand_dims(obs, axis=0)
        obs_norm = vecnorm.normalize_obs(obs_batch)
        action, _ = model.predict(obs_norm, deterministic=True)
        return int(action[0])

    return policy