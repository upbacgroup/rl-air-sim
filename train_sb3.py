import time
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import NatureCNN  # CNN from Nature paper
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from drone_env_sb3 import DroneEnvSB3
import wandb
from wandb.integration.sb3 import WandbCallback



class CustomRNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomRNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        # Assuming observations are one-dimensional for simplicity
        self.lstm = nn.LSTM(input_size=observation_space.shape[0], hidden_size=features_dim, batch_first=True)
        
    def forward(self, observations):
        # SB3 vectorizes envs by default, so we add a time dimension of 1 for compatibility with LSTM
        if observations.dim() == 2:  # (num_envs, num_features)
            observations = observations.unsqueeze(1)  # Add a time dimension (num_envs, 1, num_features)
        lstm_out, _ = self.lstm(observations)
        return lstm_out[:, -1, :]  # Return only the last output from the sequence

class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomRNNFeaturesExtractor(self.observation_space, features_dim=256)


def make_env(log_dir):
    def _init():
        env = DroneEnvSB3(
            drone_name="drone_1",
            viz_traj=False,
            viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
            viz_image_cv2=False,
            save_img=False
        )
        env = Monitor(env, log_dir)
        return env
    return _init

def main():
    n_eval_episodes = 5
    N_eval_freq = 1000
    train_timesteps = 1000000
    deterministic = True

    load_path = '' #"SB3/PPO_Reward_v4_19_03_18/saved_model/best_model/best_model"
    exp_name = 'PPO_April3_v8'
    wandb_project_name = "Airsim_Drone_Gate"
    wandb_entity = "onurakgun"
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"SB3/{run_name}/log_train"


    # run = wandb.init(
    #         project=wandb_project_name,
    #         entity=wandb_entity,
    #         sync_tensorboard=True,
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    
    
    env = DummyVecEnv([make_env(log_dir)])  # Notice make_env without parentheses

    # eval_env = DummyVecEnv([make_env])  # Notice make_env without parentheses

    # It's common to stack frames in environments with image observations to give the model a sense of motion
    # Change `n_stack` to the number of frames you want to stack
    # n_stack = 4

    # # Wrap your environment with a DummyVecEnv and stack frames
    # env = VecFrameStack(vec_env, n_stack=n_stack)

    # eval_env = VecFrameStack(DummyVecEnv([make_env]), n_stack=n_stack)

    # # Define the policy architecture. Here, we're using a CNN suitable for image observations
    # policy_kwargs = dict(
    #     features_extractor_class=NatureCNN,
    #     features_extractor_kwargs=dict(features_dim=512),
    # )

    # Initialize the PPO model
    # model = PPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)


    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"SB3/{run_name}/saved_model/best_model",
        log_path=f"SB3/{run_name}/logs",
        eval_freq=N_eval_freq,
        n_eval_episodes = n_eval_episodes,
        deterministic=deterministic,
        render=False,
        verbose=False
    )

    # callback = EvalCallback(eval_env=vec_env, n_eval_episodes = n_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder + "_log",
    #                                     best_model_save_path = current_folder + "/saved_models/", deterministic=deterministic, verbose=1)

    # wandb_callback = WandbCallback(
    #     gradient_save_freq=10,
    #     model_save_path=f"saved_models/{run_name}/wandb",
    #     verbose=2,
    # )

    # combined_callback = CallbackList([eval_callback, wandb_callback])


    if load_path == "":
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
        # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
        print ("New model is created")
    else:
        model = PPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=env)
        # model = RecurrentPPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=vec_env)
        print('The previous model is loaded from ', load_path)

    
                
    model.learn(total_timesteps=train_timesteps, callback=eval_callback)
    


if __name__ == '__main__':
    main()