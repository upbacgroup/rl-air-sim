import os
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from train_crl import Agent
from dqn_train import QNetwork
from dqn_train import evaluate_agent as dqn_evaluate
from drone_env_sb3 import DroneEnvSB3


def make_env():
    return DroneEnvSB3(
        drone_name="drone_1",
        viz_traj=False,
        viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
        viz_image_cv2=False,
        save_img=False,
        test_mode=True
    )

def ppo_evaluate(envs, agent, device, N_iter=5, num_steps=2000, verbose=False):
    mean_reward = 0
    for iteration in range(1, N_iter + 1):

        cumulative_reward = 0
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)

        for step in range(0, num_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            act_scalar = action.cpu().numpy()

            # next_obs, reward, next_done, infos = envs.step(act_scalar)
            next_obs, reward, terminations, truncations, infos = envs.step(act_scalar)
            next_obs = torch.Tensor(next_obs).to(device)

            next_done = np.logical_or(terminations, truncations)
            
            cumulative_reward += reward 
            # print ('reward: ', reward)

            if next_done:
                break

        iteration_reward = cumulative_reward / num_steps
        mean_reward += iteration_reward

        if verbose:
            print (f"Eval Iteration: {iteration}, Reward: {iteration_reward:.3}")

    mean_reward = mean_reward / N_iter
    return mean_reward

def test_cleanrl():
    model = "ppo"
    device = 'cpu'
    model_path = "saved_models/best_ppo_07_04.pt"
    envs = DroneEnvSB3(
            drone_name="drone_1",
            viz_traj=False,
            viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
            viz_image_cv2=False,
        )

    if not os.path.isfile(model_path):
        print("No pre-trained model found")
        return

    if model == "ppo":
        agent = Agent(envs).to(device)
        eval_func = ppo_evaluate
    elif model == "dqn":
        agent = QNetwork(envs).to(device)
        eval_func = dqn_evaluate

    print(f"Model loaded from {model_path}")
    agent.load_state_dict(torch.load(model_path))

    mean_reward = eval_func(envs, agent, device, N_iter=5, verbose=True)
    print (f"Evaluation mean reward: {mean_reward:.5}")



def test_sb3():

    model_path = "SB3/PPO_April13/saved_model/best_model/best_model"
    
    env = DummyVecEnv([make_env])

    model = PPO.load(model_path, env=env)
    # model = DQN.load(model_path, env=env)

    obs = env.reset()
    # print ('obs: ', obs)
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = env.step(action)
        # print ('obs: ', obs)
        # print (f"iter: {i} reward: {reward[0]:.3f}")
        # print ('terminated: ', terminated)
        # print ('truncated: ', truncated)

        # env.render()
    # env.close()


if __name__ == "__main__":
    test_sb3()
    # test_cleanrl()