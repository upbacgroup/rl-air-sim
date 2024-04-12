# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from drone_env_sb3 import DroneEnvSB3, GateNavigator


@dataclass
class Args:
    exp_name: str = 'DQN_Reward_v1'
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Airsim_Drone_Gate"
    """the wandb's project name"""
    wandb_entity: str = 'onurakgun'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = True
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Drone"
    """the id of the environment"""
    total_timesteps: int = 750000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def evaluate_agent(envs, agent, device, N_iter=5, verbose=False):
    mean_reward = 0
    num_steps = 2000
    for iteration in range(1, N_iter + 1):

        cumulative_reward = 0
        next_obs, _ = envs.reset()

        for step in range(0, num_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                q_values = agent(torch.Tensor(next_obs).to(device))
            
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.

            next_obs, reward, next_done, infos = envs.step(actions)
            
            cumulative_reward += reward 

            if next_done:
                break

        iteration_reward = cumulative_reward / num_steps
        mean_reward += iteration_reward

        if verbose:
            print (f"Eval Iteration: {iteration}, Reward: {iteration_reward:.3}")

    mean_reward = mean_reward / N_iter
    return mean_reward

def save_best_reward(best_reward, log_file="best_reward.log"):
    """Save the best reward and model path to a log file."""
    with open(log_file, 'w') as f:
        f.write(f"{best_reward}")

def load_best_reward(log_file="best_reward.log"):
    """Load the best reward and model path from the log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            best_reward = float(lines[0].strip())
            return best_reward
    except FileNotFoundError:
        print(f"No log file found at {log_file}, starting fresh.")
        return None, None  # No best reward, no model path



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    race_tier = 4
    level_name = "Soccer_Field_Easy"
    model_path = "saved_models/drone_best_dqn_model.pt"
    log_file = "saved_models/dqn_best_reward.log"
    best_reward = -100
    eval_reward = 0
    eval_freq = 500

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.exp_name}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )

    envs = DroneEnv(
            drone_name="drone_1",
            viz_traj=False,
            viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
            viz_image_cv2=False,
        )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    envs.load_level(level_name)
    envs.start_race(race_tier)
    envs.get_ground_truth_gate_poses()
    envs.navigator = GateNavigator(envs.gate_poses_ground_truth)

    # envs.start_image_callback_thread()
    envs.start_odometry_callback_thread()

    q_network = QNetwork(envs).to(device)
    

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    if os.path.isfile(model_path) and args.upload_model:
        q_network.load_state_dict(torch.load(model_path))
        best_reward = float(load_best_reward(log_file=log_file))
        print(f"Model loaded from {model_path}")
    else:
        print("No pre-trained model found. Starting training from scratch.")


    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())


    print ('Training has been started. Good luck!\n')

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(1, args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()


        # print ('actions: ', actions)
        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs, rewards, terminations, infos = envs.step(actions)

        # print (f"action: {actions[0]}, next_obs: {next_obs[-2]:.3}, reward: {rewards:.3}")

        if args.track:
            writer.add_scalar("charts/episodic_return", rewards, global_step)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if info and "episode" in info:
        #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 1 == 0 and args.track:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        
        if global_step % eval_freq == 0 and global_step > args.learning_starts:
            eval_reward = evaluate_agent(envs, q_network, device)
            print(f"Eval Iteration: {global_step}/{args.total_timesteps}: Average Reward: {eval_reward:.4f}")

            if args.track:
                writer.add_scalar("eval/episodic_return", eval_reward, global_step)

            if args.save_model and eval_reward > best_reward:
                best_reward = np.copy(eval_reward)
                torch.save(q_network.state_dict(), model_path)
                save_best_reward(best_reward, log_file=log_file)
                print(f"New best reward: {best_reward:.3} model saved to {model_path}")


    envs.close()
    if args.track:
        writer.close()
