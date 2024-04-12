# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from drone_env_sb3 import DroneEnvSB3, GateNavigator
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.torch_layers import NatureCNN  # CNN from Nature paper
from stable_baselines3.common.policies import ActorCriticPolicy



@dataclass
class Args:
    exp_name: str = 'PPO_CNN_28March'
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = "Airsim_Drone_Gate"
    """the wandb's project name"""
    wandb_entity: str = 'onurakgun'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Drone"
    """the id of the environment"""
    total_timesteps: int = 200000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# class CNNPolicy(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(CNNPolicy, self).__init__()
#         # Assuming num_inputs = 3 for RGB images
#         self.conv1 = layer_init(nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4))
#         self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
#         self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
#         # self.conv4 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        
        
#         # Compute the size of the flat features after the convolutional layers
#         def conv2d_size_out(size, kernel_size=3, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(320, 8, 4), 4, 2), 3, 1)
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(240, 8, 4), 4, 2), 3, 1)
#         linear_input_size = convw * convh * 64

#         self.fc = layer_init(nn.Linear(linear_input_size, 512))
        
#         self.critic = layer_init(nn.Linear(512, 1), std=1.)
#         self.actor = layer_init(nn.Linear(512, num_outputs), std=0.01)

#     def forward(self, x):
#         # print ("x: ", x.size())
#         # Assuming x is of shape (N, C, H, W)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         # x = F.relu(self.conv4(x))
#         # print ("x: ", x.size())
#         x = x.view(x.size(0), -1)  # Flatten the output for each image
#         # print ("x: ", x.size())
#         x = F.relu(self.fc(x))

#         return self.actor(x), self.critic(x)
    


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         num_inputs = envs.observation_space.shape[0]  # Assuming CxHxW
#         num_outputs = envs.action_space.n
        
#         self.policy = CNNPolicy(num_inputs, num_outputs)

#     def get_value(self, x):
#         _, value = self.policy(x)
#         return value

#     def get_action_and_value(self, x, action=None):
#         logits, value = self.policy(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
        
#         return action, probs.log_prob(action), probs.entropy(), value


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def evaluate_agent(envs, agent, device, N_iter=10, verbose=False):
    mean_reward = 0
    evaluation_results = []
    for iteration in range(N_iter):

        cumulative_reward = 0
        next_obs, _ = envs.reset(seed=args.seed)
        # next_obs = torch.Tensor(next_obs).to(device)
        # next_obs = torch.tensor(next_obs).to(device)  # Convert to tensor
        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float)
        next_obs = next_obs.unsqueeze(0)  # Reorder to (N, C, H, W)

        for step in range(0, args.num_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            act_scalar = action.cpu().numpy()
            act_scalar = act_scalar[0]
            # next_obs, reward, next_done, infos = envs.step(act_scalar)
            next_obs, reward, terminations, truncations, infos = envs.step(act_scalar)
            next_done = np.logical_or(terminations, truncations)

            # next_obs = torch.Tensor(next_obs).to(device)
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float)
            next_obs = next_obs.unsqueeze(0)  # Reorder to (N, C, H, W)
            
            cumulative_reward += reward 

            if next_done:
                break

        iteration_reward = cumulative_reward / (step + 1)
        mean_reward += iteration_reward
        evaluation_results.append(infos)

        if verbose:
            print (f"Eval Iteration: {iteration}, Reward: {iteration_reward:.3}")

    mean_reward = mean_reward / N_iter
    return mean_reward, evaluation_results

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
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m")
    
    # model_path = f"saved_models/best_ppo_05_04.pt"
    # log_file = f"saved_models/best_ppo_05_04.log"

    model_path = f"saved_models/best_ppo_{date_time}.pt"
    log_file = f"saved_models/best_ppo_{date_time}.log"

    # print ('args.num_iterations: ', args.num_iterations)
    # print ('args.minibatch_size: ', args.minibatch_size)
    # print ('args.batch_size: ', args.batch_size)

    run_name = f"{args.exp_name}_{date_time}"
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
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )

    envs = DroneEnvSB3(
            drone_name="drone_1",
            viz_traj=False,
            viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
            viz_image_cv2=False,
        )

    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"



    agent = Agent(envs).to(device)
    

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    best_reward = -500
    eval_reward = 0

    start_time = time.time()

    
    if os.path.isfile(model_path):
        agent.load_state_dict(torch.load(model_path))
        best_reward = float(load_best_reward(log_file=log_file))
        print(f"Model loaded from {model_path}")
    else:
        print("No pre-trained model found. Starting training from scratch.")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print ('Training has been started. Good luck!\n')

    for iteration in range(1, args.num_iterations):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow


        cumulative_reward = 0

        next_obs, _ = envs.reset(seed=args.seed)
        # next_obs = torch.tensor(next_obs).to(device)  # Convert to tensor
        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float)
        next_obs = next_obs.unsqueeze(0)  # Reorder to (N, C, H, W)
        next_done = torch.zeros(args.num_envs).to(device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            
            
            actions[step] = action
            logprobs[step] = logprob

            act_scalar = action.cpu().numpy()
            act_scalar = act_scalar[0]
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(act_scalar)
            next_done = np.logical_or(terminations, truncations)

            # next_obs, reward, next_done, infos = envs.step(act_scalar)
            
            # print (f"action: {act_scalar}, reward: {reward:.4}")

            if type(next_done) is not list:
                next_done = np.array([next_done], dtype=np.float32)

            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Now convert to PyTorch tensors
            # next_obs = torch.Tensor(next_obs).to(device)
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float)
            next_obs = next_obs.unsqueeze(0)  # Reorder to (N, C, H, W)
            next_done = torch.Tensor(next_done).to(device)

            cumulative_reward += reward 

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


            if next_done[0]:
                break


        train_reward = cumulative_reward / (step + 1)
        print(f"Train Iteration: {iteration}/{args.num_iterations}: Average Reward: {train_reward:.4f}")


        if iteration % 20 == 0:
            print ("\nEvaluation Mode")
            eval_reward, evaluation_results = evaluate_agent(envs, agent, device, verbose=False)
            print(f"Eval Iteration: {iteration}/{args.num_iterations}: Average Reward: {eval_reward:.4f}")
            envs.scheduler.update_gate_success(evaluation_results)
            print("Successfull gate passes: ", envs.scheduler.gate_successes) 

            if args.save_model and eval_reward > best_reward:
                best_reward = np.copy(eval_reward)
                torch.save(agent.state_dict(), model_path)
                save_best_reward(best_reward, log_file=log_file)
                print(f"New best reward: {best_reward:.3} model saved to {model_path}")
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        if args.track:
            writer.add_scalar("charts/train_reward", train_reward, global_step)
            writer.add_scalar("charts/eval_reward", eval_reward, global_step)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            

    envs.close()

    if args.track:
        writer.close()
    
