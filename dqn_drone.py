import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from drone_env_sb3 import DroneEnvSB3, GateNavigator


def make_env():
    return DroneEnvSB3(
        drone_name="drone_1",
        viz_traj=False,
        viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
        viz_image_cv2=False,
        save_img=False
    )


env = DummyVecEnv([make_env])  # Notice make_env without parentheses

# # Create a DummyVecEnv for main airsim gym env
# env = DummyVecEnv(
#     [
#         lambda: Monitor(
#             gym.make(
#                 "airgym:airsim-drone-sample-v0",
#                 ip_address="127.0.0.1",
#                 step_length=0.25,
#                 image_shape=(84, 84, 1),
#             )
#         )
#     ]
# )

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
# eval_callback = EvalCallback(
#     env,
#     callback_on_new_best=None,
#     n_eval_episodes=5,
#     best_model_save_path=".",
#     log_path=".",
#     eval_freq=10000,
# )

run_name = 'DQN_April1'

eval_callback = EvalCallback(
    env,
    best_model_save_path=f"SB3/{run_name}/saved_model/best_model",
    log_path=f"SB3/{run_name}/logs",
    eval_freq=10000,
    n_eval_episodes = 5,
    deterministic=True,
    render=False,
    verbose=False
)


callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name=f"SB3/{run_name}/tb_model/",
    **kwargs
)
