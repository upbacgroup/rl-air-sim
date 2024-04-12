import os
from argparse import ArgumentParser
from client_updated import *
import airsimdroneracinglab as airsim
import cv2
import threading
import time
import utils
import numpy as np
import math
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from PIL import Image

def quaternion_orientational_distance(q1, q2):
    # Ensure quaternions are normalized
    q1 = R.from_quat(q1).as_quat()
    q2 = R.from_quat(q2).as_quat()

    # Calculate the relative rotation quaternion
    q_r = R.from_quat(q2) * R.from_quat(q1).inv()
    
    # Extract the real part of the quaternion
    w = q_r.as_quat()[0]
    
    # Calculate the orientational distance
    theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    return theta


def calculate_angle_difference(vector1, vector2):
    # Convert airsim.Vector3r to numpy arrays
    v1 = np.array([vector1.x_val, vector1.y_val, vector1.z_val])
    v2 = np.array([vector2.x_val, vector2.y_val, vector2.z_val])
    
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Ensure cos_theta is within the valid range for arccos to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    theta = np.arccos(cos_theta)
    
    # Optionally, convert the angle to degrees
    angle_degrees = np.degrees(theta)
    
    return theta

def get_neighborhood_position(position, deviation=0.25):
    """Discretize the space to identify the neighborhood of a given position."""
    x, y, z = position
    neighborhood_x = round(x / deviation) * deviation
    neighborhood_y = round(y / deviation) * deviation
    neighborhood_z = round(z / deviation) * deviation
    return (neighborhood_x, neighborhood_y, neighborhood_z)


class TrainingLogger:
    def __init__(self):
        self.episode_logs = []
        self.position_change_scores = {}  # Tracks scores by neighborhood

    def log_step(self, position, reward):
        """Log the drone's position and reward for the current step."""
        self.episode_logs.append((position, reward))

    def end_episode(self):
        """Analyze the episode and update scores based on neighborhoods."""
        for i in range(1, len(self.episode_logs)):
            prev_position, prev_reward = self.episode_logs[i - 1]
            position, reward = self.episode_logs[i]
            reward_change = reward - prev_reward

            if reward_change < 0:  # Focus on negative changes
                neighborhood = get_neighborhood_position(prev_position)
                if neighborhood not in self.position_change_scores:
                    self.position_change_scores[neighborhood] = 0
                self.position_change_scores[neighborhood] += abs(reward_change)
                
        self.episode_logs = []

    def get_challenging_start_position(self, episodes_to_consider):
        if len(self.position_change_scores) == 0 or episodes_to_consider == 0:
            return None
        
        for position in self.position_change_scores:
            self.position_change_scores[position] /= episodes_to_consider
        
        challenging_neighborhood = max(self.position_change_scores, key=self.position_change_scores.get)
        return challenging_neighborhood



class DynamicCurriculumScheduler:
    def __init__(self, total_gates, review_frequency=0.1):
        self.total_gates = total_gates
        self.review_frequency = review_frequency
        self.current_start_gate = 0
        self.gate_successes = [0] * total_gates
        self.success_threshold = 0.6  # Success rate threshold to advance
    
    def update_gate_success(self, evaluation_results):
        """
        Updates gate success rates and adjusts the starting gate based on the evaluation results.
        """
        for result in evaluation_results:
            passed_gate = result["passed_gates"]
            # Increment attempts and successes up to and including the passed gate
            for gate in range(passed_gate + 1):
                self.gate_successes[gate] += 1

        # Calculate success rates and determine the furthest gate meeting the 60% success criterion
        for gate in range(self.current_start_gate, self.total_gates):
            if self.gate_successes[gate] > 0:
                success_rate = self.gate_successes[gate] / len(evaluation_results)
                # print (gate, ' success rate: ', success_rate)
                if success_rate >= self.success_threshold:
                    # Update current start gate to the next gate after the furthest successful gate
                    self.current_start_gate = gate + 1
                else:
                    break  # Stop if a gate does not meet the success criterion

    def select_start_gate(self):
        """
        Selects a start gate for the next training session, considering review frequency.
        """
        return np.random.randint(0, self.current_start_gate + 1)
        # if np.random.rand() < self.review_frequency:
        #     # Choose randomly among the gates up to the current starting gate for review
        #     return np.random.randint(0, self.current_start_gate + 1)
        # else:
        #     # Otherwise, start from the current start gate, but not beyond the last gate
        #     return min(self.current_start_gate, self.total_gates - 1)

        

class GateNavigator:
    def __init__(self, gates_positions):
        """
        Initialize the navigator with the positions of all gates.
        
        Parameters:
        - gates_positions: A list of gate positions, where each position is np.array([x, y, z]).
        """
        self.gates_positions = gates_positions
        self.current_gate_index = 0  # Start with the first gate as the target
        self.reached_gate = False
    
    def update_drone_position(self, drone_position, threshold=1.5):
        """
        Update the drone's position and determine if it's time to target the next gate.
        
        Parameters:
        - drone_position: The current position of the drone as np.array([x, y, z]).
        """
        if self.current_gate_index >= len(self.gates_positions):
            print("All gates have been passed.")
            return
        
        # Calculate the distance to the current target gate
        current_gate = self.gates_positions[self.current_gate_index]
        gate_position = np.array([current_gate.position.x_val, current_gate.position.y_val, current_gate.position.z_val])

        distance_to_gate = np.linalg.norm(drone_position - gate_position)
        
        # Check if the drone has reached the current gate
        if distance_to_gate < threshold:
            self.reached_gate = True
        
        # If the drone has reached the gate and is now moving away, switch to the next gate
        if self.reached_gate: #and distance_to_gate > threshold:
            self.current_gate_index += 1  # Move to the next gate
            self.reached_gate = False  # Reset the reached_gate flag
            # if self.current_gate_index < len(self.gates_positions):
            #     print(f"Switched to gate {self.current_gate_index}.")
            # else:
            #     print("All gates have been passed.")
    
    def get_current_target_gate(self):
        """
        Get the position of the current target gate.
        
        Returns:
        - The position of the current target gate as np.array([x, y, z]), or None if all gates are passed.
        """
        if self.current_gate_index < len(self.gates_positions):
            return self.gates_positions[self.current_gate_index]
        else:
            return None


# drone_name should match the name in ~/Document/AirSim/settings.json
class DroneEnvSB3(gym.Env):
    def __init__(
        self,
        drone_name="drone_1",
        viz_traj=False,
        viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0],
        viz_image_cv2=False,
        save_img=False,
        test_mode=False,
        folder="image_outputs"
    ):
        super(DroneEnvSB3, self).__init__()
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba
        self.save_img = save_img
        
        self.navigator = None
        
        self.airsim_client = MultirotorClient()
        self.airsim_client.confirmConnection()

        
        
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        
        # self.home_position = np.array([state.position.x_val, state.position.y_val - 2.5, state.position.z_val + 1.0])
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands
        self.airsim_client_images = MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03)
        )

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

        self.odometry_callback_thread = threading.Thread(
            target=self.repeat_timer_odometry_callback,
            args=(self.odometry_callback, 0.02),
        )
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = (
            10  # see https://github.com/microsoft/AirSim-Drone-Racing-Lab/issues/38
        )

        # Define action and observation space
        # Actions: Assume 8 discrete actions (e.g., forward, backward, left, right, up, down, no-op)
        self.action_space = spaces.Discrete(8)
        
        # Observation space: position (x, y, z) and orientation (w, x, y, z) of the drone, 
        # and the position of the next gate (x, y, z), and the position and orientation distance between drone and gate
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        # image_shape=(84, 84, 1)
        # self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.step_length = 0.1
        self.prev_gate_index = 0
        self.gate_index = 0
        self.timesteps = 0
        self.totalsteps = 0
        self.max_steps = 1000
        self.positions = []
        self.max_positions_stored = 50  # Max history to track
        self.max_history_length = 10
        self.drifting_threshold = 10
        self.previous_distance = 0
        self.img_index = 0
        self.episode_number = 1
        self.test_mode = test_mode

        race_tier = 1
        level_name = "Soccer_Field_Easy"
        self.load_level(level_name)
        self.start_race(race_tier)
        self.get_ground_truth_gate_poses()
        self.navigator = GateNavigator(self.gate_poses_ground_truth)

        self.total_gates = len(self.gate_poses_ground_truth)
        self.scheduler = DynamicCurriculumScheduler(self.total_gates, review_frequency=0.25)

        
        self.start_odometry_callback_thread()
        # self.start_image_callback_thread()

        if self.save_img:
            self.start_image_callback_thread()
            self.img_folder = folder
            if not os.path.exists(self.img_folder): 
                os.makedirs(self.img_folder) 

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.logger = TrainingLogger()

    

    def reset(self, seed=None):
        self.timesteps = 0
        episodes_to_consider = 20
        std_dev = np.array([0.2, 0.2, 0.2])
        self.consecutive_stuck_steps = 0 
        self.drift_duration = 0
        self.stuck_threshold = 25

        self.positions = []
        self.distance_history = []
        self.airsim_client.reset()
        self.navigator = GateNavigator(self.gate_poses_ground_truth)
        self.scheduler = DynamicCurriculumScheduler(self.total_gates, review_frequency=0.25)
        self.initialize_drone()

        
        self.takeoff_with_moveOnSpline(takeoff_height=-1.0) #
        self.airsim_client.moveToYawAsync(yaw=90).join()

        state = self.airsim_client.getMultirotorState().kinematics_estimated
        self.home_position = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        
        gate_index = 0 #self.scheduler.select_start_gate()

        gate_state = self.gate_poses_ground_truth[gate_index]
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        
        

        if self.episode_number % episodes_to_consider == 0:
            start_position = self.logger.get_challenging_start_position(episodes_to_consider)

            if start_position is not None:
                # for log in self.logger.episode_logs[0]:
                #     print (log)

                # print("New start position: ", start_position)
                init_position = airsim.Vector3r(start_position[0], start_position[1], start_position[2])
                self.airsim_client.moveOnSplineAsync(
                    [init_position],
                    vel_max=15.0,
                    acc_max=5.0,
                    add_position_constraint=True,
                    add_velocity_constraint=False,
                    add_acceleration_constraint=False,
                    viz_traj=self.viz_traj,
                    viz_traj_color_rgba=self.viz_traj_color_rgba,
                    vehicle_name=self.drone_name,
                ).join()

        else:
            
            # Initialize the drone's position with Gaussian noise
            initial_position_array = self.home_position + np.random.normal(0, std_dev, size=self.home_position.shape)
            initial_position = airsim.Vector3r(initial_position_array[0], initial_position_array[1], initial_position_array[2])
                    
            # self.airsim_client.moveToYawAsync(yaw=90).join()

            self.airsim_client.moveToPositionAsync(x=initial_position_array[0], y=initial_position_array[1], z=initial_position_array[2], velocity=15).join()

            # self.airsim_client.moveOnSplineAsync(
            #     [initial_position],
            #     vel_max=15.0,
            #     acc_max=5.0,
            #     add_position_constraint=True,
            #     add_velocity_constraint=False,
            #     add_acceleration_constraint=False,
            #     viz_traj=self.viz_traj,
            #     viz_traj_color_rgba=self.viz_traj_color_rgba,
            #     vehicle_name=self.drone_name,
            # ).join()


        # # gate_index = np.random.randint(0,6)
        # # print(f"Drone goes towards to the Gate {gate_index} for training")

        # for i in range(gate_index):
        #     gate_state = self.gate_poses_ground_truth[i]
        #     gate_position = airsim.Vector3r(gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val)
        #     # print(f"First target Gate {i} x:{gate_state.position.x_val:.4f}, y:{gate_state.position.y_val:.4f}, z:{gate_state.position.z_val:.4f}")

        #     self.airsim_client.moveOnSplineAsync(
        #         [gate_position],
        #         vel_max=15.0,
        #         acc_max=5.0,
        #         add_position_constraint=True,
        #         add_velocity_constraint=False,
        #         add_acceleration_constraint=False,
        #         viz_traj=self.viz_traj,
        #         viz_traj_color_rgba=self.viz_traj_color_rgba,
        #         vehicle_name=self.drone_name,
        #     ).join()

        # #     state = self.airsim_client.getMultirotorState().kinematics_estimated
        #     # print(f"Drone Position x:{state.position.x_val:.4f}, y:{state.position.y_val:.4f}, z:{state.position.z_val:.4f}")


        
        
        # if gate_index > 0:
        #     prev_gate_state = self.gate_poses_ground_truth[gate_index - 1]
        #     prev_gate_position = np.array([prev_gate_state.position.x_val, prev_gate_state.position.y_val, prev_gate_state.position.z_val])
        #     init_position_mean = (gate_position + prev_gate_position) / 2
        # else:
        #     init_position_mean = np.copy(self.home_position)


        # # Define the standard deviation for the Gaussian noise
        # std_dev = np.array([0.2, 0.2, 0.2])  # Adjust these values based on the desired variability

        # # Initialize the drone's position with Gaussian noise
        # initial_position_array = init_position_mean + np.random.normal(0, std_dev, size=init_position_mean.shape)
        # initial_position = airsim.Vector3r(initial_position_array[0], initial_position_array[1], initial_position_array[2])
                
        # # self.airsim_client.moveToYawAsync(yaw=90).join()

        # # self.airsim_client.moveToPositionAsync(x=initial_position_array[0], y=initial_position_array[1], z=initial_position_array[2], velocity=15).join()

        # self.airsim_client.moveOnSplineAsync(
        #     [initial_position],
        #     vel_max=15.0,
        #     acc_max=5.0,
        #     add_position_constraint=True,
        #     add_velocity_constraint=False,
        #     add_acceleration_constraint=False,
        #     viz_traj=self.viz_traj,
        #     viz_traj_color_rgba=self.viz_traj_color_rgba,
        #     vehicle_name=self.drone_name,
        # ).join()


        # if gate_index == 0:
        #     self.airsim_client.moveToYawAsync(yaw=90).join()


        self.previous_position = np.copy(self.home_position)
        self.previous_distance = np.linalg.norm(self.home_position - gate_position)

        # print(f"Initial Distance {self.previous_distance:.5f}")
        
        observation = self._get_observation()

        return observation, {}

    def _do_action_angle_rate(self, action, step_length=0.25, duration=0.5):
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        angular_rates = state.angular_velocity
        quad_vel = self.airsim_client.getMultirotorState().kinematics_estimated

        # angle_step_length = self.step_length * 180 / np.pi

        if action == 0:
            self.airsim_client.moveByAngleRatesZAsync(roll_rate=angular_rates.x_val + step_length, pitch_rate=0, yaw_rate=0, z=state.position.z_val, duration=duration).join()
        elif action == 1:
            self.airsim_client.moveByAngleRatesZAsync(roll_rate=angular_rates.x_val - step_length, pitch_rate=0, yaw_rate=0, z=state.position.z_val, duration=duration).join()
        elif action == 2:
            self.airsim_client.moveByAngleRatesZAsync(roll_rate=0, pitch_rate=angular_rates.y_val + step_length, yaw_rate=0, z=state.position.z_val, duration=duration).join()
        elif action == 3:
            self.airsim_client.moveByAngleRatesZAsync(roll_rate=0, pitch_rate=angular_rates.y_val - step_length, yaw_rate=0, z=state.position.z_val, duration=duration).join()
        elif action == 4:
            self.airsim_client.moveByAngleRatesZAsync(roll_rate=0, pitch_rate=0, yaw_rate=angular_rates.z_val + step_length, z=state.position.z_val, duration=duration).join()
        elif action == 5:
            self.airsim_client.moveByAngleRatesZAsync(roll_rate=0, pitch_rate=0, yaw_rate=angular_rates.z_val - step_length, z=state.position.z_val, duration=duration).join()
        elif action == 6:
            # self.airsim_client.moveByAngleRatesZAsync(roll_rate=0, pitch_rate=0, yaw_rate=0, z=state.position.z_val + 2*step_length, duration=duration).join()
            self.airsim_client.moveByVelocityAsync(vx = quad_vel.linear_velocity.x_val, vy = quad_vel.linear_velocity.y_val, vz = quad_vel.linear_velocity.z_val + step_length, duration = duration).join()
        elif action == 7:
            # self.airsim_client.moveByAngleRatesZAsync(roll_rate=0, pitch_rate=0, yaw_rate=0, z=state.position.z_val - 2*step_length, duration=duration).join()
            self.airsim_client.moveByVelocityAsync(vx = quad_vel.linear_velocity.x_val, vy = quad_vel.linear_velocity.y_val, vz = quad_vel.linear_velocity.z_val - step_length, duration = duration).join()
        
        state = self.airsim_client.getMultirotorState().kinematics_estimated
    

    def transform_obs(self, responses):
        # Use np.float64 instead of np.float
        img1d = np.array(responses[0].image_data_float, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])


    def _get_obs(self):
        responses = self.airsim_client.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.airsim_client.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.airsim_client.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image


    def step(self, action, verbose=True):
        # Implement action effect
        # Update drone position and orientation based on action taken
        info = {}
        self.timesteps += 1
        self.totalsteps += 1
        step_length = 0.1

        self._do_action_angle_rate(action, step_length=step_length, duration=0.25)
        
        # Calculate new state and reward
        observation = self._get_observation()
        reward, terminated = self._compute_reward_April12(action=action, verbose=verbose)

        # time.sleep(0.25)

        truncated = False
        if self.timesteps >= self.max_steps:
            truncated = True
        
        if terminated or truncated:
            self.logger.end_episode()  # End of episode, save logs
            self.episode_number += 1


        info["passed_gates"] = self.navigator.current_gate_index - 1

        return observation, reward, terminated, truncated, info

    
    def _check_positions_change(self, diff_thresh=0.2):
        # Implement this method to check if the change in positions is not remarkable
        # This is a placeholder for your logic to determine if the changes are significant
        # Example: Calculate the variance of the positions and check if it's below a threshold
        if len(self.positions) < self.max_positions_stored:
            return False  # Not enough data to decide
        
        # Example criterion: Check if the standard deviation of all x, y, z positions is below a threshold
        positions_array = np.array(self.positions)  # Convert list of positions to a NumPy array for easy processing
        position_changes = np.std(positions_array, axis=0)
        threshold = np.array([diff_thresh, diff_thresh, diff_thresh])  # Example threshold for x, y, z changes
        return np.all(position_changes < threshold)



    def _get_observation(self):
        # Return observation
        # For simplicity, let's say it's the drone's position and orientation, and the next gate's position
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        self.gate_index = self.navigator.current_gate_index
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        drone_position = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        drone_orientation = np.array([state.orientation.w_val, state.orientation.x_val, state.orientation.y_val, state.orientation.z_val])
        gate_orientation = np.array([gate_state.orientation.w_val, gate_state.orientation.x_val, gate_state.orientation.y_val, gate_state.orientation.z_val])

        position_distance = np.linalg.norm(drone_position - gate_position)
        orientation_distance = quaternion_orientational_distance(drone_orientation, gate_orientation)


        observation = np.array([self.previous_position[0], self.previous_position[1], self.previous_position[2],
                                drone_position[0], drone_position[1], drone_position[2],
                                drone_orientation[0], drone_orientation[1], drone_orientation[2], drone_orientation[3],
                                gate_position[0], gate_position[1], gate_position[2],
                                gate_orientation[0], gate_orientation[1], gate_orientation[2], gate_orientation[3],
                                self.previous_distance, position_distance, orientation_distance])
        
        # observation = self.get_image()


        return observation

    
    
    def calculate_threshold(self, threshold_coeff=2.5):
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])

        if (self.gate_index - 1) < 0:
            threshold = np.linalg.norm(self.home_position - gate_position) + threshold_coeff
        else:
            previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
            previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
            threshold = np.linalg.norm(previous_gate_position - gate_position) + threshold_coeff

        return threshold



    def _compute_reward_April12(self, action, threshold_coeff=5.0, verbose=False):
        done = False

        state = self.airsim_client.getMultirotorState()
        self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
        drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        current_distance = np.linalg.norm(drone_position - gate_position)
        distance_difference = self.previous_distance - current_distance
        drone_velocity_norm = np.linalg.norm(drone_velocity)

        # Update positions list with current position
        current_position = list(drone_position)
        self.positions.append(current_position)
        if len(self.positions) > self.max_positions_stored:
            self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

        self.distance_history.append(current_distance)
        if len(self.distance_history) > self.max_history_length:
            self.distance_history.pop(0)


        optimal_speed = 1.5  # Define an optimal speed that is safe but effective
        alignment_scale = 5

        speed_reward = -abs(drone_velocity_norm - optimal_speed)  # Penalize deviation from the optimal speed

        # Adjust current distance reward to be less aggressive
        distance_reward_coefficient = -0.1  # Less severe than previous
        reward = (current_distance * distance_reward_coefficient) + speed_reward + distance_difference

        alignment_reward_axis0 = self.calculate_alignment_reward(drone_position, gate_position, state.kinematics_estimated.orientation, alignment_scale, 0)
        # alignment_reward_axis1 = self.calculate_alignment_reward(drone_position, gate_position, state.kinematics_estimated.orientation, alignment_scale, 1)
        # alignment_reward_axis2 = self.calculate_alignment_reward(drone_position, gate_position, state.kinematics_estimated.orientation, alignment_scale, 2)

        reward += alignment_reward_axis0


        movement_reward = self.update_reward_for_movement()
        reward += movement_reward


        if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):  # Track completed
            done = True
            reward = 100.0  # Completion bonus
            print('Track completed!')
            return reward, done
        
        if self.gate_index > self.prev_gate_index:  # A gate has been passed
            reward = 20.0  # Reward for passing a gate plus movement reward
            self.prev_gate_index = np.copy(self.gate_index)

        # Collision or out of bounds
        if state.collision.has_collided or drone_position[2] < -1.0 or drone_position[2] > 3.5:
            reward -= 5.0  # Severe penalty for collision or out of bounds
            # print('Collision')

        # Out of track
        if current_distance >= self.calculate_threshold(threshold_coeff):
            done = True
            reward = -10.0  # Severe penalty for being out of track
            # print('Out of track')

        # Stuck detection
        if self.consecutive_stuck_steps >= self.stuck_threshold:
            done = True
            reward -= 10.0  # Significant penalty for being stuck enough to end the episode
            # print("Terminating episode due to being stuck.")


        # Reward based on making continual progress towards the gate, penalize early for lack of movement
        if len(self.positions) >= self.max_positions_stored and not self._check_positions_change():
            incremental_stuck_penalty = -1  # Apply smaller penalties earlier
            reward += incremental_stuck_penalty

        if verbose:
            print(f"Episode: {self.episode_number} Step: {self.timesteps}/{self.totalsteps} Reward: {reward:.3f} Velocity: {drone_velocity_norm:.3f} Distance: {current_distance:.3f} Distance Difference: {distance_difference:.3f} Alignment: {alignment_reward_axis0:.3f} Movement: {movement_reward:.3f} Target Gate: {self.gate_index}")
            
        self.previous_distance = np.copy(current_distance)
        self.previous_position = np.copy(drone_position)

        self.logger.log_step(drone_position, reward)

        return reward, done


    def calculate_alignment_reward(self, drone_position, gate_position, drone_orientation, scale=5, axis=1):
        # Calculate the vector from drone to gate
        vector_to_gate = gate_position - drone_position
        vector_to_gate_norm = np.linalg.norm(vector_to_gate)
        if vector_to_gate_norm == 0:
            print("Warning: Drone position is identical to gate position.")
            return 0  # Avoid division by zero
        vector_to_gate /= vector_to_gate_norm

        # Calculate the facing direction of the drone
        facing_vector = self.get_vector_from_quaternion(quat = drone_orientation, axis = axis)
        facing_vector_norm = np.linalg.norm(facing_vector)
        if facing_vector_norm == 0:
            print("Warning: Facing vector norm is zero.")
            return 0  # Avoid division by zero
        facing_vector /= facing_vector_norm

        # Dot product to determine alignment
        alignment = np.dot(vector_to_gate, facing_vector)
        alignment_reward = scale * alignment  # Scale as needed

        # Debug output
        # print(f"Vector to gate: {vector_to_gate}, Facing vector: {facing_vector}, Alignment: {alignment}")

        return alignment_reward

    def update_reward_for_movement(self):
        """Updates the reward based on the drone's movement and progress."""
        reward = 0
        if len(self.positions) >= self.max_positions_stored:
            if self._check_positions_change():
                # If positions haven't changed sufficiently, assume potential stuck condition
                self.consecutive_stuck_steps += 1
                incremental_stuck_penalty = -0.5 * self.consecutive_stuck_steps  # Increase penalty with time stuck
                reward += incremental_stuck_penalty
            else:
                self.consecutive_stuck_steps = 0  # Reset if there's been adequate movement
        return reward


    # loads desired level
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=3):
        self.airsim_client.simStartRace(tier)

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        self.airsim_client.simResetRace()

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )

        self.airsim_client.setTrajectoryTrackerGains(
            traj_tracker_gains, vehicle_name=self.drone_name
        )
        time.sleep(0.2)

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        takeoff_waypoint = airsim.Vector3r(
            start_position.x_val,
            start_position.y_val,
            start_position.z_val - takeoff_height,
        )

        self.airsim_client.moveOnSplineAsync(
            [takeoff_waypoint],
            vel_max=15.0,
            acc_max=5.0,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        ).join()

    def get_vector_from_quaternion(self, quat, axis=1, scale=1.0):

        # convert gate quaternion to rotation matrix.
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array(
            [
                quat.w_val,
                quat.x_val,
                quat.y_val,
                quat.z_val,
            ],
            dtype=np.float64,
        )
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsim.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )
        # gate_facing_vector = rotation_matrix[:, 1]
        gate_facing_vector = rotation_matrix[:, axis]

        return np.array([scale * gate_facing_vector[0],
                         scale * gate_facing_vector[1],
                         scale * gate_facing_vector[2]])

        # return airsim.Vector3r(
        #     scale * gate_facing_vector[0],
        #     scale * gate_facing_vector[1],
        #     scale * gate_facing_vector[2],
        # )
    
    def get_gate_horizontal_normal(self, gate_facing_vector):
        # Zero out the Y-component to project the vector onto the horizontal plane
        horizontal_normal = np.array([gate_facing_vector.x_val, 0, gate_facing_vector.z_val])
        # Normalize the resulting vector
        norm = np.linalg.norm(horizontal_normal)
        if norm < np.finfo(float).eps:
            # Avoid division by zero if the normal is too small
            return airsim.Vector3r(1.0, 0.0, 0.0)
        horizontal_normal /= norm
        return airsim.Vector3r(horizontal_normal[0], horizontal_normal[1], horizontal_normal[2])
    
    # stores gate ground truth poses as a list of airsim.Pose() objects in self.gate_poses_ground_truth
    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        self.gate_poses_ground_truth = []
        self.gate_pos_list = []            

        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (
                math.isnan(curr_pose.position.x_val)
                or math.isnan(curr_pose.position.y_val)
                or math.isnan(curr_pose.position.z_val)
            ) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(
                curr_pose.position.x_val
            ), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.y_val
            ), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.z_val
            ), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)

            self.gate_pos_list.append([curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val])

        self.gate_pos_list = np.array(self.gate_pos_list)

        


        # gate0 = gate_names_sorted[0]
        # curr_pose = self.airsim_client.simGetObjectPose(gate0)
        # print ('curr_pose0: ', curr_pose.position)
        # curr_pose = airsim.Pose(airsim.Vector3r(curr_pose.position.x_val, curr_pose.position.y_val, curr_pose.position.z_val-5.0))
        # self.airsim_client.simSetObjectPose(gate0, curr_pose, teleport=False)

        # curr_pose = self.airsim_client.simGetObjectPose(gate0)
        # print ('curr_pose1: ', curr_pose.position)



    def fly_through_all_gates_at_once_with_moveOnSplineVelConstraints(self):
        if self.level_name in [
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
        ]:
            vel_max = 15.0
            acc_max = 7.5
            speed_through_gate = 2.5

        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0
            speed_through_gate = 1.0

        return self.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position for gate_pose in self.gate_poses_ground_truth],
            [
                self.get_vector_from_quaternion(
                    gate_pose.orientation, scale=speed_through_gate
                )
                for gate_pose in self.gate_poses_ground_truth
            ],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=True,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        # img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if self.viz_image_cv2:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)
        
        # print ("img size: ", img_rgb.shape)
        if self.save_img:
            filename = self.img_folder + "/img_" + str(self.img_index) + ".png"
            # print (filename)
            cv2.imwrite(filename, img_rgb) 
            self.img_index += 1    

    def get_image(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        # img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(3, response[0].height, response[0].width)

        return img_rgb

    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState().kinematics_estimated
        drone_position = np.array([drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val])
        drone_orientation = np.array([drone_state.orientation.w_val, drone_state.orientation.x_val, drone_state.orientation.y_val, drone_state.orientation.z_val])

        # Update with drone's position (should be done continuously in your update loop)
        self.navigator.update_drone_position(drone_position)
        # Get the current target gate
        current_gate = self.navigator.get_current_target_gate()


        # print ('')
        check_collision = self.airsim_client_odom.getMultirotorState().collision.has_collided
        # print('collision: ', check_collision)
        # print("Current target gate index:", self.navigator.current_gate_index)


        index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
        gate = self.gate_poses_ground_truth[index]
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
        distance = float(np.linalg.norm(gate_position - drone_position))


        # in world frame:
        position = drone_state.position
        orientation = drone_state.orientation
        linear_velocity = drone_state.linear_velocity
        angular_velocity = drone_state.angular_velocity

    # call task() method every "period" seconds.
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            # print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")
