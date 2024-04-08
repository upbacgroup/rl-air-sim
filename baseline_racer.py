from argparse import ArgumentParser
import airsimdroneracinglab as airsim
import cv2
import threading
import time
import utils
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import time


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
    
    def update_drone_position(self, drone_position, threshold=0.25):
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
        if self.reached_gate and distance_to_gate > threshold:
            self.current_gate_index += 1  # Move to the next gate
            self.reached_gate = False  # Reset the reached_gate flag
            if self.current_gate_index < len(self.gates_positions):
                print(f"Switched to gate {self.current_gate_index + 1}.")
            else:
                print("All gates have been passed.")
    
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
class BaselineRacer(object):
    def __init__(
        self,
        drone_name="drone_1",
        viz_traj=True,
        viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0],
        viz_image_cv2=True,
    ):
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba
        
        self.navigator = None

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()

        self.home_position = np.array([self.airsim_client.getMultirotorState().kinematics_estimated.position.x_val, self.airsim_client.getMultirotorState().kinematics_estimated.position.y_val, 
                                       self.airsim_client.getMultirotorState().kinematics_estimated.position.z_val])
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03)
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

        self.prev_gate_index = 0
        self.positions = []
        self.max_positions_stored = 50

        self.previous_distance = 10
        self.previous_position = 0

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

    # def _compute_new_reward(self, threshold=10.0):
    #     state = self.airsim_client_odom.getMultirotorState(vehicle_name="drone_1")
    #     self.gate_index = self.navigator.current_gate_index
    #     gate_state = self.gate_poses_ground_truth[self.gate_index]
    #     drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
    #     gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
    #     drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
    #     position_distance = np.linalg.norm(drone_position - gate_position)
    #     done = False

    #     # print (f"position_distance: {position_distance:.4}")

    #     dist = 1000
    #     minGate = 0
    #     beta = 1
    #     thresh_dist = 5
    #     minimum_dist = 0.5
    #     distCurrent = dist

    #     if self.gate_index == len(self.gate_poses_ground_truth): # track completed
    #         done = True
    #         reward = 100
    #         print ('Track has been completed. Congrats!')

        
    #     if state.collision.has_collided:
    #         print ('Collision')
    #         reward = -1.0


    #     for i in range(0, len(self.gate_pos_list) - 1):
            
    #         dist = min(
    #             dist,
    #             np.linalg.norm(np.cross((drone_position - self.gate_pos_list[i]), (drone_position - self.gate_pos_list[i + 1])))
    #             / np.linalg.norm(self.gate_pos_list[i] - self.gate_pos_list[i + 1]),
    #         )

    #         if(dist < distCurrent):
    #             distCurrent = dist
    #             minGate = i


    #     # print("Distance the closest gate and its distance = ",minGate,dist,"\n")

    #     if(self.gate_index != minGate and ( abs(self.gate_index-minGate) > 1 )):
    #         reward = -1.0
    #         print("Moving wrong gate direction ! I should be going ",self.gate_index ,"but im going ",minGate,"\n")
    #         done = True
    #         # Moving direction of wrong gate shit thing , it shouldnt go !
    #     elif position_distance < minimum_dist: #distCurrent < minimum_dist: #means passed gate
    #         reward = 10.0
    #         print ("Passed the gate ", self.gate_index)
    #         # print("If deletion of Gate is going to be it?\n")
    #         #reward += 100 #150 total?
    #         # print("before gate it should go deletion and len",self.gate_index ,"\n")
    #         #pts = np.delete(pts,minGate)
    #         self.gate_index += 1
    #         self.distance_last_step = position_distance
    #         # print("after gate it should go deletion and len",self.gate_index ,"\n")

    #         """if(isPassedGate(current_position,pts[minGate]) == True):
    #             print("If deletion of Gate is going to be it?\n")
    #             reward += 100 #150 total?
    #             print("before gate deletion and len",pts,len(pts),"\n")
    #             #pts = np.delete(pts,minGate)
    #             currentGateIndexItShouldMoveFor += 1
    #             print("after gate deletion and len",pts,len(pts),"\n")
    #             #global disari cikiyor mu"""
        
    #     elif self.distance_last_step > distCurrent:
    #         #it get closed to target gate
    #         print("Close to target gate")
    #         reward = 3 
    #         if(distCurrent < thresh_dist):
    #             reward  += 1
    #     elif distCurrent < thresh_dist:
    #         reward = 1
    #     else:
    #         # print("Distance is smaller than threst distance but not good as minimum_dist \n")
    #         reward_dist = (math.exp(-beta*dist) - 0.5) 
    #         reward_speed = (np.linalg.norm(drone_velocity) - 0.5)
    #         reward = reward_dist + reward_speed

    #     self.distance_last_step = distCurrent        

        
    #     # else:
    #     #     if self.gate_index > self.prev_gate_index:
    #     #         reward = 100.0
    #     #         self.prev_gate_index = np.copy(self.gate_index)
    #     #     else:
    #     #         reward = np.clip(1 / (0.1 + position_distance), 0.0, 10)

        

    #     # if position_distance >= threshold: # drone out of track
    #     #     done = True
    #     #     reward = -100.0

    #     return reward, done
    
    def _compute_reward(self, threshold=10.0):
        state = self.airsim_client_odom.getMultirotorState()
        self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
        drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        position_distance = np.linalg.norm(drone_position - gate_position)
        done = False

        # Your existing setup code...
        gate_direction = gate_position - drone_position
        gate_direction_norm = np.linalg.norm(gate_direction)
        gate_direction_unit = gate_direction / gate_direction_norm if gate_direction_norm > 0 else gate_direction
        
        # Project velocity onto the direction towards the gate
        velocity_towards_gate = np.dot(drone_velocity, gate_direction_unit)

        print ('velocity_towards_gate: ', velocity_towards_gate)

        # print (f"position_distance: {position_distance:.4}")
        
        if state.collision.has_collided:
            print ('Collision')
            reward = -10.0
        elif drone_position[2] > 3.5: # if drone hits the ground
            done = True
            reward = -10.0
        else:
            if self.gate_index > self.prev_gate_index:
                reward = 25.0
                self.prev_gate_index = np.copy(self.gate_index)
                print (self.gate_index,' Gate has been passed')
            else:
                # reward = np.clip(1 / (0.1 + position_distance), 0.0, 10)
                velocity_diff = np.linalg.norm(drone_velocity) - 0.5
                reward = np.clip(-0.1 * position_distance, -10, 0) + velocity_diff

        if self.navigator.current_gate_index == len(self.gate_poses_ground_truth): # track completed
            done = True
            reward = 100.0
            print ('Track completed!')


        # if (self.gate_index+1) < len(self.gate_poses_ground_truth):
        #     next_gate_state = self.gate_poses_ground_truth[self.gate_index + 1]
        #     next_gate_position = np.array([next_gate_state.position.x_val, next_gate_state.position.y_val, next_gate_state.position.z_val])
        #     threshold = np.linalg.norm(next_gate_position - gate_position) + 2.5
            
        if (self.gate_index - 1) < 0:
            threshold = np.linalg.norm(self.home_position - gate_position) + 2.5
        else:
            previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
            previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
            threshold = np.linalg.norm(previous_gate_position - gate_position) + 2.5      

        if position_distance >= threshold : # drone out of track
            done = True
            reward = -10.0
            print(f'Drone is out of track. Distance: {position_distance:.4f} Threshold: {threshold:.4f}')


        return reward, done
    
    def _compute_reward_gpt(self, threshold_coeff=5.0):
        state = self.airsim_client_odom.getMultirotorState()
        self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
        drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        position_distance = np.linalg.norm(drone_position - gate_position)
        done = False
        threshold = 0  # Initialize threshold

        # # Reward for moving towards the gate
        # velocity_towards_gate = np.dot(drone_velocity, gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
        # reward_movement = np.clip(velocity_towards_gate * 0.1, -10, 10)  # Encourage forward movement, penalize backward or very slow movement

        # Reward for moving towards the gate
        velocity_towards_gate = np.dot(drone_velocity, gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
        velocity_towards_gate = np.clip(velocity_towards_gate * 0.1, -10, 10)  # Encourage forward movement, penalize backward or very slow movement


        direction_to_gate = (gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
        alignment_with_gate = np.dot(drone_velocity / (np.linalg.norm(drone_velocity) + 1e-6), direction_to_gate)
        velocity_alignment_gate = np.clip(alignment_with_gate * 10, -10, 10)  # Scale based on alignment, not velocity magnitude


        # Update positions list with current position
        current_position = list(drone_position)
        self.positions.append(current_position)
        if len(self.positions) > self.max_positions_stored:
            self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

        if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):  # Track completed
            done = True
            reward = 100.0  # Completion bonus
            print('Track completed!')

        else:
            if self.gate_index > self.prev_gate_index:  # A gate has been passed
                reward = 15.0  # Reward for passing a gate plus movement reward
                self.prev_gate_index = np.copy(self.gate_index)
                print(f'{self.gate_index} Gate has been passed')
            else:
                # Adjust reward based on distance to next gate and velocity towards it
                reward = np.clip(1 / (position_distance + 0.1), 0, 10) + velocity_towards_gate

            if (self.gate_index - 1) < 0:
                threshold = np.linalg.norm(self.home_position - gate_position) + threshold_coeff
            else:
                previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
                previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
                threshold = np.linalg.norm(previous_gate_position - gate_position) + threshold_coeff

            # Collision or out of bounds
            if state.collision.has_collided or drone_position[2] > 3.5 or drone_position[2] < 0.0:
                done = True
                reward = -100.0  # Severe penalty for collision or out of bounds
                print('Collision, or out of bounds')

            # Out of track
            if position_distance >= threshold:
                done = True
                reward = -100.0  # Severe penalty for being out of track
                print('Out of track')

            # Stuck detection
            if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
                done = True
                reward = -100.0  # Severe penalty for being stuck
                print("Stuck at a location")

        # print(f"Reward: {reward:.5f} Reward Movement: {velocity_towards_gate:.5f} Z: {current_position[2]:.4f} Target: {self.navigator.current_gate_index} Distance: {position_distance:.5f}")
        print(f"Reward: {reward:.5f} Velocity to Gate: {velocity_towards_gate:.5f} Velocity Alignment: {velocity_alignment_gate:.5f} Target Gate: {self.gate_index} Distance: {position_distance:.5f}")


        return reward, done
    

    def _compute_reward_28March(self, threshold_coeff=2.5):
        state = self.airsim_client_odom.getMultirotorState()
        self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
        drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        current_distance = np.linalg.norm(drone_position - gate_position)
        done = False

        # Update positions list with current position
        current_position = list(drone_position)
        self.positions.append(current_position)
        if len(self.positions) > self.max_positions_stored:
            self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

        
        # self.distance_history.append(current_distance)
        # if len(self.distance_history) > self.max_history_length:
        #     self.distance_history.pop(0)
            

        reward = np.clip(self.previous_distance - current_distance, -100, 100)
        if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):  # Track completed
            done = True
            reward = 100.0  # Completion bonus
            print('Track completed!')
            return reward, done
        
        if self.gate_index > self.prev_gate_index:  # A gate has been passed
            reward = 50.0  # Reward for passing a gate plus movement reward
            self.prev_gate_index = np.copy(self.gate_index)
            # print(f'{self.gate_index} Gate has been passed')
        # else:
        #     # Adjust reward based on distance to next gate and velocity towards it
        #     reward = np.clip(1 / (position_distance + 0.1), 0, 10) + velocity_towards_gate

        # Collision or out of bounds
        if state.collision.has_collided or drone_position[2] > 3.5 or drone_position[2] < -1.5:
            reward += -2.5  # Severe penalty for collision or out of bounds
            # print('Collision, or out of bounds')

        
        # hover_threshold = 0.5  # Velocity threshold to consider the drone as hovering
        # if np.linalg.norm([drone_velocity[0], drone_velocity[1]]) < hover_threshold:
        #     reward -= 2.0  # Penalize hovering
        #     # print('Penalizing hover state')

        # Out of track
        # if current_distance >= self.calculate_threshold(threshold_coeff):
        #     done = True
        #     reward = -50.0  # Severe penalty for being out of track
            # print('Out of track')

        # Stuck detection
        if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
            done = True
            reward = -50.0  # Severe penalty for being stuck
            print("Stuck at a location")

        print(f"Reward: {reward:.5f} Target Gate: {self.gate_index} Distance: {current_distance:.5f}")
            
        self.previous_distance = np.copy(current_distance)

        return reward, done
    
    

    def _check_positions_change(self):
        # Implement this method to check if the change in positions is not remarkable
        # This is a placeholder for your logic to determine if the changes are significant
        # Example: Calculate the variance of the positions and check if it's below a threshold
        if len(self.positions) < self.max_positions_stored:
            return False  # Not enough data to decide
        
        # Example criterion: Check if the standard deviation of all x, y, z positions is below a threshold
        positions_array = np.array(self.positions)  # Convert list of positions to a NumPy array for easy processing
        position_changes = np.std(positions_array, axis=0)
        threshold = np.array([0.25, 0.25, 0.25])  # Example threshold for x, y, z changes
        return np.all(position_changes < threshold)
    

    def _compute_reward_12march(self, threshold_coeff=5.0):
        state = self.airsim_client_odom.getMultirotorState()
        self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
        gate_state = self.gate_poses_ground_truth[self.gate_index]
        drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
        drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
        gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
        position_distance = np.linalg.norm(drone_position - gate_position)
        done = False


        # Update positions list with current position
        current_position = list(drone_position)
        self.positions.append(current_position)
        if len(self.positions) > self.max_positions_stored:
            self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

        
        if self.navigator.current_gate_index == len(self.gate_poses_ground_truth): # track completed
            done = True
            reward = 100.0
            print ('Track completed!')
            
        else:
            if self.gate_index > self.prev_gate_index:
                reward = 10.0
                self.prev_gate_index = np.copy(self.gate_index)
                print (self.gate_index,' Gate has been passed')
            else:
                # reward = np.clip(1 / (0.1 + position_distance), 0.0, 10)
                velocity_diff = np.linalg.norm(drone_velocity) # - 0.5
                # reward = np.clip(-1 * position_distance, -10, 0) + velocity_diff
                reward = np.clip(1 / (position_distance + 0.1), 0, 10) + velocity_diff


            if (self.gate_index - 1) < 0:
                threshold = np.linalg.norm(self.home_position - gate_position) + threshold_coeff
            else:
                previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
                previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
                threshold = np.linalg.norm(previous_gate_position - gate_position) + threshold_coeff          
                # print(f'Drone is out of track. Distance: {position_distance:.4f} Threshold: {threshold:.4f}')

            
            if state.collision.has_collided or drone_position[2] > 3.5: # if drone collides to the gate or hits to the ground
                print ('Collision')
                reward += -10.0

            if position_distance >= threshold : # if drone is out of track
                done = True
                reward = -100.0
                print ("drone is out of track")

            # Check for significant change in the last position measurements 
            # if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
            #     done = True  # Trigger environment reset by indicating the episode is done
            #     reward = -100.0
            #     print ("stuck at a location")

            #     for pos in self.positions:
            #         print (pos)
            #     stop

        return reward, done
    
    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height=2.0):
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

    # this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints()
    # the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale=1.0):
        # convert gate quaternion to rotation matrix.
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array(
            [
                airsim_quat.w_val,
                airsim_quat.x_val,
                airsim_quat.y_val,
                airsim_quat.z_val,
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
        gate_facing_vector = rotation_matrix[:, 1]
        return airsim.Vector3r(
            scale * gate_facing_vector[0],
            scale * gate_facing_vector[1],
            scale * gate_facing_vector[2],
        )
    
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


    def fly_through_all_gates_one_by_one_with_moveOnSpline(self):
        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0

        if self.level_name in [
            "Soccer_Field_Medium",
            "Soccer_Field_Easy",
            "ZhangJiaJie_Medium",
        ]:
            vel_max = 10.0
            acc_max = 5.0

        return self.airsim_client.moveOnSplineAsync(
            [gate_pose.position],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_at_once_with_moveOnSpline(self):
        if self.level_name in [
            "Soccer_Field_Medium",
            "Soccer_Field_Easy",
            "ZhangJiaJie_Medium",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ]:
            vel_max = 30.0
            acc_max = 15.0

        if self.level_name == "Building99_Hard":
            vel_max = 4.0
            acc_max = 1.0

        return self.airsim_client.moveOnSplineAsync(
            [gate_pose.position for gate_pose in self.gate_poses_ground_truth],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints(self):
        add_velocity_constraint = True
        add_acceleration_constraint = False

        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy"]:
            vel_max = 15.0
            acc_max = 3.0
            speed_through_gate = 2.5

        if self.level_name == "ZhangJiaJie_Medium":
            vel_max = 10.0
            acc_max = 3.0
            speed_through_gate = 1.0

        if self.level_name == "Building99_Hard":
            vel_max = 2.0
            acc_max = 0.5
            speed_through_gate = 0.5
            add_velocity_constraint = False

        # scale param scales the gate facing vector by desired speed.
        return self.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position],
            [
                self.get_gate_facing_vector_from_quaternion(
                    gate_pose.orientation, scale=speed_through_gate
                )
            ],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=add_velocity_constraint,
            add_acceleration_constraint=add_acceleration_constraint,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

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
                self.get_gate_facing_vector_from_quaternion(
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
    

    def quaternion_to_rotation_matrix(self, q):
        # Assuming q is in the form [w, x, y, z]
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])

    def rotate_vector_by_quaternion(self, v, q):
        rotation_matrix = self.quaternion_to_rotation_matrix(q)
        return rotation_matrix.dot(v)
    

    def project_point_onto_line(self, point, line_start, line_end):
        """Project the point onto the line defined by line_start and line_end."""
        line_vec = line_end - line_start
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        point_vec = point - line_start
        projection_length = np.dot(point_vec, line_vec_norm)
        return line_start + projection_length * line_vec_norm

    def calculate_progress_reward(self, current_position, previous_position, gate1_position, gate2_position):
        """Calculate the progress reward based on the current and previous positions of the drone."""
        # Project the current and previous positions onto the line segment
        current_projection = self.project_point_onto_line(current_position, gate1_position, gate2_position)
        previous_projection = self.project_point_onto_line(previous_position, gate1_position, gate2_position)
        
        # Compute the progress along the path segment
        current_progress = np.dot(current_position - gate1_position, gate2_position - gate1_position) / np.linalg.norm(gate2_position - gate1_position)
        previous_progress = np.dot(previous_position - gate1_position, gate2_position - gate1_position) / np.linalg.norm(gate2_position - gate1_position)
        
        # The progress reward is the difference in the progress
        rp = current_progress - previous_progress
        return rp
    

    def calculate_safety_reward(self, drone_position, gate):
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
        gate_normal = self.get_gate_facing_vector_from_quaternion(gate.orientation)
        # dp is the horizontal distance from the drone to the gate's center, projected onto gate_normal_horizontal
        gate_normal_horizontal = self.get_gate_horizontal_normal(gate_normal)
        gate_normal_horizontal = np.array([gate_normal_horizontal.x_val, gate_normal_horizontal.y_val, gate_normal_horizontal.z_val])
        gate_normal = np.array([gate_normal.x_val, gate_normal.y_val, gate_normal.z_val])

        norm_gate_horizontal = np.linalg.norm(gate_normal_horizontal)
        norm_gate = np.linalg.norm(gate_normal)


        dp_vector = drone_position - gate_position
        dp_vector[1] = 0  # Remove the Y component for horizontal projection
        dp = np.linalg.norm(dp_vector - np.dot(dp_vector, gate_normal_horizontal) * gate_normal_horizontal)

        # dn is the vertical distance from the drone to the gate plane
        dn = np.abs(np.dot(drone_position - gate_position, gate_normal))

        w_g = 1
        d_max = 0.5

        # Calculate f based on dp and d_max
        f = max(1 - (dp / d_max), 0)

        # Calculate v based on f and w_g
        v = max((1 - f) * (w_g / 6), 0.05)

        # Calculate the safety reward rs
        safety_reward = -f**2 * (1 - np.exp(-0.5 * dn**2 / v))

        return safety_reward


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


    

    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState().kinematics_estimated
        drone_position = np.array([drone_state.position.x_val, drone_state.position.y_val, drone_state.position.z_val])
        drone_orientation = np.array([drone_state.orientation.w_val, drone_state.orientation.x_val, drone_state.orientation.y_val, drone_state.orientation.z_val])

        # Update with drone's position (should be done continuously in your update loop)
        self.navigator.update_drone_position(drone_position)
        # Get the current target gate
        current_gate = self.navigator.get_current_target_gate()

        reward, done = self._compute_reward_gpt()
        # reward, done = self._compute_reward()

        # print ('Current reward: ', reward, ' done: ', done)
        # time.sleep(0.01)
        check_collision = self.airsim_client_odom.getMultirotorState().collision.has_collided
        # print('collision: ', check_collision)
        # print("Current target gate index:", self.navigator.current_gate_index)

        gate_index = self.navigator.current_gate_index
        gate = self.gate_poses_ground_truth[gate_index]
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])

        if (gate_index - 1) < 0:
            previous_gate_position = np.copy(self.home_position)
        else:
            previous_gate_state = self.gate_poses_ground_truth[gate_index - 1]
            previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
        

        progress_reward = self.calculate_progress_reward(drone_position, self.previous_position, previous_gate_position, gate_position)
        safety_reward = self.calculate_safety_reward(drone_position, gate)

        print (f"progress: {progress_reward:.4f} safety: {safety_reward:.3f}")
        # time.sleep(0.1)

        self.previous_position = np.copy(drone_position)
        
        # print (f"dn: {norm_gate:.3f} x: {gate_normal.x_val:.3f} y: {gate_normal.y_val:.3f} z: {gate_normal.z_val:.3f}")
        # print(f"Safety reward: {rs}")

        # print (f"gate normal: norm: {norm_gate:.3f} x: {gate_normal[0]:.3f} y: {gate_normal[1]:.3f} z: {gate_normal[2]:.3f}")
        # print (f"gate normal horiz: norm: {norm_gate_horizontal:.3f} x: {gate_normal_horizontal[0]:.3f} y: {gate_normal_horizontal[1]:.3f} z: {gate_normal_horizontal[2]:.3f}")
        
        # up_vector = np.array([0, 1, 0])  # 'Up' vector in many 3D coordinate systems

        # # Rotate the vector by the quaternion
        # gate_normal = self.rotate_vector_by_quaternion(up_vector, gate.orientation)
        # print (f"gate normal: norm: {np.linalg.norm(gate_normal):.3f} x: {gate_normal[0]:.3f} y: {gate_normal[1]:.3f} z: {gate_normal[2]:.3f}")


        # for index in range(len(self.gate_poses_ground_truth)):
            
        #     gate_orientation = np.array([gate.orientation.w_val, gate.orientation.x_val, gate.orientation.y_val, gate.orientation.z_val])
        #     distance = np.linalg.norm(gate_position - drone_position)
        #     drone_angle = self.get_gate_facing_vector_from_quaternion(drone_state.orientation)
        #     gate_angle = self.get_gate_facing_vector_from_quaternion(gate.orientation)
        #     angle_diff = calculate_angle_difference(drone_angle, gate_angle)
        #     quat_diff = quaternion_orientational_distance(drone_orientation, gate_orientation)

        #     # Calculate the drone forward vector
        #     local_forward_vector = np.array([1, 0, 0])  # Assuming forward direction is along the local x-axis
        #     rotation = R.from_quat(drone_orientation)
        #     drone_forward = rotation.apply(local_forward_vector)

        #     # Calculate the vector from the drone to the gate
        #     drone_to_gate = gate_position - drone_position
        #     # print(f"Orientational Distance (radians): {theta}")
        #     # print(f"Orientational Distance (degrees): {np.degrees(theta)}")

        #     # print (f"gate {index}  pos dist: {distance:.4}, quat diff: {quat_diff:.4}, angle diff: {angle_diff:.4}")

        # # in world frame:
        # position = drone_state.position
        # orientation = drone_state.orientation
        # linear_velocity = drone_state.linear_velocity
        # angular_velocity = drone_state.angular_velocity

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
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")



def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer = BaselineRacer(
        drone_name="drone_1",
        viz_traj=args.viz_traj,
        viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
        viz_image_cv2=args.viz_image_cv2,
    )
    baseline_racer.load_level(args.level_name)
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3
    baseline_racer.start_race(args.race_tier)
    baseline_racer.initialize_drone()
    baseline_racer.takeoff_with_moveOnSpline()
    baseline_racer.get_ground_truth_gate_poses()

    baseline_racer.navigator = GateNavigator(baseline_racer.gate_poses_ground_truth)

    baseline_racer.start_image_callback_thread()
    baseline_racer.start_odometry_callback_thread()

    

    if args.planning_baseline_type == "all_gates_at_once":
        if args.planning_and_control_api == "moveOnSpline":
            baseline_racer.fly_through_all_gates_at_once_with_moveOnSpline().join()
        if args.planning_and_control_api == "moveOnSplineVelConstraints":
            baseline_racer.fly_through_all_gates_at_once_with_moveOnSplineVelConstraints().join()

    if args.planning_baseline_type == "all_gates_one_by_one":
        if args.planning_and_control_api == "moveOnSpline":
            baseline_racer.fly_through_all_gates_one_by_one_with_moveOnSpline().join()
        if args.planning_and_control_api == "moveOnSplineVelConstraints":
            baseline_racer.fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints().join()

    # Comment out the following if you observe the python script exiting prematurely, and resetting the race
    baseline_racer.stop_image_callback_thread()
    baseline_racer.stop_odometry_callback_thread()
    baseline_racer.reset_race()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--level_name",
        type=str,
        choices=[
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
            "Building99_Hard",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ],
        default="Soccer_Field_Easy",
    )
    parser.add_argument(
        "--planning_baseline_type",
        type=str,
        choices=["all_gates_at_once", "all_gates_one_by_one"],
        default="all_gates_at_once",
    )
    parser.add_argument(
        "--planning_and_control_api",
        type=str,
        choices=["moveOnSpline", "moveOnSplineVelConstraints"],
        default="moveOnSpline",
    )
    parser.add_argument(
        "--enable_viz_traj", dest="viz_traj", action="store_true", default=False
    )
    parser.add_argument(
        "--enable_viz_image_cv2",
        dest="viz_image_cv2",
        action="store_true",
        default=True,
    )
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(args)
