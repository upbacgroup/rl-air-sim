    
# def reset(self, seed=None):
    #     self.timesteps = 0
    #     self.positions = []
    #     self.distance_history = []
    #     self.airsim_client.reset()
    #     self.navigator = GateNavigator(self.gate_poses_ground_truth)
    #     self.scheduler = DynamicCurriculumScheduler(self.total_gates, review_frequency=0.25)
    #     self.initialize_drone()
    #     self.takeoff_with_moveOnSpline(takeoff_height=-1.5)
    #     self.airsim_client.moveToYawAsync(yaw=90).join()

    #     # gate_index = np.random.randint(0, 3)
    #     gate_index = np.random.choice(np.arange(3), p=[0.45, 0.35, 0.2])

    #     for i in range(gate_index):
    #         gate_state = self.gate_poses_ground_truth[i]
    #         gate_position = airsim.Vector3r(gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val)
    #         # print(f"First target Gate {i} x:{gate_state.position.x_val:.4f}, y:{gate_state.position.y_val:.4f}, z:{gate_state.position.z_val:.4f}")

    #         self.airsim_client.moveOnSplineAsync(
    #             [gate_position],
    #             vel_max=15.0,
    #             acc_max=5.0,
    #             add_position_constraint=True,
    #             add_velocity_constraint=False,
    #             add_acceleration_constraint=False,
    #             viz_traj=self.viz_traj,
    #             viz_traj_color_rgba=self.viz_traj_color_rgba,
    #             vehicle_name=self.drone_name,
    #         ).join()


    #     state_pos = self.airsim_client.getMultirotorState().kinematics_estimated.position
    #     self.home_position = np.array([state_pos.x_val, state_pos.y_val, state_pos.z_val])

    #     self.navigator.update_drone_position(self.home_position)


    #     self.previous_position = np.copy(self.home_position)
    #     observation = self._get_observation()

        # return observation, {}

def _compute_reward_original(self):
    thresh_dist = 7
    beta = 1

    z = -10
    pts = np.copy(self.gate_pos_list[0:6])

    

    quad_pt = np.array(
        list(
            (
                self.state["position"].x_val,
                self.state["position"].y_val,
                self.state["position"].z_val,
            )
        )
    )

    if self.state["collision"]:
        reward = -100
    else:
        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        if dist > thresh_dist:
            reward = -10
        else:
            reward_dist = math.exp(-beta * dist) - 0.5
            reward_speed = (
                np.linalg.norm(
                    [
                        self.state["velocity"].x_val,
                        self.state["velocity"].y_val,
                        self.state["velocity"].z_val,
                    ]
                )
                - 0.5
            )
            reward = reward_dist + reward_speed

    done = False
    if reward <= -10:
        done = True


    state = self.airsim_client.getMultirotorState()
    self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
    gate_state = self.gate_poses_ground_truth[self.gate_index]
    drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
    gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
    current_distance = np.linalg.norm(drone_position - gate_position)
    threshold_coeff = 2.5

    # Update positions list with current position
    current_position = list(drone_position)
    self.positions.append(current_position)
    if len(self.positions) > self.max_positions_stored:
        self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

    
    self.distance_history.append(current_distance)
    if len(self.distance_history) > self.max_history_length:
        self.distance_history.pop(0)

    # Out of track
    if current_distance >= self.calculate_threshold(threshold_coeff) or self.check_if_drifting_away():
        done = True
        reward = -50.0  # Severe penalty for being out of track
        # print (self.distance_history)
        # print('Out of track')

    # Stuck detection
    if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
        done = True
        reward = -50.0  # Severe penalty for being stuck
        # print("Stuck at a location")

    return reward, done

def _compute_reward(self, threshold_coeff=5.0):
    state = self.airsim_client.getMultirotorState()
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
        if self.gate_index > self.prev_gate_index: # a gate has been passed
            reward = 15.0
            self.prev_gate_index = np.copy(self.gate_index)
            print (self.gate_index,' Gate has been passed')
        else:
            # reward = np.clip(1 / (0.1 + position_distance), 0.0, 10)
            velocity_diff = np.linalg.norm([drone_velocity[0], drone_velocity[1]]) - 0.5
            # reward = np.clip(-1 * position_distance, -10, 0) + velocity_diff
            # reward = np.clip(1 / (position_distance + 0.1), 0, 10) + velocity_diff
            reward = np.clip(1 / (position_distance + 0.1), 0, 10) + velocity_diff


        if (self.gate_index - 1) < 0:
            threshold = np.linalg.norm(self.home_position - gate_position) + threshold_coeff
        else:
            previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
            previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
            threshold = np.linalg.norm(previous_gate_position - gate_position) + threshold_coeff          
            # print(f'Drone is out of track. Distance: {position_distance:.4f} Threshold: {threshold:.4f}')

        
        if state.collision.has_collided or drone_position[2] > 3.5 or drone_position[2] < 0.0: # if drone collides to the gate or hits to the ground
            # print ('Collision, or taking off too much, or hitting to the ground')
            reward += -10.0

        if position_distance >= threshold : # if drone is out of track
            done = True
            reward = -100.0
            print ('out of track')

        # Check for significant change in the last position measurements 
        if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
            done = True  # Trigger environment reset by indicating the episode is done
            reward = -100.0
            print ("stuck at a location")

    print (f"Step: {self.timesteps}/{self.totalsteps} Reward: {reward:.5f} Z: {current_position[2]:.4f} Target: {self.navigator.current_gate_index} Distance: {position_distance:.5f} Threshold: {threshold:.5f}")

    return reward, done


def _compute_reward_gpt(self, threshold_coeff=2.5):
    state = self.airsim_client.getMultirotorState()
    self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
    gate_state = self.gate_poses_ground_truth[self.gate_index]
    drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
    drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
    gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
    position_distance = np.linalg.norm(drone_position - gate_position)
    done = False
    threshold = 0  # Initialize threshold
    # proximity_threshold = 1.0

    # Reward for moving towards the gate
    velocity_towards_gate = np.dot(drone_velocity, gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
    reward_movement = np.clip(velocity_towards_gate * 0.1, -10, 10)  # Encourage forward movement, penalize backward or very slow movement


    direction_to_gate = (gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
    alignment_with_gate = np.dot(drone_velocity / (np.linalg.norm(drone_velocity) + 1e-6), direction_to_gate)
    reward_movement_2 = np.clip(alignment_with_gate * 10, -10, 10)  # Scale based on alignment, not velocity magnitude


    # Update positions list with current position
    current_position = list(drone_position)
    self.positions.append(current_position)
    if len(self.positions) > self.max_positions_stored:
        self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

    
    self.distance_history.append(position_distance)
    if len(self.distance_history) > self.max_history_length:
        self.distance_history.pop(0)
        

    if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):  # Track completed
        done = True
        reward = 100.0  # Completion bonus
        print('Track completed!')

    else:
        if self.gate_index > self.prev_gate_index:  # A gate has been passed
            reward += 50.0  # Reward for passing a gate plus movement reward
            self.prev_gate_index = np.copy(self.gate_index)
            print(f'{self.gate_index} Gate has been passed')
            self.distance_history = []
        else:
            # Adjust reward based on distance to next gate and velocity towards it
            reward = np.clip(1 / (position_distance + 0.1), 0, 10) + reward_movement

        if (self.gate_index - 1) < 0:
            threshold = np.linalg.norm(self.home_position - gate_position) + threshold_coeff
        else:
            previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
            previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
            threshold = np.linalg.norm(previous_gate_position - gate_position) + threshold_coeff

        # Collision or out of bounds
        if state.collision.has_collided or drone_position[2] > 3.5 or drone_position[2] < -1.5:
            reward += -10.0  # Severe penalty for collision or out of bounds
            print('Collision, or out of bounds')

        
        hover_threshold = 0.5  # Velocity threshold to consider the drone as hovering
        if np.linalg.norm([drone_velocity[0], drone_velocity[1]]) < hover_threshold:
            reward -= 5.0  # Penalize hovering
            print('Penalizing hover state')

        # Out of track
        if position_distance >= threshold or self.check_if_drifting_away():
            done = True
            reward = -100.0  # Severe penalty for being out of track
            # print (self.distance_history)
            print('Out of track')

        # Stuck detection
        if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
            done = True
            reward = -100.0  # Severe penalty for being stuck
            print("Stuck at a location")

    print(f"Step: {self.timesteps}/{self.totalsteps} Reward: {reward:.5f} Reward Movement: {reward_movement:.5f} Target Gate: {self.gate_index} Distance: {position_distance:.5f}")

    return reward, done

def _compute_reward_gpt_2(self, threshold_coeff=2.5, action=None, verbose=False):
    state = self.airsim_client.getMultirotorState()
    self.gate_index = np.clip(self.navigator.current_gate_index, 0, len(self.gate_poses_ground_truth) - 1)
    gate_state = self.gate_poses_ground_truth[self.gate_index]
    drone_position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val])
    drone_velocity = np.array([state.kinematics_estimated.linear_velocity.x_val, state.kinematics_estimated.linear_velocity.y_val, state.kinematics_estimated.linear_velocity.z_val])
    gate_position = np.array([gate_state.position.x_val, gate_state.position.y_val, gate_state.position.z_val])
    position_distance = np.linalg.norm(drone_position - gate_position)
    done = False

    # Reward for moving towards the gate
    velocity_towards_gate = np.dot(drone_velocity, gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
    velocity_towards_gate = np.clip(velocity_towards_gate * 0.1, -10, 10)  # Encourage forward movement, penalize backward or very slow movement


    direction_to_gate = (gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
    alignment_with_gate = np.dot(drone_velocity / (np.linalg.norm(drone_velocity) + 1e-6), direction_to_gate)
    velocity_alignment_gate = np.clip(alignment_with_gate, -10, 10)  # Scale based on alignment, not velocity magnitude


    reward = np.copy(velocity_alignment_gate)
    # Update positions list with current position
    current_position = list(drone_position)
    self.positions.append(current_position)
    if len(self.positions) > self.max_positions_stored:
        self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

    
    self.distance_history.append(position_distance)
    if len(self.distance_history) > self.max_history_length:
        self.distance_history.pop(0)
        

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

    
    hover_threshold = 0.5  # Velocity threshold to consider the drone as hovering
    if np.linalg.norm([drone_velocity[0], drone_velocity[1]]) < hover_threshold:
        reward -= 2.0  # Penalize hovering
        # print('Penalizing hover state')

    # Out of track
    if position_distance >= self.calculate_threshold(threshold_coeff) or self.check_if_drifting_away():
        done = True
        reward = -50.0  # Severe penalty for being out of track
        # print (self.distance_history)
        # print('Out of track')

    # Stuck detection
    if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
        done = True
        reward = -50.0  # Severe penalty for being stuck
        # print("Stuck at a location")

    if verbose:
        print(f"Step: {self.timesteps}/{self.totalsteps} Reward: {reward:.5f} Velocity Alignment: {velocity_alignment_gate:.5f} Target Gate: {self.gate_index} Distance: {position_distance:.5f}")

    return reward, done


def _compute_reward_2(self, threshold_coeff=5.0):
    distance_penalty_coeff = -0.1
    max_velocity_reward = 5.0
    reward_for_completing_track = 100.0
    penalty_for_leaving_track = -10.0
    penalty_for_collision = -10.0
    reward_for_passing_gate = 25.0

    state = self.airsim_client.getMultirotorState()
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
    
    # Calculate distance-based reward component
    distance_reward = -position_distance * distance_penalty_coeff  # distance_penalty_coeff to be tuned

    # Calculate velocity reward component
    velocity_reward = np.clip(velocity_towards_gate, 0, max_velocity_reward)  # max_velocity_reward to be tuned
    
    # Combine rewards
    reward = distance_reward + velocity_reward

    if (self.gate_index - 1) < 0:
        threshold = np.linalg.norm(self.home_position - gate_position) + threshold_coeff
    else:
        previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
        previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
        threshold = np.linalg.norm(previous_gate_position - gate_position) + threshold_coeff          

    # Add specific rewards/penalties for events
    if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):
        reward = reward_for_completing_track
        print ('completing the track')
    elif drone_position[2] > 3.5 or position_distance >= threshold:
        reward = penalty_for_leaving_track
        print ('leaving the track')
    elif state.collision.has_collided:
        reward = penalty_for_collision
        print ('collision')
    elif self.gate_index > self.prev_gate_index:
        reward = reward_for_passing_gate
        self.prev_gate_index = np.copy(self.gate_index)
        print ('gate has been passed!')
    
    # Adjustments for smoothness, stuck detection, etc., similar to your existing logic

    return reward, done


def _compute_reward_March28(self, action, threshold_coeff=2.5, verbose=False):
    state = self.airsim_client.getMultirotorState()
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

    
    self.distance_history.append(current_distance)
    if len(self.distance_history) > self.max_history_length:
        self.distance_history.pop(0)
        

    reward = np.clip((self.previous_distance - current_distance) * 10, -100, 100)
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
    if current_distance >= self.calculate_threshold(threshold_coeff) or self.check_if_drifting_away():
        done = True
        reward = -50.0  # Severe penalty for being out of track
        # print (self.distance_history)
        # print('Out of track')

    # Stuck detection
    if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
        done = True
        reward = -50.0  # Severe penalty for being stuck
        # print("Stuck at a location")

    if verbose:
        print(f"Step: {self.timesteps}/{self.totalsteps} Action: {action} Reward: {reward:.5f} Target Gate: {self.gate_index} Distance: {current_distance:.5f}")
        
    self.previous_distance = np.copy(current_distance)

    return reward, done

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
    current_progress = np.dot(current_projection - gate1_position, gate2_position - gate1_position) / np.linalg.norm(gate2_position - gate1_position)
    previous_progress = np.dot(previous_projection - gate1_position, gate2_position - gate1_position) / np.linalg.norm(gate2_position - gate1_position)
    
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


def _compute_reward_April2(self, action, threshold_coeff=5.0, verbose=False):
    state = self.airsim_client.getMultirotorState()
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


    if (self.gate_index - 1) < 0:
        previous_gate_position = np.copy(self.home_position)
    else:
        previous_gate_state = self.gate_poses_ground_truth[self.gate_index - 1]
        previous_gate_position = np.array([previous_gate_state.position.x_val, previous_gate_state.position.y_val, previous_gate_state.position.z_val])
    

    progress_reward = self.calculate_progress_reward(drone_position, self.previous_position, previous_gate_position, gate_position)
    safety_reward = self.calculate_safety_reward(drone_position, gate_state)

    # reward = np.clip((self.previous_distance - current_distance) * 10, -100, 100)
    reward = 10 * progress_reward + 0.1*safety_reward

    if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):  # Track completed
        done = True
        reward = 100.0  # Completion bonus
        print('Track completed!')
        return reward, done
    
    if self.gate_index > self.prev_gate_index:  # A gate has been passed
        reward = 50.0  # Reward for passing a gate plus movement reward
        self.prev_gate_index = np.copy(self.gate_index)

    # Collision or out of bounds
    if state.collision.has_collided:
        reward += -2.5  # Severe penalty for collision or out of bounds
        print('Collision')

    # Out of track
    if current_distance >= self.calculate_threshold(threshold_coeff):
        done = True
        reward = -50.0  # Severe penalty for being out of track
        # print (self.distance_history)
        # print('Out of track')

    # Stuck detection
    if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
        done = True
        reward = -50.0  # Severe penalty for being stuck
        # print("Stuck at a location")


    if verbose:
        print(f"Step: {self.timesteps}/{self.totalsteps} Action: {action} Reward: {reward:.4f} Progress: {progress_reward:.4f} Safety: {safety_reward:.4f} Target Gate: {self.gate_index} Distance: {current_distance:.4f}")
        
    self.previous_distance = np.copy(current_distance)
    self.previous_position = np.copy(drone_position)

    return reward, done


def _compute_reward_April4(self, action, threshold_coeff=5.0, verbose=False):
    state = self.airsim_client.getMultirotorState()
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


    safety_reward = self.calculate_safety_reward(drone_position, gate_state)

    # Reward for moving towards the gate
    velocity_towards_gate = np.dot(drone_velocity, gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
    velocity_towards_gate = np.clip(velocity_towards_gate * 0.1, -10, 10)  # Encourage forward movement, penalize backward or very slow movement


    direction_to_gate = (gate_position - drone_position) / (np.linalg.norm(gate_position - drone_position) + 1e-6)
    alignment_with_gate = np.dot(drone_velocity / (np.linalg.norm(drone_velocity) + 1e-6), direction_to_gate)
    velocity_alignment_gate = np.clip(alignment_with_gate, -10, 10)  # Scale based on alignment, not velocity magnitude

    # reward = np.clip((self.previous_distance - current_distance) * 10, -100, 100)
    reward = velocity_alignment_gate + 0.1*safety_reward

    if self.navigator.current_gate_index == len(self.gate_poses_ground_truth):  # Track completed
        done = True
        reward = 100.0  # Completion bonus
        print('Track completed!')
        return reward, done
    
    if self.gate_index > self.prev_gate_index:  # A gate has been passed
        reward = 50.0  # Reward for passing a gate plus movement reward
        self.prev_gate_index = np.copy(self.gate_index)

    # Collision or out of bounds
    if state.collision.has_collided:
        reward += -10.0  # Severe penalty for collision or out of bounds
        # print('Collision')

    # Out of track
    if current_distance >= self.calculate_threshold(threshold_coeff):
        done = True
        reward = -50.0  # Severe penalty for being out of track
        # print (self.distance_history)
        # print('Out of track')

    # Stuck detection
    if self._check_positions_change() and len(self.positions) == self.max_positions_stored:
        done = True
        reward = -50.0  # Severe penalty for being stuck
        # print("Stuck at a location")


    if verbose:
        print(f"Step: {self.timesteps}/{self.totalsteps} Action: {action} Reward: {reward:.4f} Velocity Reward: {velocity_alignment_gate:.4f} Safety: {0.1*safety_reward:.4f} Target Gate: {self.gate_index} Distance: {current_distance:.4f}")
        
    self.previous_distance = np.copy(current_distance)
    self.previous_position = np.copy(drone_position)

    return reward, done


def compute_pseudo_reward(self):
    dist = 10000000
    thresh_dist = 7
    beta = 1

    state = self.airsim_client.getMultirotorState()
    velocity = state.kinematics_estimated.linear_velocity
    position = state.kinematics_estimated.position
    quad_pos = np.array([position.x_val, position.y_val, position.z_val])
    done = False

    for i in range(0, len(self.gate_pos_list) - 1):
        
        dist = min(
            dist,
            np.linalg.norm(np.cross((quad_pos - self.gate_pos_list[i]), (quad_pos - self.gate_pos_list[i + 1])))
            / np.linalg.norm(self.gate_pos_list[i] - self.gate_pos_list[i + 1]),
        )


    
    if state.collision.has_collided:
        print ('Collision')
        reward = -100.0
    else:
        if self.gate_index > self.prev_gate_index:
            reward = 100.0
            self.prev_gate_index = np.copy(self.gate_index)
        elif dist > thresh_dist:
            reward = -10
        else:
            reward_dist = math.exp(-beta * dist) - 0.5
            reward_speed = (
                np.linalg.norm(
                    [
                        velocity.x_val,
                        velocity.y_val,
                        velocity.z_val,
                    ]
                )
                - 0.5
            )
            reward = reward_dist + reward_speed

    if self.gate_index == len(self.gate_poses_ground_truth): # track completed
        done = True

    return reward, done