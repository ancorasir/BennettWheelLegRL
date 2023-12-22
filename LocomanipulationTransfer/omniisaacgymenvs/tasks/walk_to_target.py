import torch

from robot.bennett_robot import BennettRobot
from robot.omni_robots import QuadrupedRobot
from objects.pose_indicator import PoseIndicator
from tasks.utils.success_tracker import SuccessTracker
from tasks.base.rl_task import RLTask

from utils.math import rand_quaternions, lgsk_kernel
from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_conjugate, quat_mul, quat_axis, get_euler_xyz
from utils.terrain_utils.random_uniform_terrain import RandomUniformTerrain
from omni.isaac.core.utils.stage import get_current_stage

class WalkToTargetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        # The marker object above the robot to display the goal rotation
        self.pose_indicator_loco = PoseIndicator(object_name="pose_indicator_loco",
                                                 object_prim_name="/pose_indicator_xyz", # object_prim_name="/pose_indicator_xyz",
                                                 scale=0.25)

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"] # Distance between each environment
        self._dt = self._task_cfg['sim']['dt'] # simulation time interval

        # Robot configurations
        # Robot configs 
        self.default_robot_states = self._task_cfg["env"]["default_robot_states"]
        self.default_joint_positions = self.default_robot_states["default_joint_positions"]
        self.default_base_positions = self.default_robot_states["default_base_positions"]
        self.default_base_quaternions = self.default_robot_states["default_base_quaternions"]
        self.default_actuator_params = self._task_cfg["env"]["default_actuator_params"]
        self.control_kp = self.default_actuator_params["control_kp"]
        self.control_kd = self.default_actuator_params["control_kd"]
        self.action_scale = self.default_actuator_params["action_scale"]
        self.joint_friction = self.default_actuator_params["joint_friction"]
        self.joint_damping = self.default_actuator_params["joint_damping"]
        self.max_torque = self.default_actuator_params["max_torque"]
        self.control_decimal = self.default_actuator_params["control_decimal"]
        self.swing_extension_action = self.default_actuator_params.get("swing_extension", True)
        self.actuatl_control_decimal = self.control_decimal - 1 # isaac sim intrinsicly contains 1 simulation step that we cannot control
        self.action_latency = self.default_actuator_params.get("action_latency", False) 
        if self.action_latency:
            self.action_probability = self.default_actuator_params.get("action_probability", [0.2, 0.8])
            self.action_buffer_size = len(self.action_probability)
        self.robot = BennettRobot(fixed=False, # Floating base for quadruped robot
                                    module_config="horizontal",

                                    actual_control_decimal=self.actuatl_control_decimal,

                                    default_joint_positions=self.default_joint_positions,
                                    default_base_position=self.default_base_positions,
                                    default_base_quaternion=self.default_base_quaternions,
                                    
                                    control_kp=self.control_kp,
                                    control_kd=self.control_kd,
                                    action_scale=self.action_scale,
                                    joint_friction=self.joint_friction,
                                    joint_damping=self.joint_damping,
                                    max_torque=self.max_torque,
                                    
                                    enable_tip_contact_sensor=True, # self.obs_include_tip_contact_forces,
                                    enable_contact_termination=True)

        # self._max_episode_length = self._task_cfg['sim']['max_episode_length']
        self.indicator_position = self._task_cfg["env"]["task"]["loco_indicator_position"]

        self._task_related_cfg = self._task_cfg["env"]["task"]
        self.max_goal_distance = self._task_related_cfg["max_goal_distance"]
        self.max_episode_length = self._task_related_cfg["max_episode_length"]
        self.min_episdoe_length = self._task_related_cfg["min_episode_length"]

        self.obs_scale_config = self._task_related_cfg["obs_scales"]
        self.obs_position_scale = self.obs_scale_config["position_scale"]
        self.obs_quaternion_scale = self.obs_scale_config["quaternion_scale"]
        self.obs_up_vec_scale = self.obs_scale_config["up_vec_scale"]
        self.obs_linear_vel_scale = self.obs_scale_config["linear_vel_scale"]
        self.obs_angular_vel_scale = self.obs_scale_config["angular_vel_scale"]
        self.obs_joint_position_scale = self.obs_scale_config["joint_position_scale"]
        self.obs_joint_velocity_scale = self.obs_scale_config["joint_velocity_scale"]
        self.obs_contact_force_scale = self.obs_scale_config["contact_force_scale"]
        self.obs_joint_torque_scale = self.obs_scale_config["joint_torque_scale"]
        
        self.obs_cfg = self._task_cfg['env']['task']["obs"]
        self.use_symmetric_joint_obs = self.obs_cfg.get("use_symmetric_joint_obs", False)
        self.use_swing_extension_obs = self.obs_cfg.get("use_swing_extension_obs", False)
        self.obs_include_joint_torques = self.obs_cfg["include_joint_torques"]
        self.obs_include_velocities = self.obs_cfg["include_velocities"]
        self.obs_include_tip_contact_forces = self.obs_cfg["include_tip_contact_forces"]
        self.obs_full_contact_forces = self.obs_cfg["full_contact_forces"]
        self.obs_binary_force_threshold = self.obs_cfg["binary_force_threshold"]

        # reward scales
        self.rew_scale_config = self._task_related_cfg["rew_scale"]
        self.position_error_penalty_scale = self.rew_scale_config["position_error_penalty_scale"]
        self.position_eps = self.rew_scale_config["position_eps"]
        self.heading_vel_rew_scale = self.rew_scale_config["heading_vel_rew"]
        self.deviation_distance_penalty_scale = self.rew_scale_config["deviation_distance_penalty_scale"]
        self.deviation_vel_penalty_scale = self.rew_scale_config["deviation_vel_penalty_scale"]
        self.angular_vel_penalty_scale = self.rew_scale_config["angular_vel_penalty_scale"]
        self.rot_penalty_scale = self.rew_scale_config["rot_penalty_scale"]
        self.rew_joint_acc_scale = self.rew_scale_config["joint_acc_scale"]
        self.rew_action_rate_scale = self.rew_scale_config["action_rate_scale"]
        self.cosmetic_penalty_scale = self.rew_scale_config["cosmetic_penalty"]
        self.rew_success = self.rew_scale_config["success_reward"]
        self.rew_fall = self.rew_scale_config["fall_penalty"]

        self.clip_reward = self._task_related_cfg["clip_reward"]

        self.dist_thresh = self._task_related_cfg["success_conditions"]["dist_thresh"]
        self.consecutive_success_steps = self._task_related_cfg["success_conditions"]["consecutive_success_steps"]

        self.early_term_config = self._task_related_cfg["early_termination_conditions"]
        self.max_roll = self.early_term_config["max_roll"]
        self.max_pitch = self.early_term_config["max_pitch"]
        self.max_yaw = self.early_term_config["max_yaw"]
        self.max_deviation_dist = self.early_term_config["max_deviation_distance"]
        self.baseline_base_height = self.early_term_config["baseline_base_height"]
        self.baseline_knee_height = self.early_term_config["baseline_knee_height"]
        self.baseline_corner_height = self.early_term_config["baseline_corner_height"]

        # Randomize params
        self.rand_config = self._task_cfg["env"]["randomization"]
        ## Joint position randomization
        self.rand_init_joint_positions = self.rand_config["randomize_init_joint_positions"]
        if self.rand_init_joint_positions:
            self.rand_joint_position_deviation = self.rand_config["rand_joint_position"]
        else:
            self.rand_joint_position_deviation = 0.0
        ## Joint velocities randomization
        self.rand_init_joint_velocities = self.rand_config["randomize_init_joint_velocities"]
        if self.rand_init_joint_velocities:
            self.rand_joint_velocity_deviation = self.rand_config["rand_joint_velocities"]
        else:
            self.rand_joint_velocity_deviation = 0.0
        ## Base position randomization
        self.rand_init_base_positions = self.rand_config["randomize_init_base_positions"]
        if self.rand_init_base_positions:
            self.rand_base_xy_deviation = self.rand_config["rand_xy_position"]
            self.rand_base_z_deviation = self.rand_config["rand_z_position"]
        else:
            self.rand_base_xy_deviation = 0.0
            self.rand_base_z_deviation = 0.0
        ## Base quaternion randomization
        self.rand_init_base_quaternions = self.rand_config["randomize_init_base_quaternions"]
        if self.rand_init_base_quaternions:
            self.rand_max_roll = self.rand_config["rand_max_roll"]
            self.rand_max_pitch = self.rand_config["rand_max_pitch"]
            self.rand_max_yaw = self.rand_config["rand_max_yaw"]
        else:
            self.rand_max_roll = 0.0
            self.rand_max_pitch = 0.0
            self.rand_max_yaw = 0.0
        ## Base velocity randomization
        self.rand_init_base_velocities = self.rand_config["randomize_init_base_velocities"]
        if self.rand_init_base_velocities:
            self.rand_max_linear_vel = self.rand_config["rand_max_linear_vel"]
            self.rand_max_angular_vel = self.rand_config["rand_max_angular_vel"]
        else:
            self.rand_max_linear_vel = 0.0
            self.rand_max_angular_vel = 0.0

        self._num_actions = 12 # 12 joints
        self.num_basic_observations = 79
        self._num_observations = self.num_basic_observations
        # Add extra observations if needed
        if self.obs_include_joint_torques:
            self._num_observations += 12
        if self.obs_include_velocities:
            self._num_observations += 6
        if self.obs_include_tip_contact_forces:
            if self.obs_full_contact_forces:
                self._num_observations += 12
            else:
                self._num_observations += 4
        self._num_states = self._num_observations

        RLTask.__init__(self, name, env)

        if self.action_latency:
            # Action latency buffer
            from collections import deque
            # Action query
            self.action_buffer = torch.zeros(self.action_buffer_size, self._num_envs, 12, dtype=torch.float32, device=self._device) # [oldest, ..., newest]
            self.action_probability = torch.tensor(self.action_probability, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        
        # Action buffer for observation
        self.last_actions = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float32, device=self._device)
        self.current_actions = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float32, device=self._device)

        self.success_tracker = SuccessTracker(max_reset_counts=2048,
                                              rot_thresh=self.dist_thresh,
                                              consecutive_success_steps=self.consecutive_success_steps)
        
        self.success_tracker.init_success_tracker(self._num_envs, self._device)

        # Default pose indicator position tensor
        self.default_pose_indicator_positions = torch.tensor(self.indicator_position, dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))

        # Goal quaternion
        self.goal_position_2d = torch.zeros(self._num_envs, 2, dtype=torch.float32, device=self._device)
        self.goal_direction_xy = torch.zeros(self._num_envs, 2, dtype=torch.float32, device=self._device)
        self.goal_quaternions = torch.tensor(self.robot.default_base_quaternion, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)

        self.last_joint_velocities = torch.zeros(self._num_envs, 12, dtype=torch.float32, device=self._device)

        self.episode_length = torch.tensor([self.max_episode_length]*self._num_envs, dtype=torch.float32, device=self._device)

        self._time_inv = self._dt * self.control_frequency_inv
        self.actual_time_inv = self._dt * self._task_cfg["env"]["default_actuator_params"]["control_decimal"]

        # Set up terrains
        self.terrain_cfg = self._task_cfg["env"]["terrain"]
        self.enable_terrain = self.terrain_cfg["enable_terrain"]
        if self.enable_terrain:
            self._task_cfg["sim"]["add_ground_plane"] = False
            self.terrain_generator = RandomUniformTerrain(self._task_cfg, self._num_envs)
        # else:
        #     self._task_cfg["sim"]["add_ground_plane"] = True
        

    def set_up_scene(self, scene) -> None:
        # Add one robot to the scene
        self.robot.init_omniverse_robot(self.default_zero_env_path, self._sim_config)

        # Add one marker to the scene
        self.pose_indicator_loco.init_stage_object(self.default_zero_env_path, self._sim_config)

        # Set up random terrain if required
        if self.enable_terrain:
            self.terrain_generator.get_terrain()
        
        super().set_up_scene(scene, False) # This will replicate the robot and objects
    
        # Init robot views
        self.robot.init_robot_views(scene)

        # Init pose indicator views
        self.pose_indicator_loco.init_object_view(scene)

    def post_reset(self):
        # Post reset robots and objects
        self.robot.post_reset_robot(self._env_pos, self._num_envs, self._device)
        self.pose_indicator_loco.post_reset_object(self._env_pos, self._num_envs, self._device)

    def pre_physics_step(self, actions):
        actions = actions.to(self._device)
        current_actions = actions

        # current_actions[0, :] = 0.0 # *torch.ones(self._num_envs, 12, dtype=torch.float32, device=self._device)

        self.current_actions = current_actions.clone()

        # transform joint actions if needed
        # Actions: (a1-dof1, a2-dof1, a3-dof1, a4-dof1,
        # a1-swing, a1-extension, a2-swing, a2-extension, a3-swing, a3-extension, a4-swing, a4-extension)
        if self.use_symmetric_joint_obs:
            if self.swing_extension_action:
                current_actions[:, 0] = -current_actions[:, 0] # Invert sign of a1-dof1
                current_actions[:, 3] = -current_actions[:, 3] # Invert sign of a4-dof1
                # invert sign of swing
                current_actions[:, 4] = -current_actions[:, 4] # Invert sign of a1-swing
                current_actions[:, 10] = -current_actions[:, 10] # Invert sign of a4-swing
            else:
                # For dof1 joint actions
                current_actions[:, 0] = -current_actions[:, 0] # Invert sign of a1-dof1
                current_actions[:, 3] = -current_actions[:, 3] # Invert sign of a4-dof1

                # for dof2 and dof3 joint actions
                current_actions[:, 4] = -current_actions[:, 4] # Invert sign of a1-dof2
                current_actions[:, 5] = -current_actions[:, 5] # Invert sign of a1-dof3
                current_actions[:, 10] = -current_actions[:, 10] # Invert sign of a4-dof2
                current_actions[:, 11] = -current_actions[:, 11] # Invert sign of a4-dof3
                # Swap dof2 and dof3 for a1
                temp_dof2_actions = current_actions[:, 4].clone()
                current_actions[:, 4] = current_actions[:, 5]
                current_actions[:, 5] = temp_dof2_actions
                # Swap dof2 and dof3 for a4
                temp_dof2_actions = current_actions[:, 10].clone()
                current_actions[:, 10] = current_actions[:, 11]
                current_actions[:, 11] = temp_dof2_actions

        # Append action to action buffer
        if self.action_latency:
            self.action_buffer[-1, :, :] = current_actions # self.current_actions

            # Randomly select action from action probability
            action_indices = torch.flatten(torch.multinomial(self.action_probability, 1))
            gather_indices = action_indices.view(self._num_envs, 1).repeat(1, 12).repeat(self.action_buffer_size, 1, 1)
            gathered_actions = torch.gather(self.action_buffer, dim=0, index=gather_indices)
            current_actions = gathered_actions[0, :, :] # self.current_actions = gathered_actions[0, :, :]

            # Append actions to buffer
            action_buffer_clone = self.action_buffer.clone()
            old_actions = action_buffer_clone[1:self.action_buffer_size, :, :]
            self.action_buffer[0:self.action_buffer_size-1, :, :] = old_actions
            self.action_buffer[self.action_buffer_size-1, :, :] = current_actions # self.current_actions

        # Check if the environment needes to be reset for this specific task
        reset_buf = self.reset_buf.clone()
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # self.robot.take_action(self.current_actions, self.rl_task._env._world)
        self.robot.take_action(current_actions.clone(), 
                               self._env._world, 
                               control_mode="increment",
                               swing_extension_control=self.swing_extension_action,
                               render_subframes=False)

    def reset_idx(self, env_ids):
        # Check if the env ids belongs to this evironment
        num_resets = len(env_ids)

        # Reset joint positions
        self.robot.reset_robot_joint_positions(env_ids, self.rand_joint_position_deviation)

        # Reset joint velocities
        self.robot.reset_robot_joint_velocities(env_ids, self.rand_joint_velocity_deviation)

        # Reset robot base poses
        self.robot.reset_robot_poses(env_ids,
                                    self.rand_base_xy_deviation,
                                    self.rand_base_z_deviation,
                                    rand_max_roll=self.rand_max_roll,
                                    rand_max_pitch=self.rand_max_pitch,
                                    rand_max_yaw=self.rand_max_yaw)
        
        # Reset robot base velocities
        self.robot.reset_robot_velocities(env_ids,
                                        rand_max_linear_velocity=self.rand_max_linear_vel,
                                        rand_max_angular_velocity=self.rand_max_angular_vel)
        
        rand_distance = torch.rand(num_resets, device=self._device) * self.max_goal_distance
        # 2 * (torch.rand(num_loco_resets, device=self._device) - 0.5) * self.max_goal_distance
        # torch.ones(num_loco_resets, device=self._device) * self.max_goal_distance 
        # rand_distance = torch.clamp(rand_distance, min=0.3)
        # torch.rand(num_loco_resets, device=self._device) * self.max_goal_distance # from 0 ~ max_goal_distance
        # Calculate episode length depending on the goal positions
        episode_length = self.max_episode_length * torch.abs(rand_distance)/self.max_goal_distance
        self.episode_length[env_ids] = torch.clamp(episode_length, min=self.min_episdoe_length)

        # Sample goal directions
        ## randomly sample a theta (relative to positive x)
        direction_theta = 2.0 * (torch.rand(num_resets, device=self._device) - 0.5) * torch.pi
        # torch.pi/2.0 * torch.ones(num_loco_resets, dtype=torch.float32, device=self._device)
        # torch.pi/2.0 * torch.ones(num_loco_resets, dtype=torch.float32, device=self._device)
        # (torch.rand(num_loco_resets, device=self._device) - 0.5) * torch.pi # from 0 ~ pi
        # 2.0 * (torch.rand(num_loco_resets, device=self._device) - 0.5) * torch.pi # from -pi ~ pi
        direction_x = torch.cos(direction_theta)
        direction_y = torch.sin(direction_theta)
        new_unit_direction_vector = torch.cat((
            direction_x.view(num_resets, 1),
            direction_y.view(num_resets, 1)
        ), dim=-1)
        self.goal_direction_xy[env_ids] = new_unit_direction_vector

        # Calculate xy from directions
        rand_x_position = rand_distance * direction_x
        rand_y_position = rand_distance * direction_y
        self.goal_position_2d[env_ids] = torch.cat((
            rand_x_position.view(num_resets, 1),
            rand_y_position.view(num_resets, 1)
        ), dim=-1)

        # Reset pose indicator
        new_indicator_positions = torch.cat((
            self.goal_position_2d[env_ids],
            self.default_pose_indicator_positions[env_ids, 2].view(num_resets, 1)
        ), dim=-1)
        self.pose_indicator_loco.set_object_poses(positions=new_indicator_positions,
                                                  indices=env_ids)

        # Reset history buffers for env_ids
        self.last_actions[env_ids] = torch.zeros(num_resets, self._num_actions, dtype=torch.float32, device=self._device)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        self.last_joint_velocities[env_ids, :] = self.robot.current_joint_velocities[env_ids, :]
        self.success_tracker.reset_bookkeeping(env_ids)

    def get_observations(self):
        # Get raw observations
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        joint_torques = self.robot.get_joint_torques()
        base_positions, base_quaternions = self.robot.get_base_poses()
        base_linear_vels = self.robot.get_base_linear_velocities()
        base_angular_vels = self.robot.get_base_angular_velocities()
        base_height = base_positions[:, 2]
        position_error_2d =  self.goal_position_2d - base_positions[:, 0:2]

        # Transform joint observations if needed
        if self.use_symmetric_joint_obs:
            if not self.use_swing_extension_obs:
                # For dof1 joint positions
                joint_positions[:, 0] = -joint_positions[:, 0] # Invert sign of a1-dof1
                joint_positions[:, 3] = -joint_positions[:, 3] # Invert sign of a4-dof1

                # for dof2 and dof3 joint positions
                joint_positions[:, 4] = -joint_positions[:, 4] # Invert sign of a1-dof2
                joint_positions[:, 5] = -joint_positions[:, 5] # Invert sign of a1-dof3
                joint_positions[:, 10] = -joint_positions[:, 10] # Invert sign of a4-dof2
                joint_positions[:, 11] = -joint_positions[:, 11] # Invert sign of a4-dof3
                # Swap dof2 and dof3 for a1
                temp_dof2_positions = joint_positions[:, 4].clone()
                joint_positions[:, 4] = joint_positions[:, 5]
                joint_positions[:, 5] = temp_dof2_positions
                # Swap dof2 and dof3 for a4
                temp_dof2_positions = joint_positions[:, 10].clone()
                joint_positions[:, 10] = joint_positions[:, 11]
                joint_positions[:, 11] = temp_dof2_positions

                # For joint_velocities
                joint_velocities[:, 0] = -joint_velocities[:, 0] # Invert sign of a1-dof1
                joint_velocities[:, 3] = -joint_velocities[:, 3] # Invert sign of a4-dof1
                # for dof2 and dof3 joint velocities
                joint_velocities[:, 4] = -joint_velocities[:, 4] # Invert sign of a1-dof2
                joint_velocities[:, 5] = -joint_velocities[:, 5] # Invert sign of a1-dof3
                joint_velocities[:, 10] = -joint_velocities[:, 10] # Invert sign of a4-dof2
                joint_velocities[:, 11] = -joint_velocities[:, 11] # Invert sign of a4-dof3
                # Swap dof2 and dof3 for a1
                temp_dof2_velocities = joint_velocities[:, 4].clone()
                joint_velocities[:, 4] = joint_velocities[:, 5]
                joint_velocities[:, 5] = temp_dof2_velocities
                # Swap dof2 and dof3 for a4
                temp_dof2_velocities = joint_velocities[:, 10].clone()
                joint_velocities[:, 10] = joint_velocities[:, 11]
                joint_velocities[:, 11] = temp_dof2_velocities
            else:
                # Convert dof2 and dof3 joint positions to swing extension angles
                # For dof1 joint positions
                joint_positions[:, 0] = -joint_positions[:, 0] # Invert sign of a1-dof1
                joint_positions[:, 3] = -joint_positions[:, 3] # Invert sign of a4-dof1
                # calculate swings
                swings = (joint_positions[:, [4,6,8,10]] + joint_positions[:, [5,7,9,11]])/2.0
                extensions = joint_positions[:, [4,6,8,10]] - joint_positions[:, [5,7,9,11]]
                joint_positions[:, [4,6,8,10]] = swings # assign swings
                joint_positions[:, [5,7,9,11]] = extensions # assign extensions
                # Convert the sign of swing for a1 and a4
                joint_positions[:, 4] = -joint_positions[:, 4]
                joint_positions[:, 10] = -joint_positions[:, 10]

                # Calculate the swing velocity and extension velocity
                joint_velocities[:, 0] = -joint_velocities[:, 0] # Invert sign of a1-dof1
                joint_velocities[:, 3] = -joint_velocities[:, 3] # Invert sign of a4-dof1
                # calculate swing velocities
                swing_vels = (joint_velocities[:, [4,6,8,10]] + joint_velocities[:, [5,7,9,11]])/2.0
                extensions_vels = joint_velocities[:, [4,6,8,10]] - joint_velocities[:, [5,7,9,11]]
                joint_velocities[:, [4,6,8,10]] = swing_vels # assign swings
                joint_velocities[:, [5,7,9,11]] = extensions_vels # assign extensions
                # Convert the sign of swing for a1 and a4
                joint_velocities[:, 4] = -joint_velocities[:, 4]
                joint_velocities[:, 10] = -joint_velocities[:, 10]

                # For current_joint_position_targets in swing extension form
                current_joint_position_targets = self.robot.current_joint_position_targets_se.clone()
                # For dof1 joint positions
                current_joint_position_targets[:, 0] = -current_joint_position_targets[:, 0] # Invert sign of a1-dof1
                current_joint_position_targets[:, 3] = -current_joint_position_targets[:, 3] # Invert sign of a4-dof1
                # Convert the sign of swing for a1 and a4
                current_joint_position_targets[:, 4] = -current_joint_position_targets[:, 4]
                current_joint_position_targets[:, 10] = -current_joint_position_targets[:, 10]

                # For last joint position targets in swing extension form
                last_joint_position_targets = self.robot.last_joint_position_targets_se.clone()
                # For dof1 joint positions
                last_joint_position_targets[:, 0] = -last_joint_position_targets[:, 0] # Invert sign of a1-dof1
                last_joint_position_targets[:, 3] = -last_joint_position_targets[:, 3] # Invert sign of a4-dof1
                # Convert the sign of swing for a1 and a4
                last_joint_position_targets[:, 4] = -last_joint_position_targets[:, 4]
                last_joint_position_targets[:, 10] = -last_joint_position_targets[:, 10]

        # Scale the observations
        scaled_base_height = self.obs_position_scale * base_height
        scaled_base_positions = self.obs_position_scale * base_positions
        scaled_base_quaternion = self.obs_quaternion_scale * base_quaternions
        scaled_linear_vel = self.obs_linear_vel_scale * base_linear_vels
        scaled_angular_vel = self.obs_angular_vel_scale * base_angular_vels
        scaled_position_error_2d = self.obs_position_scale * position_error_2d
        scaled_goal_positions_2d = self.obs_position_scale * self.goal_position_2d
        scaled_joint_torques = self.obs_joint_torque_scale * joint_torques
        scaled_joint_position_targets = self.obs_joint_position_scale * current_joint_position_targets
        scaled_last_joint_position_targets = self.obs_joint_position_scale * last_joint_position_targets

        scaled_joint_positions = self.obs_joint_position_scale * joint_positions
        scaled_joint_velocities = self.obs_joint_velocity_scale * joint_velocities

        obs_tensor = torch.cat((
            scaled_base_height.view(self._num_envs, 1),
            scaled_position_error_2d,
            scaled_base_quaternion,

            scaled_joint_positions,
            scaled_joint_velocities,

            scaled_joint_position_targets,
            scaled_last_joint_position_targets,

            self.current_actions,
            self.last_actions
        ), dim=-1)

        if self.obs_include_joint_torques:
            obs_tensor = torch.cat((
                obs_tensor,

                scaled_joint_torques
            ), dim=-1)

        if self.obs_include_velocities:
            obs_tensor = torch.cat((
                obs_tensor,

                scaled_linear_vel,
                scaled_angular_vel,
            ), dim=-1)

        if self.obs_include_tip_contact_forces:
            # Get the tip contact forces in ground frame
            tip_contact_forces = self.robot._tip_view.get_net_contact_forces(clone=False, dt=self._dt)
            if self.obs_full_contact_forces:
                # Transform the tip contact forces to robot frame
                default_robot_quaternions_in_ground = torch.repeat_interleave(self.robot.default_base_quaternions, 
                                                                              torch.tensor([4],device=self._device), 
                                                                              dim=0)
                tip_contact_forces_in_robot = quat_rotate_inverse(default_robot_quaternions_in_ground, tip_contact_forces)
                tip_contact_forces = tip_contact_forces_in_robot.view(self._num_envs, 4, 3)
                scaled_contact_forces = self.obs_contact_force_scale * tip_contact_forces.view(self._num_envs, 12)
                obs_tensor = torch.cat((
                    obs_tensor,
                    scaled_contact_forces
                ), dim=-1)
            else:
                tip_contact_force_mag = torch.norm(tip_contact_forces.view(self._num_envs, 4, 3), dim=-1)
                tip_contact = (tip_contact_force_mag > self.obs_binary_force_threshold).to(torch.float32)
                obs_tensor = torch.cat((
                    obs_tensor,
                    tip_contact
                ), dim=-1)

        self.obs_buf[:] = obs_tensor
        self.states_buf[:] = self.obs_buf[:]

        self.base_positions = base_positions
        self.position_error_2d = position_error_2d
        self.joint_velocities = joint_velocities
        self.base_linear_vel = base_linear_vels
        self.base_angular_vel = base_angular_vels
        self.base_quaternions = base_quaternions
        self.joint_positions = joint_positions

    def calculate_metrics(self):
        '''Heading distance reward'''
        # Heading distance = bg, where b is the actual base xy position, g is the goal direction
        heading_dist = torch.sum(self.base_positions[:, 0:2] * self.goal_direction_xy, dim=-1)
        goal_heading_dist = torch.sum(self.goal_position_2d * self.goal_direction_xy, dim=-1)
        heading_dist_error = torch.abs(goal_heading_dist - heading_dist)
        # Longer goal distance receive higher reward
        # heading_dist_rew = 1.0 / (heading_dist_error + self.position_eps) * self.position_error_penalty_scale + goal_heading_dist 
        # heading_dist_rew = torch.clamp(-heading_dist_error*2.0 + 1.0, min=0.0)
        heading_dist_rew = 4.0 * lgsk_kernel(heading_dist_error, scale=8, eps=2) # From 0~1

        '''Heading velocity reward'''
        # Heading velocity = vg, where v is the actual base xy velocity, g is the goal direction
        heading_vel = torch.sum(self.base_linear_vel[:, 0:2] * self.goal_direction_xy, dim=-1)
        heading_vel_rew = torch.clamp(self.heading_vel_rew_scale * heading_vel, min=0.0) # Positive only

        '''Deviation distance penalty'''
        # Deviation distance = |b x g|, where b is the actual base xy position, g is the goal direction
        base_positions = torch.cat((
            self.base_positions[:, 0:2],
            torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self._device) # z = 0
        ), dim=-1)
        goal_directions = torch.cat((
            self.goal_direction_xy,
            torch.zeros(self._num_envs, 1, dtype=torch.float32, device=self._device) # z = 0
        ), dim=-1)
        b_cross_g = torch.cross(base_positions, goal_directions, dim=-1)
        self.deviation_distance = torch.norm(b_cross_g, dim=-1)
        deviation_distance_penalty = self.deviation_distance_penalty_scale * self.deviation_distance
        # # Apply only when the robot is not at the origin
        # out_of_origin = torch.norm(self.base_positions[:, 0:2], dim=-1) > 0.1
        # deviation_distance_penalty = deviation_distance_penalty * out_of_origin.to(torch.float32)

        '''Deviation velocity penalty'''
        # Deviation velocity = | v x g | where v is the actual base velocity, g is the goal direction
        v_cross_g = torch.cross(self.base_linear_vel, goal_directions)
        deviation_vel = torch.norm(v_cross_g, dim=-1)
        deviation_vel_penalty = self.deviation_vel_penalty_scale * deviation_vel

        '''Angular velocity penalty'''
        angular_vel_penalty = self.angular_vel_penalty_scale * torch.norm(self.base_angular_vel, dim=-1)

        '''Rotation penalty'''
        quat_diff = quat_mul(self.base_quaternions, quat_conjugate(self.goal_quaternions))
        self.rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention
        rot_penalty = self.rot_penalty_scale * self.rot_dist

        '''joint acceleration penalty'''
        joint_acceleration = (self.joint_velocities - self.last_joint_velocities) / self.actual_time_inv
        joint_acc_penalty = torch.sum(torch.abs(joint_acceleration)*self.rew_joint_acc_scale, dim=1)
        self.last_joint_velocities = self.joint_velocities

        '''action rate penalty'''
        action_rate_penalty = torch.sum(torch.abs(self.current_actions), dim=1)*self.rew_action_rate_scale

        '''consecutive successes bonus reward'''
        position_error_distance = torch.norm(self.base_positions[:, 0:2] - self.goal_position_2d, dim=-1)
        self.success = self.success_tracker.track_success(position_error_distance)
        basic_successes_rew = self.rew_success*self.success.to(torch.float32)
        total_success_rew = basic_successes_rew # basic_successes_rew * (1 + goal_heading_dist) # Longer distance receive higher reward

        '''Cosmetic penalty'''
        joint_diff = torch.sum(torch.abs(self.joint_positions[:, 0:4] - self.robot.default_joint_positions[:, 0:4]), dim=-1)
        cosmetic_penalty = self.cosmetic_penalty_scale * joint_diff

        total_rew = heading_dist_rew + \
                    heading_vel_rew + \
                    deviation_distance_penalty + \
                    deviation_vel_penalty + \
                    angular_vel_penalty + \
                    rot_penalty + \
                    joint_acc_penalty + \
                    action_rate_penalty + \
                    cosmetic_penalty + \
                    total_success_rew
        
        if self.clip_reward:
            total_rew = torch.clamp(total_rew, min=0.0)
        
        log_dict = {}
        log_dict['heading_dist_rew_loco'] = heading_dist_rew
        log_dict['heading_vel_rew_loco'] = heading_vel_rew
        log_dict['deviation_distance_penalty_loco'] = deviation_distance_penalty
        log_dict['deviation_vel_penalty_loco'] = deviation_vel_penalty
        log_dict['angular_vel_penalty_loco'] = angular_vel_penalty
        log_dict['rot_penalty_loco'] = rot_penalty
        log_dict['joint_acc_penalty_loco'] = joint_acc_penalty
        log_dict['action_rate_penalty_loco'] = action_rate_penalty
        log_dict['cosmetic_penalty_loco'] = cosmetic_penalty
        log_dict['consecutive_successes_rew_loco'] = total_success_rew

        # Update rew_buf, goal_reset_buf
        self.rew_buf[:] = total_rew

        # Update last actions
        self.last_actions = self.current_actions.clone()

         # Update last joint position targets
        self.robot.last_joint_position_targets_se = self.robot.current_joint_position_targets_se.clone()

        # Too check if crash is observed in simulation
        _rew_is_nan = torch.isnan(self.rew_buf)
        _rew_nan_sum = torch.sum(_rew_is_nan)
        if _rew_nan_sum != torch.tensor(0.0):
            print("NaN Value found in reward")
            print(self.rew_buf)
        rew_below_thresh = self.rew_buf < -50.0
        is_any_rew_neg = rew_below_thresh.nonzero()
        assert len(is_any_rew_neg) == 0

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in log_dict.items()})

    def is_done(self):
        # Rotation reset
        # Reset envs if the rotation is too large from the default
        ## Get the quaternion between the default and the current rotation
        quat_diff_from_default = quat_mul(self.base_quaternions, quat_conjugate(self.goal_quaternions))
        roll, pitch, yaw = get_euler_xyz(quat_diff_from_default)
        ## Reset if roll too large
        self.reset_buf[:] = torch.where(torch.logical_and(roll > self.max_roll, roll < 2*torch.pi - self.max_roll),
                                                          torch.ones_like(self.reset_buf[:]),
                                                          self.reset_buf[:])
        ## Reset if pitch too large
        self.reset_buf[:] = torch.where(torch.logical_and(pitch > self.max_pitch, pitch < 2*torch.pi - self.max_pitch),
                                                          torch.ones_like(self.reset_buf[:]),
                                                          self.reset_buf[:])
        ## Reset if yaw too large
        self.reset_buf[:] = torch.where(torch.logical_and(yaw > self.max_yaw, yaw < 2*torch.pi - self.max_yaw),
                                                          torch.ones_like(self.reset_buf[:]),
                                                          self.reset_buf[:])
        
        # Deviation distance reset
        ## Reset envs if the deviation distance is too large
        self.reset_buf[:] = torch.where(self.deviation_distance > self.max_deviation_dist,
                                        torch.ones_like(self.reset_buf[:]),
                                        self.reset_buf[:])

        # Fall down reset
        ## Body heights for locomotion
        body_heights_loco = self.base_positions[:, 2]
        self.reset_buf[:] = torch.where(body_heights_loco <= self.baseline_base_height, 
                                        torch.ones_like(self.reset_buf[:]), 
                                        self.reset_buf[:])

        # Joint limit reset
        joint_break_limit = self.robot.check_links_self_collision()
        self.reset_buf[:] = torch.where(joint_break_limit, 
                                        torch.ones_like(self.reset_buf[:]), 
                                        self.reset_buf[:])
        
        # Reset env if contact force of the base/links is larger than 2N
        base_contact = self.robot.check_base_contact(2, dt=self._dt) 
        link_contact = self.robot.check_link_contact(2, dt=self._dt)
        self.reset_buf[:] = torch.where(torch.logical_or(base_contact, link_contact), 
                                        torch.ones_like(self.reset_buf[:]), 
                                        self.reset_buf[:])

        # Add falling penalty
        fall_penalty = (self.rew_fall * self.reset_buf).to(torch.float32)
        self.rew_buf += fall_penalty
        self.extras.update({"env/rewards/fall_penalty_loco": fall_penalty.mean()})

        # success reset
        self.reset_buf[:] = torch.where(self.success == 1, 
                                        torch.ones_like(self.reset_buf[:]), 
                                        self.reset_buf[:])

        # Check if maximum progress reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.episode_length - 1, 
                                        torch.ones_like(self.reset_buf[:]), 
                                        self.reset_buf[:])

        success_rate = self.success_tracker.update_success_rate(self.reset_buf[:])
        self.extras.update({"env/success_rate_loco": success_rate})