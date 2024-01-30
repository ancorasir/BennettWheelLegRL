# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BennettParallelRoughCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
            pos = [0.0, 0.0, 0.24] # x,y,z [m]
            default_joint_angles = { # = target angles [rad] when action = 0.0
                'FL_SingleMotor_doubleMotor_joint': 0.2,     # [rad]
                'RL_SingleMotor_doubleMotor_joint': 0.2,   # [rad]
                'FR_SingleMotor_doubleMotor_joint': 0.2,     # [rad]
                'RR_SingleMotor_doubleMotor_joint': 0.2,

                'FL_DoubleMotor_link12_joint': 0.,   # [rad]
                'RL_DoubleMotor_link12_joint': 0.,   # [rad]
                'FR_DoubleMotor_link12_joint': 0.,     # [rad]
                'RR_DoubleMotor_link12_joint': 0.,

                'FL_Leg_link12_link23_joint': 0.,   # [rad]
                'RL_Leg_link12_link23_joint': 0.,    # [rad]               
                'FR_Leg_link12_link23_joint': 0.,     # [rad]               
                'RR_Leg_link12_link23_joint': 0.
        }
            

    class control( LeggedRobotCfg.control ):
                # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bennett_parallel/bennett_parallel.urdf'
        name = "bennett_parallel"
        foot_name = "link23"
        penalize_contacts_on = ["link12", "link23"]
        terminate_after_contacts_on = ["Body"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.01

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 1
        soft_dof_vel_limit = 1
        base_height_target = 0.25
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        
        class scales( LeggedRobotCfg.rewards.scales ):
            dof_pos_limits = -1.0   # Penalize dof positions too close to the limit
            termination = -1.05     # Terminal reward / penalty
            tracking_lin_vel = 1.0  # Tracking of linear velocity commands (xy axes)
            tracking_ang_vel = 0.5  # Tracking of angular velocity commands (yaw) 
            lin_vel_z = -1.0        # Penalize z axis base linear velocity
            ang_vel_xy = -0.05      # Penalize xy axes base angular velocity
            orientation = -0.       # Penalize non flat base orientation
            torques = -0.00001      # Penalize torques
            dof_vel = -0.           # Penalize dof velocities
            dof_acc = -2.5e-7       # Penalize dof accelerations
            base_height = -0.       # Penalize base height away from target
            feet_air_time =  0.5    # Reward long steps
            collision = -0.1        # Penalize collisions on selected bodies
            feet_stumble = -0.0     # Penalize feet hitting vertical surfaces
            action_rate = -0.01     # Penalize changes in actions
            stand_still = -0.1      # Penalize motion at zero commands

    
    class terrain( LeggedRobotCfg.terrain ):
        vertical_scale = 0.0025 # [m]
        mesh_type = 'plane'
        measure_heights = True
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            # for training
            lin_vel_x = [-0.6, 0.6] # min max [m/s]
            lin_vel_y = [-0.6, 0.6] # min max [m/s]
            ang_vel_yaw = [-1, 1]   # min max [rad/s]
            heading = [-3.14, 3.14]



class BennettParallelRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_parallel_bennett'
        max_iterations = 5000 # number of policy updates

 
 