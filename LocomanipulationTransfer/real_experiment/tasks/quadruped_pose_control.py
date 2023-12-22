import os
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data", "quadruped_pose_control")

from base.fixed_frequency_task import FixedFrequencyTask
from base.task_log import QuadrupedPoseControlLog
from controllers.policy import Policy
from controllers.quadruped_controller import QuadrupedPositionController
from controllers.quadruped_pose_reader import QuadrupedVispPoseReader

from tasks.trajectory_tracker import QuadrupedTrajectoryLoader

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

DEBUG = True

def quat_conjugate(isaac_quat:np.array):
    conjugate = np.array([isaac_quat[0], -isaac_quat[1], -isaac_quat[2], -isaac_quat[3]])
    return conjugate

def quat_mul(isaac_quat_a:np.array, isaac_quat_b:np.array):
    w1, x1, y1, z1 = isaac_quat_a[0], isaac_quat_a[1], isaac_quat_a[2], isaac_quat_a[3]
    w2, x2, y2, z2 = isaac_quat_b[0], isaac_quat_b[1], isaac_quat_b[2], isaac_quat_b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.array([w, x, y, z])

class QuadrupedPoseControlTask(FixedFrequencyTask):
    # Control
    action_scale = 0.1

    base_position_scale = 1.0
    base_quaternion_scale = 1

    joint_position_scale = 0.3
    joint_velocity_scale = 0.05

    min_joint_pos_swing_ext = [-2.35, -0.78, -0.78, -2.35,
                                -2.09, 0.52, # Swing, extension
                                1.05, 0.52,
                                1.05, 0.52,
                                -2.09, 0.52]
    max_joint_pos_swing_ext = [0.78,  2.35,  2.35,  0.78,
                                -1.05,   2.09, # Swing, extension
                                2.09, 2.09,
                                2.09, 2.09,
                                -1.05, 2.09]

    def __init__(self, 
                 quadruped_controller: QuadrupedPositionController,
                 pose_reader: QuadrupedVispPoseReader,
                 goal_quaternion: np.array,
                 policy: Policy,
                 num_obs=86,
                 num_actions=12,
                 log_file_name="test",
                 freq_hz=40, 
                 episode_len=120) -> None:
        self.quadruped_controller = quadruped_controller
        self.pose_reader = pose_reader
        self.goal_quaternion = goal_quaternion
        self.policy = policy
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.log_file_name = log_file_name

        # Data logging
        self.data_logger = QuadrupedPoseControlLog(self.log_file_name)
        

        self.base_position = None
        self.base_quaternion = None
        self.joint_positions = None
        self.joint_velocities = None

        self.current_joint_position_targets = np.array([-1.57, 1.57, 1.57, -1.57,
                                                        -1.04, -2.09, 
                                                        2.09, 1.04, 
                                                        2.09, 1.04, 
                                                        -1.04, -2.09])
        self.last_joint_position_targets = np.array([-1.57, 1.57, 1.57, -1.57,
                                                    -1.04, -2.09, 
                                                    2.09, 1.04, 
                                                    2.09, 1.04, 
                                                    -1.04, -2.09])
        self.current_joint_position_targets_se = np.array([-1.57, 1.57, 1.57, -1.57,
                                                            -1.57, 1.05,
                                                            1.57, 1.05,
                                                            1.57, 1.05,
                                                            -1.57, 1.05])
        
        self.joint_position_target_se_upper = np.array(self.max_joint_pos_swing_ext)
        self.joint_position_target_se_lower = np.array(self.min_joint_pos_swing_ext)

        self.last_actions = np.zeros(self.num_actions)
        self.current_actions = np.zeros(self.num_actions)
        
        super().__init__(freq_hz, episode_len)

        # Disable torque for testing
        # self.quadruped_controller.dc.torque_off()

        # run policy once
        self.policy.get_actions(np.zeros(self.num_obs))

        ## For debug purpose
        if DEBUG:
            traj_file = "/home/bionicdl/SHR/LocomanipulationTransfer/RobotLearning/real_experiment/data/trajectory_tracking/0609-exp2"
            # self.trajectory = QuadrupedTrajectoryLoader(traj_file)
            with open(traj_file, "rb") as f:
                self.traj = pickle.load(f)
                self.current_traj_i = 0
                self.num_traj = len(self.traj)

    def task(self, current_timestamp_ms, current_progress_step):
        # For debug purpose
        # self.current_actions[:] = 0.0

        # Convert actions to joint position targets
        delta_joint_position_targets = self.current_actions * self.action_scale
        self.current_joint_position_targets_se = self.current_joint_position_targets_se + delta_joint_position_targets
        self.current_joint_position_targets_se = np.clip(self.current_joint_position_targets_se, 
                                                          a_min=self.joint_position_target_se_lower, 
                                                          a_max=self.joint_position_target_se_upper)
        # Convert swing extension to dof2 and dof3 angles
        joint_swing = self.current_joint_position_targets_se[[4,6,8,10]]
        joint_extension = self.current_joint_position_targets_se[[5,7,9,11]]
        dof2_joint_positions = joint_swing + joint_extension/2.0
        dof3_joint_positions = joint_swing - joint_extension/2.0
        self.current_joint_position_targets[0:4] = self.current_joint_position_targets_se[0:4] # For dof1, they are the same
        self.current_joint_position_targets[[4,6,8,10]] = dof2_joint_positions # For dof2 joint positions
        self.current_joint_position_targets[[5,7,9,11]] = dof3_joint_positions # For dof3 joint positions

        # Apply actions
        ## For debug purposes, tracking a fixed trajectory
        if DEBUG:
            # joint_trajectory = self.trajectory.get_current_traj()
            # if len(joint_trajectory) > 0:
            #     self.quadruped_controller.set_joint_position_targets(joint_trajectory)
            if self.current_traj_i < self.num_traj:
                joint_trajectory = self.traj[self.current_traj_i].cpu().numpy()
                self.quadruped_controller.set_joint_position_targets(joint_trajectory)
                self.current_traj_i += 1
            else:
                pass
        else:
            self.quadruped_controller.set_joint_position_targets(self.current_joint_position_targets)
        
        # Get observations
        self.update_obs()
        current_rotations = R.from_quat(self.base_quaternion)
        rot_mat = current_rotations.as_matrix()
        up_vector = rot_mat[0:3, 2]
        isaac_quaternion = [self.base_quaternion[3], self.base_quaternion[0], self.base_quaternion[1], self.base_quaternion[2]] # Convert to isaac sim convention
        quat_diff = quat_mul(isaac_quaternion, quat_conjugate(self.goal_quaternion))
        self.quat_diff = quat_diff
        # Scale raw observation
        scaled_base_positions = self.base_position_scale * self.base_position
        scaled_joint_positions = self.joint_position_scale * self.joint_positions
        scaled_joint_velocities = self.joint_velocity_scale * self.joint_velocities
        # scaled_joint_position_targets = self.current_joint_position_targets * 0.3
        # scaled_last_joint_position_targets = self.last_joint_position_targets * 0.3

        obs_vec = np.concatenate((
            scaled_base_positions,
            up_vector,
            quat_diff,

            scaled_joint_positions,
            scaled_joint_velocities,

            self.current_actions,
            self.last_actions,

            # scaled_joint_position_targets,
            # scaled_last_joint_position_targets
        ), axis=0)

        # Update buffers
        self.last_actions = self.current_actions.copy()
        self.last_joint_position_targets = self.current_joint_position_targets.copy()

        # Get actions
        self.current_actions = self.policy.get_actions(obs_vec)

        self.data_logger.add_data(timestamp=current_timestamp_ms,
                                  progress_step=current_progress_step,
                                  base_position=self.base_position,
                                  base_quaternion=isaac_quaternion,
                                  joint_positions=self.joint_positions,
                                  joint_velocities=self.joint_velocities,
                                  goal_quaternion=self.goal_quaternion,
                                  quaternion_diff=quat_diff,
                                  joint_position_target=self.current_joint_position_targets,
                                  action=self.current_actions,
                                  observation=obs_vec)

    def update_obs(self):
        self.base_position, self.base_quaternion = self.pose_reader.get_pose_in_origin()
        self.quadruped_controller.update_joint_states()
        self.joint_positions = self.quadruped_controller.joint_positions
        self.joint_velocities = self.quadruped_controller.joint_velocities

    def close_task(self):
        print(self.quat_diff)
        self.data_logger.save_data()
