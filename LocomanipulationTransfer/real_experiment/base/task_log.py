import os
root_dir = os.path.dirname(os.path.dirname(__file__))
pose_control_dir = s=os.path.join(root_dir, "data", "quadruped_pose_control")

import pickle

class QuadrupedPoseControlData:
    def __init__(self) -> None:
        # Timestamps
        self.timestamps = []
        self.progress_step = []

        # Some observations
        self.base_positions = []
        self.base_quaternions = []
        self.joint_positions = []
        self.joint_velocities = []
        self.goal_quaternions = []
        self.quaternion_diff = []

        # Action information
        self.joint_position_targets = []
        self.actions = []
        self.observations = []

class QuadrupedPoseControlLog:
    def __init__(self, output_file_name):
        self.output_file_path = os.path.join(pose_control_dir, output_file_name)

        self.__pose_control_data = QuadrupedPoseControlData()

    def add_data(self,
                 timestamp,
                 progress_step,
                 base_position,
                 base_quaternion,
                 joint_positions,
                 joint_velocities,
                 goal_quaternion,
                 quaternion_diff,
                 joint_position_target,
                 action,
                 observation):
        self.__pose_control_data.timestamps.append(timestamp)
        self.__pose_control_data.progress_step.append(progress_step)
        
        self.__pose_control_data.base_positions.append(base_position)
        self.__pose_control_data.base_quaternions.append(base_quaternion)
        self.__pose_control_data.joint_positions.append(joint_positions)
        self.__pose_control_data.joint_velocities.append(joint_velocities)
        self.__pose_control_data.goal_quaternions.append(goal_quaternion)
        self.__pose_control_data.quaternion_diff.append(quaternion_diff)
        
        self.__pose_control_data.joint_position_targets.append(joint_position_target)
        self.__pose_control_data.actions.append(action)
        self.__pose_control_data.observations.append(observation)

    def save_data(self):
        with open(self.output_file_path, "wb+") as output_file:
            pickle.dump(self.__pose_control_data,
                        output_file)