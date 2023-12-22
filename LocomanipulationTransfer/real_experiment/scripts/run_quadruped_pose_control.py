import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

import numpy as np

from controllers.quadruped_controller import QuadrupedPositionController
from controllers.quadruped_pose_reader import QuadrupedVispPoseReader
from controllers.policy import Policy
from tasks.quadruped_pose_control import QuadrupedPoseControlTask

DXL_PORT_NAME = "/dev/ttyUSB0"
DXL_BAUDRATE = 2000000
MOTOR_IDS = [10, 20, 30, 40,
             11, 12,
             21, 22,
             31, 32,
             41, 42]
INIT_JOINT_POS = [-1.57, 1.57, 1.57, -1.57,
                    -1.04, -2.09, 
                    2.09, 1.04, 
                    2.09, 1.04, 
                    -1.04, -2.09]
OPERAING_MODE = "current_based_position_control" # "current_based_position_control"
# PROFILE_VELOCITY = [125] * 12
# PROFILE_ACC = [20] * 12
# POSITION_P_GAIN = [200] * 12
# POSITION_D_GAIN = [200] * 12
# GOAL_CURRNT = [557] * 12

PROFILE_VELOCITY = [125] * 12
PROFILE_ACC = [70] * 12
POSITION_P_GAIN = [500] * 12
POSITION_D_GAIN = [100] * 12
GOAL_CURRNT = [557] * 12

T_MARKER_BASE = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -0.0775],
                          [0, 0, 0, 1]])
ORIGIN_CAMERA_TRANFORM = np.array([[ 0.99884784, -0.0010262 ,  0.04797819,  0.05502866],
       [ 0.02123072, -0.88716   , -0.46097335,  0.08488891],
       [ 0.04303738,  0.46146086, -0.8861161 ,  0.822894  ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
MARKER_LEN = 0.08
CHECKPOINT_PATH = "/home/bionicdl/SHR/LocomanipulationTransfer/" + \
                   "RobotLearning/omniisaacgymenvs/runs/" + \
                   "SKRL-QuadrupedPoseControlCustomControllerDR/" + \
                   "0531-sim2real-no_noise_test/checkpoints/best_agent.pt"

GOAL_QUATERNION = np.array([0.866, 0.0, 0.0, 0.500]) # np.array([0.866, 0.0, 0.0, 0.500]) # np.array([0.9689, 0.0, 0.0, -0.2474])
LOG_NAME = "0530-yaw_90-add_obs"

if __name__ == "__main__":
    quadruped_controller = QuadrupedPositionController(port_name=DXL_PORT_NAME,
                                                            baudrate=DXL_BAUDRATE,
                                                            motor_ids=MOTOR_IDS,
                                                            init_joint_pos=INIT_JOINT_POS)
    pose_reader = QuadrupedVispPoseReader(base_transform=T_MARKER_BASE,
                                          origin_camera_transform=ORIGIN_CAMERA_TRANFORM,
                                          marker_len=MARKER_LEN,
                                          server_ip="localhost",
                                          server_port=8081)
    policy = Policy(checkpoint_path=CHECKPOINT_PATH,
                    num_obs=82,
                    num_actions=12,
                    device="cuda")
    
    # Initialize pose reader
    pose_reader.initialize()

    import cv2

    

    # Initialize quadruped_controller
    quadruped_controller.init_quadruped(operating_mode=OPERAING_MODE,
                                        profile_velocity=PROFILE_VELOCITY,
                                        profile_acceleration=PROFILE_ACC,
                                        goal_current=GOAL_CURRNT,
                                        position_p_gain=POSITION_P_GAIN,
                                        position_d_gain=POSITION_D_GAIN)
    
    pose_control_task = QuadrupedPoseControlTask(quadruped_controller=quadruped_controller,
                                                 pose_reader=pose_reader,
                                                 goal_quaternion=GOAL_QUATERNION,
                                                 policy=policy,
                                                 num_obs=82,
                                                 num_actions=12,
                                                 log_file_name=LOG_NAME,
                                                 freq_hz=40,
                                                 episode_len=400)
    cv2.namedWindow("Test")
    key = cv2.waitKey(50)
    while key != ord('s'):
        print(pose_reader.get_pose_in_origin())
        key = cv2.waitKey(50)
    
    input("Enter anything to start running the task...")
    pose_control_task.run_task()

    # Close the controllers
    pose_control_task.close_task()
    pose_reader.close()