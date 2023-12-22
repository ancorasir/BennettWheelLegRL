import sys, os
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from tasks.trajectory_tracker import QuadrupedTrajectoryTracker
from controllers.quadruped_controller import QuadrupedPositionController

DYNAMIXEL_PORT_NAME = "/dev/ttyUSB0"
DYNAMIXEL_BAUDRATE = 2000000
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
OPERAING_MODE = "position_control"
PROFILE_VELOCITY = [125] * 12
PROFILE_ACC = [0] * 12
POSITION_P_GAIN = [800] * 12
POSITION_D_GAIN = [0] * 12
GOAL_CURRNT = [0] * 12

TRAJ_FILE_NAME = "traj_turning.txt"
OUTPUT_FILE_NAME = "0528-test"

if __name__ == "__main__":
    traj_folder = os.path.join(root_dir, "data", "trajectory_tracking")
    traj_file_path = os.path.join(traj_folder, TRAJ_FILE_NAME)
    output_file_path = os.path.join(traj_folder, OUTPUT_FILE_NAME)

    quadruped_controller = QuadrupedPositionController(port_name=DYNAMIXEL_PORT_NAME,
                                                       baudrate=DYNAMIXEL_BAUDRATE,
                                                       motor_ids=MOTOR_IDS,
                                                       init_joint_pos=INIT_JOINT_POS)
    
    quadruped_controller.update_joint_states()
    print(quadruped_controller.joint_positions)

    quadruped_controller.init_quadruped(operating_mode=OPERAING_MODE,
                                        profile_velocity=PROFILE_VELOCITY,
                                        profile_acceleration=PROFILE_ACC,
                                        position_p_gain=POSITION_P_GAIN,
                                        position_d_gain=POSITION_D_GAIN,
                                        goal_current=GOAL_CURRNT)
    
    input("Enter to begin trajectory tracking. ")
    
    trajectory_tracking_task = QuadrupedTrajectoryTracker(trajectory_file=traj_file_path,
                                                          quadruped_controller=quadruped_controller,
                                                          output_file_path=output_file_path,
                                                          freq_hz=40,
                                                          episode_len=400)
    
    trajectory_tracking_task.run_task()

    # while True:
    #     quadruped_controller.update_joint_states()
    #     print(quadruped_controller.joint_velocities)