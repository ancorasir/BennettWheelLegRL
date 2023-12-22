from collections import deque
import time
import torch
import numpy as np

from base.task import Task
from controllers.blob_tracker import BlobTranslationTracker
from controllers.dynamixel_controller import QuadrupedPositionController
from controllers.imu_controller import QuadrupedIMUController

from rlg_policy_executor import RLGPolicyExecutor

class QuadrupedForwardLocomotion(Task):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.marker_height_offset = self.cfg.get("marker_height_offset", 0)
        # Create controllers
        self.blob_tracker = BlobTranslationTracker(self.cfg.get('tracker'))
        self.quadruped_controller = QuadrupedPositionController(self.cfg.get('controller'))
        self.imu = QuadrupedIMUController(self.cfg.get('imu'))

        # Config policy executors
        self.rlg_cfg = self.cfg.get('rlgames')
        self.train_config = self.rlg_cfg.get('train_config')
        self.trained_pth = self.rlg_cfg.get('trained_pth')
        self.task_config = self.rlg_cfg.get('task_config')
        self.num_actions = self.rlg_cfg.get('num_actions')
        self.num_obs = self.rlg_cfg.get('num_obs')
        self.device = self.rlg_cfg.get('device')
        self.is_deterministic = self.rlg_cfg.get('is_deterministic', True)
        self.seed = self.rlg_cfg.get('seed', None)
        self.policy_executor = None
        # Read task configs
        import yaml
        self.task_cfg = None
        with open(self.task_config, 'r') as task_config_f:
            self.task_cfg = yaml.safe_load(task_config_f)
        
        # Read obs weights
        self.body_height_scale = self.task_cfg['module']['body_height_scale']
        self.base_linear_v_scale = self.task_cfg['module']['base_linear_v_scale']
        self.base_angular_v_scale = self.task_cfg['module']['base_angular_v_scale']
        self.joint_position_scale = self.task_cfg['module']['joint_position_scale']
        self.joint_vel_scale = self.task_cfg['module']['joint_vel_scale']

        self.action_history = deque(maxlen=2)
        #self.action_history.appendleft()

        self.running = True

        self.__init_started = False
        self.__init_done = False
        self.__init_start_time = None
        self.__init_sleep_time = 1 # Sleep time in seconds

        self.last_actions = np.zeros(12, dtype=np.float32)
        self.current_cmd = None
        self.positions_upper = torch.tensor([np.pi], dtype=torch.float32, device=self.device).repeat(1, 12)
        self.positions_lower = -1*self.positions_upper

    def initialize(self):
        try:
            # Init policy
            self.policy_executor = RLGPolicyExecutor(self.train_config,
                                                 self.trained_pth,
                                                 self.num_actions,
                                                 self.num_obs,
                                                 is_deterministic=True,
                                                 device=self.device,
                                                 seed=self.seed)
            self.imu.initialize()
            self.blob_tracker.initialize()
            cmd = input("Input s to start...")
            if cmd[-1] == 's':
                self.quadruped_controller.initialize()
            else:
                self.stop()
                exit(0)

            # Start
            cmd = input("Input s to start task! ")
            if cmd[-1] == 's':
                for i in range(10):
                    self.logWarning("Task starts in {} seconds...".format(10-i))
                    time.sleep(1)
            else:
                self.stop()
                exit(0)

        except Exception as e:
            self.logCritical(str(e))
            exit(0)


    def stop(self):
        self.imu.close()
        self.blob_tracker.close()
        self.quadruped_controller.close()

    def step(self):
        if not self.__init_done:
            if not self.__init_started:
                self.__init_start_time = time.time()
                self.__init_started = True
            else:
                if time.time() - self.__init_start_time < self.__init_sleep_time:
                    time_start = time.time()
                    # Retrieve info from controllers
                    imu_info = self.imu.retrieveInfo()
                    tracker_info = self.blob_tracker.retrieveInfo()
                    controller_info = self.quadruped_controller.retrieveInfo()
                    print("Time for reading all info: ", time.time()-time_start)
                else:
                    self.__init_done = True
        else:
            '''Do some real stuff here'''
            if self.current_cmd is not None:
                # print("Execute actions")
                self.quadruped_controller.command(self.current_cmd)

            # Retrieve info from controllers
            try:
                imu_info = self.imu.retrieveInfo()
                tracker_info = self.blob_tracker.retrieveInfo()
                controller_info = self.quadruped_controller.retrieveInfo()
            except Exception as e:
                self.logCritical(str(e))
                self.stop()
                time.sleep(1)
                exit(0)

            # If tracker is crashed, just stop running
            if tracker_info is None:
                self.logCritical("Tracking service crashed, exiting...")
                self.running = False
            else:

                # Unpack info from imu
                projected_gravity = imu_info[0:3]
                base_angular_v = imu_info[3:6]
                # Unpack info from tracker
                pos = tracker_info[0:3]
                base_linear_v = tracker_info[3:6]
                body_height = pos[2] + self.marker_height_offset
                # Unpack info from quadruped controller
                joint_positions = controller_info[0:12]
                joint_velocities = controller_info[12:24]

                # print("body height: ", base_linear_v)
                # print("joint positions: ", joint_velocities)
                # print("projected gravity: ", projected_gravity)

                scaled_body_height = [self.body_height_scale * body_height]
                scaled_linear_v = self.base_linear_v_scale * base_linear_v
                scaled_angular_v = self.base_angular_v_scale * base_angular_v
                scaled_joint_positions = self.joint_position_scale * joint_positions
                scaled_joint_velocities = self.joint_vel_scale * joint_velocities

                obs_array = np.concatenate((scaled_linear_v,
                                            scaled_angular_v,
                                            scaled_joint_positions,
                                            scaled_joint_velocities,
                                            self.last_actions,
                                            projected_gravity,
                                            scaled_body_height), axis=0)
                obs_np = np.array([obs_array])
                obs_torch = torch.from_numpy(obs_np).to(dtype=torch.float32, 
                                                        device=self.device)
                
                raw_actions = self.policy_executor.getAction(obs_torch)

                # Unscale raw actions
                unscaled_positions = unscale_transform(raw_actions, 
                                                       self.positions_lower, 
                                                       self.positions_upper)
                position_cmd = unscaled_positions[0].cpu().numpy()
                self.current_cmd = position_cmd

                print("****")
                print(obs_array)
                print(position_cmd)

                # Update last actions
                self.last_actions = raw_actions[0].cpu().numpy()
             
@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset