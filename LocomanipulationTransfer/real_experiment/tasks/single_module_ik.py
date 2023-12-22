import time
from collections import deque
import torch
import numpy as np

from base.task import Task
from controllers.dynamixel_controller import QuadrupedPositionController
from utils.bennett_kinematics import forward_kinematics

from rlg_policy_executor import RLGPolicyExecutor

class SingleModuleIK(Task):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.time_interval_ms = self.cfg['time_interval_ms']
        self.time_interval_s = self.time_interval_ms/1000.0
        self.goal_angles = self.cfg['goal_joints']

        self.dynamixel_controller = QuadrupedPositionController(self.cfg.get('controller'))

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

        self.joint_position_scale = self.task_cfg['module']['obs_scale']['joint_positions']
        self.joint_velocity_scale = self.task_cfg['module']['obs_scale']['joint_velocities']
        self.tip_position_scale = self.task_cfg['module']['obs_scale']['tip_positions']
        self.tip_velocity_scale = self.task_cfg['module']['obs_scale']['tip_velocities']

        self.action_history = deque(maxlen=2)
        self.tip_position_history = deque(maxlen=2)

        self.goal_position = forward_kinematics(self.goal_angles[0],
                                                self.goal_angles[1],
                                                self.goal_angles[2])
        self.goal_position = np.array([self.goal_position[0,0], 
                                       self.goal_position[0,1], 
                                       self.goal_position[0,2]])

        self.running = True

        self.__init_started = False
        self.__init_done = False
        self.__init_start_time = None
        self.__init_sleep_time = 1 # Sleep time in seconds

        self.last_actions = np.zeros(3, dtype=np.float32)
        self.current_cmd = None
        self.positions_upper = torch.tensor([np.pi], dtype=torch.float32, device=self.device).repeat(1, 3)
        self.positions_lower = -1*self.positions_upper

    def initialize(self):
        # Init policy
        self.policy_executor = RLGPolicyExecutor(self.train_config,
                                                self.trained_pth,
                                                self.num_actions,
                                                self.num_obs,
                                                is_deterministic=True,
                                                device=self.device,
                                                seed=self.seed)
        
        self.dynamixel_controller.initialize()

        goal_pos_info = "Current goal position: " + str(self.goal_position)
        self.logInfo(goal_pos_info)

        # Start
        cmd = input("Input s to start task! ")
        if cmd[-1] == 's':
            sleep_seconds = 5
            for i in range(sleep_seconds):
                self.logWarning("Task starts in {} seconds...".format(sleep_seconds-i))
                time.sleep(1)
        else:
            self.stop()
            exit(0)

    def stop(self):
        self.dynamixel_controller.close()

    def step(self):
        if not self.__init_done:
            if not self.__init_started:
                self.__init_start_time = time.time()
                self.__init_started = True
            else:
                if time.time() - self.__init_start_time < self.__init_sleep_time:
                    time_start = time.time()
                    # Retrieve info from controllers
                    controller_info = self.dynamixel_controller.retrieveInfo()
                    print("Time for reading all info: ", time.time()-time_start)
                else:
                    self.__init_done = True
        else:
            '''Do some real stuff here'''
            if self.current_cmd is not None:
                # print("Execute actions")
                self.dynamixel_controller.command(self.current_cmd)

            # Retrieve info from controllers
            controller_info = self.dynamixel_controller.retrieveInfo()

            joint_positions = controller_info[0:3]
            joint_velocities = controller_info[3:6]
            
            # Get tip positions
            tip_pos = forward_kinematics(joint_positions[0],
                                         joint_positions[1],
                                         joint_positions[2])
            tip_pos = np.array([tip_pos[0,0], tip_pos[0,1], tip_pos[0,2]])

            # Get tip velocities
            if len(self.tip_position_history) == 0:
                # Not enough position history
                tip_velocities = np.array([0,0,0])
            else:
                tip_velocities = (tip_pos - self.tip_position_history[-1])/self.time_interval_s
            
            self.tip_position_history.appendleft(tip_pos)

            scaled_joint_positions = self.joint_position_scale * joint_positions
            scaled_joint_velocities = self.joint_velocity_scale * joint_velocities
            scaled_tip_positions = self.tip_position_scale * tip_pos
            scaled_tip_velocities = self.tip_velocity_scale * tip_velocities
            goal_obs = self.goal_position
            goal_pos_error_obs = tip_pos - goal_obs
            
            obs_array = np.concatenate((scaled_joint_positions,
                                        scaled_joint_velocities,
                                        scaled_tip_positions,
                                        scaled_tip_velocities,
                                        self.last_actions,
                                        goal_obs,
                                        goal_pos_error_obs), axis=0)
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
            print(joint_positions)
            print(joint_velocities)
            print(tip_pos)
            print(tip_velocities)
            print(goal_obs)
            print(goal_pos_error_obs)

            # print("****")
            # print(obs_array)
            # print(position_cmd)

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