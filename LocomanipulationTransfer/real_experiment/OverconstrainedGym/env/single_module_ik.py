from gym import spaces
import numpy as np
import torch
import time
from collections import deque
from typing import Tuple, Dict

from controllers.dynamixel_controller import QuadrupedPositionController
from utils.bennett_kinematics import forward_kinematics

class SingleModuleIK:
    def __init__(self, task_cfg):
        self.task_cfg = task_cfg
        self.omni_cfg = self.task_cfg['omni_cfg']

        self.dynamixel_controller = QuadrupedPositionController(self.task_cfg['controller'])
        
        self._num_envs = 1
        self._num_agents = 1
        self._num_observations = 21
        self._num_states = 21
        self._num_actions = 3

        # config obs/state/action spaces
        self.action_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.obs_buf = None
        self.states_buf = None
        self.reset_buf = None
        self.progress_buf = None
        self.rew_buf = None
        self.extras = None

        self.clip_obs = self.task_cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.task_cfg["env"].get("clipActions", np.Inf)
        self.time_interval_ms = self.task_cfg.get('time_interval_ms', 33.3)
        self.time_interval_s = self.time_interval_ms/1000.0
        self.last_exec_time = round(time.time()*1000, 2)
        self.max_episode_length = self.task_cfg['env']['maxEpisodeLength']
        self.device = self.task_cfg.get('device', 'cpu')
        self._dof1_range =  self.task_cfg.get('dof1_range')
        self._dof2_range = self.task_cfg.get('dof2_range')
        self._dof23_diff_range = self.task_cfg.get('dof23_diff_range')

        self.init_buffer()
        self.init_agent()

        # Read obs scales
        import yaml
        with open(self.omni_cfg, 'r') as task_config_f:
            self.omni_task_cfg = yaml.safe_load(task_config_f)
        self.joint_position_scale = self.omni_task_cfg['module']['obs_scale']['joint_positions']
        self.joint_velocity_scale = self.omni_task_cfg['module']['obs_scale']['joint_velocities']
        self.tip_position_scale = self.omni_task_cfg['module']['obs_scale']['tip_positions']
        self.tip_velocity_scale = self.omni_task_cfg['module']['obs_scale']['tip_velocities']
        # Reading rew scales
        self.goal_distance_rew_scale = self.omni_task_cfg['reward']['goal_distance_rew_scale']
        self.goal_distance_bonus = self.omni_task_cfg['reward']['goal_distance_bonus']
        self.goal_distance_bonus_thresh = self.omni_task_cfg['reward']['goal_distance_bonus_thresh']
        self.joint_vel_scale = self.omni_task_cfg['reward']['joint_vel_scale']
        self.joint_acc_scale = self.omni_task_cfg['reward']['joint_acc_scale']
        self.action_rate_scale = self.omni_task_cfg['reward']['action_rate_scale']

        self.action_history = deque(maxlen=2) # [current_actions, last_actions]
        self.action_history.appendleft(np.array([0,0,0]))
        self.action_history.appendleft(np.array([0,0,0]))
        self.tip_position_history = deque(maxlen=2)

        # obs buffers
        self.goal_position = [0,0,0]
        self.joint_positions = np.array([0,0,0])
        self.joint_velocities = np.array([0,0,0])
        self.tip_positions = np.array([0,0,0])
        self.tip_velocities = np.array([0,0,0])

        self.joint_acc = torch.zeros((self._num_envs, 3), device=self.device)
        self.joint_vel_history = deque(maxlen=2)
        self.joint_vel_history.append(np.array([0,0,0]))
        self.joint_vel_history.append(np.array([0,0,0]))

        self.last_action_time = round(time.time()*1000, 2)

    def pre_physics_step(self, actions):
        # currnet_action_time = round(time.time()*1000, 2)
        # time_elasped = currnet_action_time - self.last_action_time
        # self.last_action_time  = currnet_action_time
        # print(time_elasped)
        '''Execute the actions here'''
        # Reset the env if needed
        reset_buf = self.reset_buf.clone()
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_task(reset_env_ids)

        action_numpy = actions[0].cpu().numpy()
        # Command the controller
        self.dynamixel_controller.command(action_numpy) 

        self.action_history.appendleft(action_numpy)

    def get_observations(self):
        '''update obs buffer here'''
        controller_info = self.dynamixel_controller.retrieveInfo()
        self.joint_positions = controller_info[0:3]
        self.joint_velocities = controller_info[3:6]
        self.joint_vel_history.appendleft(self.joint_velocities) # [current, last]
        self.joint_acc = (self.joint_vel_history[0] - self.joint_vel_history[1])/self.time_interval_s

        # Get tip positions
        tip_pos = forward_kinematics(self.joint_positions[0],
                                     self.joint_positions[1],
                                     self.joint_positions[2])
        self.tip_positions = np.array([tip_pos[0,0], tip_pos[0,1], tip_pos[0,2]])

        # Get tip velocities
        if len(self.tip_position_history) == 0:
            # Not enough position history
            self.tip_velocities = np.array([0,0,0])
        else:
            self.tip_velocities = (self.tip_positions - self.tip_position_history[0])/self.time_interval_s
        self.tip_position_history.appendleft(self.tip_positions)

        # Scale observations
        scaled_joint_positions = self.joint_position_scale * self.joint_positions
        scaled_joint_velocities = self.joint_velocity_scale * self.joint_velocities
        scaled_tip_positions = self.tip_position_scale * self.tip_positions
        scaled_tip_velocities = self.tip_velocity_scale * self.tip_velocities
        goal_obs = self.goal_position
        goal_pos_error_obs = self.tip_positions - goal_obs

        # print("joint position: ", joint_positions)
        # print("joint velocities: ", joint_velocities)
        # print("tip position: ", tip_pos)
        # print("tip velocities: ", tip_velocities)
        # print(self.tip_position_history)
        # print(self.joint_velocities)
        # print(self.joint_acc)

        obs_array = np.concatenate((scaled_joint_positions,
                                    scaled_joint_velocities,
                                    scaled_tip_positions,
                                    scaled_tip_velocities,
                                    self.action_history[-1],
                                    goal_obs,
                                    goal_pos_error_obs), axis=0)
        obs_np = np.array([obs_array])
        self.obs_buf[:] = torch.from_numpy(obs_np).to(dtype=torch.float32, 
                                                device=self.device)
        self.states_buf[:] = torch.from_numpy(obs_np).to(dtype=torch.float32, 
                                                device=self.device)

    def get_states(self):
        '''update state buffer here (for asymmetric policy)'''
        pass

    def calculate_metrics(self):
        '''update rew buffer here'''
        torch_tip_positions = torch.tensor([self.tip_positions], device=self.device)
        torch_goal_position = torch.tensor([self.goal_position], device=self.device)
        torch_joint_velocities = torch.tensor([self.joint_velocities], device=self.device)
        torch_joint_accelerations = torch.tensor([self.joint_acc], device=self.device)
        torch_last_action = torch.tensor([self.action_history[-1]], device=self.device)
        torch_current_action = torch.tensor([self.action_history[0]], device=self.device)

        self.rew_buf[:], log_dict = calc_rewards(
            torch_tip_positions,
            torch_goal_position,
            torch_joint_velocities,
            torch_joint_accelerations,
            torch_last_action,
            torch_current_action,
            self.goal_distance_rew_scale,
            self.goal_distance_bonus,
            self.goal_distance_bonus_thresh,
            self.joint_vel_scale,
            self.joint_acc_scale,
            self.action_rate_scale
        )

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in log_dict.items()})

    def is_done(self):
        '''update reset buffer here'''
        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_task(self, env_ids):
        '''reset task here'''
        num_resets = len(env_ids)
        # reset goal position
        rand_range = torch.rand((num_resets, 3), device=self.device)
        self.rand_dof1 = (rand_range[:, 0] * (self._dof1_range[1] - self._dof1_range[0]) + self._dof1_range[0]).view(num_resets, 1)
        self.rand_dof2 = (rand_range[:, 1] * (self._dof2_range[1] - self._dof2_range[0]) + self._dof2_range[0]).view(num_resets, 1)
        rand_dof23_diff = (rand_range[:, 2] * (self._dof23_diff_range[1] - self._dof23_diff_range[0]) + self._dof23_diff_range[0]).view(num_resets, 1)
        self.rand_dof3 = self.rand_dof2 - rand_dof23_diff

        ''''For testing'''
        q1 = -0.1495
        q2 = 0.7300
        q3 = -0.8099
        self.rand_dof1 = torch.tensor([[q1]], device=self.device)
        self.rand_dof2 = torch.tensor([[q2]], device=self.device)
        self.rand_dof3 = torch.tensor([[q3]], device=self.device)

        rand_dof_positions = torch.cat((self.rand_dof1, self.rand_dof2, self.rand_dof3), dim=-1)

        goal_positions = torch.zeros((num_resets, 3), dtype=torch.float32, device=self.device)
        for row in range(0, rand_dof_positions.shape[0]):
            angles = rand_dof_positions[row, :].view(3).clone().cpu().numpy()
            np_goal_pos = forward_kinematics(angles[0], angles[1], angles[2])
            torch_goal_pos = torch.tensor(np_goal_pos, dtype=torch.float32, device=self.device)
            goal_positions[row, :] = torch_goal_pos
        self.goal_position = goal_positions.view(3).clone().cpu().numpy()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        print("New goal: {}; angles: {}. ".format(self.goal_position, rand_dof_positions))
    
    @property
    def num_envs(self):
        return self._num_envs
    @property
    def num_observations(self):
        return self._num_observations
    @property
    def num_states(self):
        return self._num_states
    @property
    def num_actions(self):
        return self._num_actions
    @property
    def num_agents(self):
        return self._num_agents
    
    def init_buffer(self):
        self.obs_buf = torch.zeros((self._num_envs, self.num_observations), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def init_agent(self):
        # Initialize the motors
        self.dynamixel_controller.initialize()

    def step(self, actions):
        actions = torch.clamp(actions,
                              -self.clip_actions,
                              self.clip_actions)
        self.pre_physics_step(actions)

        # TODO Wait for fixed interval of time
        while round(time.time()*1000, 2) - self.last_exec_time < self.time_interval_ms:
            time.sleep(0.0005)
        self.last_exec_time = round(time.time()*1000, 2)

        # post physics_steps (update buffers)
        self.post_physics_step()

        obs_dict = {'obs': self.obs_buf,
                    'states': self.states_buf}

        return obs_dict, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.progress_buf[:] += 1

        self.get_observations()
        self.get_states()
        self.calculate_metrics()
        self.is_done()

    def reset(self):
        print("Resetting task...")
        reset_ids = torch.tensor([0])
        self.reset_task(reset_ids)
        # Reset actions
        actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict

    def get_number_of_agents(self):
        return 1


@torch.jit.script
def calc_rewards(current_tip_position: torch.Tensor,
                 goal_tip_position: torch.Tensor,
                 joint_vel: torch.Tensor,
                 joint_acc: torch.Tensor,
                 last_action: torch.Tensor,
                 current_action: torch.Tensor,
                 goal_distance_rew_scale: float,
                 goal_distance_bonus: float,
                 bonus_threshold: float,
                 joint_vel_scale: float,
                 joint_acc_scale: float,
                 action_rate_scale: float
                 )-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                 
    '''goal distance reward'''
    goal_distance = torch.norm(current_tip_position - goal_tip_position, p=2, dim=-1)
    distance_rew = goal_distance_rew_scale * goal_distance
    # If distance is within certain range, give it a bonus
    give_bonus = torch.where(goal_distance < bonus_threshold, goal_distance_bonus*torch.ones_like(distance_rew), torch.zeros_like(distance_rew))
    give_higher_bonus = torch.where(goal_distance < bonus_threshold/2.0, 2*goal_distance_bonus*torch.ones_like(distance_rew), torch.zeros_like(distance_rew))
    distance_rew += give_bonus + give_higher_bonus

    '''joint velocity and joint acceleration penalty'''
    joint_vel_penalty = torch.sum(torch.abs(joint_vel)*joint_vel_scale, dim=1)
    joint_acc_penalty = torch.sum(torch.abs(joint_acc)*joint_acc_scale, dim=1)

    '''action rate penalty'''
    action_rate_penalty = torch.sum(torch.abs(current_action - last_action), dim=1) * action_rate_scale



    total_rew = distance_rew + \
                joint_vel_penalty + \
                joint_acc_penalty + \
                action_rate_penalty

    log_dict = {}
    log_dict['distance_rew'] = distance_rew
    log_dict['joint_vel_penalty'] = joint_vel_penalty
    log_dict['joint_acc_penalty'] = joint_acc_penalty
    log_dict['action_rate_penalty'] = action_rate_penalty

    return total_rew, log_dict