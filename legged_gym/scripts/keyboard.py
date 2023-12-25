import os
import sys

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'envs')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'utils')))

from pynput import keyboard
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot import LeggedRobot, LeggedRobotCfg
from legged_gym.utils.sim_utils import *
from legged_gym.utils.math import *
from legged_gym.utils.helpers import *
from collections.abc import Callable, Iterable, Mapping
from threading import Thread
from typing import *
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

import torch
import queue
import time



class keyboard_control_legged_robot(LeggedRobot):

    def _keyboard_commands(self, env_ids, on_keyboard:bool = True):
        if on_keyboard == False:
            super()._resample_commands(env_ids)
        else:
            self.commands[env_ids, 0] =1
            self.commands[env_ids,1] =0
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] =0
            else: 
                self.commands[env_ids, 2] =0
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        #self._resample_commands(env_ids)
        self._keyboard_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        #self._resample_commands(env_ids)
        self._keyboard_commands(env_ids)


        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()



    




class keyboard_commands:
    def __init__(self,
                 env_num) -> None:
        self.queue = queue.Queue()
        self.env_num = env_num
        self.num_commands = class_to_dict(LeggedRobotCfg.commands.num_commands)
        self.command = torch.ones((self.env_num, self.num_commands))
        self.commands = torch.zeros((self.env_num, self.num_commands))

    def key_press(self,
                  key):#定义按键按下时触发的函数
        
        lin_vel = 0.1
        ang_vel = 0.1
        
        try:
            if key.char == 'w':
                self.commands[:,0] += lin_vel
            elif key.char == 's':
                self.commands[:,0] -= lin_vel
            elif key.char == 'a':
                self.commands[:,1] += lin_vel
            elif key.char == 'd':
                self.commands[:,1] -= lin_vel
            elif key.char == 'q':
                self.commands[:,2] += ang_vel
            elif key.char == 'e':
                self.commands[:,2] -= ang_vel
            elif key.char == 'r':
                self.commands = torch.zeros((self.env_num, self.num_commands))

            self.command[self.command<1e-5]=0
            self.queue.put(self.commands)
        
        except:
            pass

    def key_release(self,key):
        try:
            pass
        except:
            pass

    def keyboard_control_on(self):
        
        with keyboard.Listener(on_press = self.key_press, 
                            on_release = self.key_release) as listener:
            while True:
                if not listener.running:
                    break
            
                #print(self.commands)

                time.sleep(0.1)



class env:
    def __init__(self) -> None:
        self.env = None
        self.queue = queue.Queue()
        self.commands = None
    
    def keyboard_test_env(self):
        sim_params = get_sim_params(dt=0.005, use_gpu_pipeline=True)

        BennettRoughCfg.env.num_envs = 60
        BennettRoughCfg.commands.heading_command = False

        self.env = keyboard_control_legged_robot(
            cfg=BennettRoughCfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device='cuda:0',
            headless=False
        )

        train_cfg_dict = class_to_dict(BennettRoughCfgPPO)

        train_runner = OnPolicyRunner(self.env, train_cfg_dict, './log', device='cuda:0')

        cur_path = os.path.dirname(__file__)
        # model_path = os.path.join(cur_path, '../../logs/rough_bennett/Dec20_17-47-57_/model_1400.pt')
        model_path = os.path.join(cur_path, '../../logs/rough_bennett/Dec22_13-48-55_/model_1400.pt')
        train_runner.load(model_path)
        print('[Info] Successfully load pre-trained model from {}.'.format(model_path))
        policy = train_runner.get_inference_policy(device='cuda:0')

        # action = 10 * torch.zeros(env_paras.num_envs, env_paras.num_actions, device=TorchParas().device)
        obs = self.env.get_observations()

        iter = 0
        while not self.env.gym.query_viewer_has_closed(self.env.viewer):
            actions = policy(obs.detach())
            obs, _, _, _, _ = self.env.step(actions.detach())
            if iter == 0:
                print(obs)
            iter += 1
        self.env.close()

    def put_commands(self,commands):
        self.queue.put(commands.to(self.env.device))
        self.env.commands=self.queue.get(timeout=0.1)






if __name__ == '__main__':
    
    shared_commands = queue.Queue()
    num_envs = 60
    
    test_env = env()
    keyboard_controller = keyboard_commands(num_envs)

    t1 = Thread(target=test_env.keyboard_test_env)
    t2 = Thread(target=keyboard_controller.keyboard_control_on)

    t1.start()
    t2.start()

    
    while True:
        try:
            commands = keyboard_controller.queue.get(timeout=0.1)
            test_env.put_commands(commands)

            shared_commands.put(commands)
            print(shared_commands.get(timeout=0.1))
        except:
            pass