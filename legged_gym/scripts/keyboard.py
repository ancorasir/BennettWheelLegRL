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
from datetime import datetime
import csv



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

    def close(self):
        # sys.exit()
        os._exit(0)





class keyboard_commands:
    def __init__(self,
                 env_num) -> None:
        self.queue = queue.Queue()
        self.env_num = env_num
        self.num_commands = class_to_dict(LeggedRobotCfg.commands.num_commands)
        self.command = torch.ones((self.env_num, self.num_commands))
        self.commands = torch.zeros((self.env_num, self.num_commands))
        self.reset = False

    def key_press(self,
                  key):#定义按键按下时触发的函数
        
        lin_vel = 0.1
        ang_vel = 0.1
        
        # set velocity limit
        lin_max = 1.0
        lin_min = -1.0

        ang_max = 1.0
        ang_min = -1.0

        self.reset = False

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
            elif key.char == '0':
                self.reset = True


            # Check velocity limits
            self.commands[:, 0] = torch.clamp(self.commands[:, 0], lin_min, lin_max)
            self.commands[:, 1] = torch.clamp(self.commands[:, 1], lin_min, lin_max)
            self.commands[:, 2] = torch.clamp(self.commands[:, 2], ang_min, ang_max)

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
    
    def get_reset_flag(self):
        return self.reset
    
    def move_forward(self, vel):
        self.commands[:, 0] = np.ones(self.commands[:, 0]) * vel
        self.commands[:, 1] = np.zeros(self.commands[:, 1])
        self.commands[:, 2] = np.zeros(self.commands[:, 2]) 

        self.queue.put(self.commands)





class env:
    def __init__(self, num_envs, reset_time=1e6) -> None:
        self.env = None
        self.num_envs = num_envs
        self.queue = queue.Queue()
        self.commands = None
        self.reset_time = reset_time

        
        self.positions = []
        self.velocities = []     
        self.shared_torques = [] # To store torques

        self.exp_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')




    
    def keyboard_test_env(self):
        sim_params = get_sim_params(dt=0.005, use_gpu_pipeline=True)

        BennettRoughCfg.env.num_envs = self.num_envs
        BennettRoughCfg.env.episode_length_s = self.reset_time
        BennettRoughCfg.commands.resampling_time = self.reset_time
        BennettRoughCfg.commands.heading_command = False
        BennettRoughCfg.terrain.num_rows = 10
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        BennettRoughCfg.terrain.num_cols = 5
        # BennettRoughCfg.terrain.curriculum = False

        # # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        BennettRoughCfg.terrain.mesh_type = "trimesh"
        BennettRoughCfg.terrain.terrain_proportions = [0.0001, 0.0001, 0.9996, 0.0001, 0.0001]
        BennettRoughCfg.terrain.vertical_scale = 0.0025
        # BennettRoughCfg.terrain.selected = True
        # BennettRoughCfg.terrain.terrain_kwargs = {
        #     'type': 'rough_slope',
        #     'slope': 0.2,
        #     'step_height': 0.1
        # }
        
        # BennettRoughCfg.terrain.horizontal_scale = 0.01

        # BennettRoughCfg.terrain.border_size = 25

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
        model_path = os.path.join(cur_path, '../../logs/rough_bennett/Dec22_13_terrain/model_5000.pt')
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

    def restart_env(self, flag=False):
        if flag is True:
            self.env.reset()
      
    def flatten_list(self, nested_list):
        return [item for sublist in nested_list for item in sublist]


    def save_to_csv(self, filename, data):
        # print("data: {}".format(data))
        flattened_data = [self.flatten_list(row) for row in data]
        # print("flattened data: {}".format(flattened_data))



        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            

            # Write header
            writer.writerow([f'Torque_{i+1}' for i in range(len(flattened_data[0]))])
            # Write torques data
            writer.writerows(flattened_data)


    def save_to_incremental_csv(self, data, experiment_time, base_filename='data', env_ids=None):
        """
        Save NumPy array data to incremental CSV files for each environment.

        Parameters:
        - data (np.ndarray): NumPy array with shape (N, M), where N is the number of environments and M is the number of joints.
        - experiment_time (str): Experiment time to use for creating subfolders. If None, current time will be used.
        - base_filename (str): Base filename for CSV files. The actual filenames will be of the form 'base_filename_envID.csv'.
        - env_ids (list): List of environment IDs to save. If None, save all environments in data.

        Returns:
        - None
        """
        if env_ids is None:
            env_ids = range(data.shape[0])

        # if experiment_time is None:
        #     experiment_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Create the experiment_data directory
        experiment_dir = os.path.join(os.getcwd(), 'experiment_data', experiment_time)
        os.makedirs(experiment_dir, exist_ok=True)

        for env_id in env_ids:
            filename = os.path.join(experiment_dir, f'{base_filename}_env{env_id}.csv')
            write_header = not os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if write_header:
                    # Write header only if the file is created
                    writer.writerow([f'Joint_{i+1}' for i in range(data.shape[1])])

                # Write the data for the current environment
                writer.writerows(data[env_id:env_id+1, :])

    def save_contact_forces_z_to_incremental_csv(self, data, experiment_time, base_filename='contact_forces_z'):
        """
        Save NumPy array data to incremental CSV files for each environment.

        Parameters:
        - data (np.ndarray): NumPy array with shape (N, M), where N is the number of environments and M is the number of joints.
        - experiment_time (str): Experiment time to use for creating subfolders. If None, current time will be used.
        - base_filename (str): Base filename for CSV files. The actual filenames will be of the form 'base_filename_envID.csv'.
        - env_ids (list): List of environment IDs to save. If None, save all environments in data.

        Returns:
        - None
        """
        
        # if experiment_time is None:
        #     experiment_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Create the experiment_data directory
        experiment_dir = os.path.join(os.getcwd(), 'experiment_data', experiment_time)
        os.makedirs(experiment_dir, exist_ok=True)

        for env_id in range(data.shape[0]):
            filename = os.path.join(experiment_dir, f'{base_filename}_env{env_id}.csv')
            write_header = not os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if write_header:
                    # Write header only if the file is created
                    writer.writerow([f'feet_{i+1}' for i in range(data.shape[1])])

                # Write the data for the current environment
                writer.writerows(data[env_id:env_id+1, :, 2])

            

    def save_data_periodically(self, save_interval_seconds=0.05):
        last_save_time = time.time()

        while True:
            # Sleep until the next save interval
            elapsed_time = time.time() - last_save_time
            time_to_sleep = max(0, save_interval_seconds - elapsed_time)
            time.sleep(time_to_sleep)

            # Update the last save time
            last_save_time = time.time()

            if test_env.env:
                # Your existing data saving code here
                pos = test_env.env.dof_pos.cpu().numpy()
                vel = test_env.env.dof_vel.cpu().numpy()
                torques = test_env.env.torques.cpu().numpy()
                commands = test_env.env.commands.cpu().numpy()
                base_lin_vel = test_env.env.base_lin_vel.cpu().numpy()
                base_ang_vel = test_env.env.base_ang_vel.cpu().numpy()
                contact_forces_z = test_env.env.contact_forces[:, test_env.env.feet_indices, :].cpu().numpy()

                self.save_to_incremental_csv(data=torques, experiment_time=self.exp_time, base_filename="joint_torque")
                self.save_to_incremental_csv(data=pos, experiment_time=self.exp_time, base_filename="joint_position")
                self.save_to_incremental_csv(data=vel, experiment_time=self.exp_time, base_filename="joint_vel")
                self.save_command_to_incremental_csv(data=commands, experiment_time=self.exp_time)
                self.save_base_vel_to_incremental_csv(data1=base_lin_vel, data2=base_ang_vel, experiment_time=self.exp_time)
                self.save_contact_forces_z_to_incremental_csv(data=contact_forces_z, experiment_time=self.exp_time)

    def save_command_to_incremental_csv(self, data, experiment_time, base_filename='command'):
        """
        Save NumPy array data with shape (N, 3) to incremental CSV files for each environment.

        Parameters:
        - data (np.ndarray): NumPy array with shape (N, 3), where N is the number of environments, and columns are x, y, yaw.
        - experiment_time (str): Experiment time to use for creating subfolders. If None, current time will be used.
        - base_filename (str): Base filename for CSV files. The actual filenames will be of the form 'base_filename_envID.csv'.

        Returns:
        - None
        """
        

        # Create the experiment_data directory
        experiment_dir = os.path.join(os.getcwd(), 'experiment_data', experiment_time)
        os.makedirs(experiment_dir, exist_ok=True)

        for env_id in range(data.shape[0]):
            filename = os.path.join(experiment_dir, f'{base_filename}_env{env_id}.csv')
            write_header = not os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if write_header:
                    # Write header only if the file is created
                    writer.writerow(['command_x', 'command_y', 'command_yaw'])

                # Write the data for the current environment
                writer.writerow(data[env_id, 0:3])

    def save_base_vel_to_incremental_csv(self, data1, data2, experiment_time, base_filename='base_vel'):
        """
        Save NumPy array data with shape (N, 3) to incremental CSV files for each environment.

        Parameters:
        - data (np.ndarray): NumPy array with shape (N, 3), where N is the number of environments, and columns are x, y, yaw.
        - experiment_time (str): Experiment time to use for creating subfolders. If None, current time will be used.
        - base_filename (str): Base filename for CSV files. The actual filenames will be of the form 'base_filename_envID.csv'.

        Returns:
        - None
        """
        

        # Create the experiment_data directory
        experiment_dir = os.path.join(os.getcwd(), 'experiment_data', experiment_time)
        os.makedirs(experiment_dir, exist_ok=True)

        for env_id in range(data1.shape[0]):
            filename = os.path.join(experiment_dir, f'{base_filename}_env{env_id}.csv')
            write_header = not os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if write_header:
                    # Write header only if the file is created
                    writer.writerow(['base_vel_x', 'base_vel_y', 'base_vel_z', 'base_vel_yaw'])

                # Write the data for the current environment
                writer.writerow([data1[env_id, 0], data1[env_id, 1], data1[env_id, 2], data2[env_id,2]])




if __name__ == '__main__':
    
    shared_commands = queue.Queue()
    num_envs = 2

    
    test_env = env(num_envs)
    keyboard_controller = keyboard_commands(num_envs)

    t1 = Thread(target=test_env.keyboard_test_env)
    t2 = Thread(target=keyboard_controller.keyboard_control_on)
    
    # thread to save torques periodically
    t3 = Thread(target=test_env.save_data_periodically)  # Change 60 to your desired interval in seconds


    t1.start()
    t2.start()
    t3.start()

    
    while True:
        try:
            # fix the task to move forward
            # keyboard_controller.move_forward(0)


            commands = keyboard_controller.queue.get(timeout=0.1)

            reset_flag = keyboard_controller.get_reset_flag()
            test_env.restart_env(reset_flag)
            

            test_env.put_commands(commands)


            # print("torques are {}".format(test_env.env.torques))

            # torques = test_env.env.torques.cpu().numpy()  # Convert torch tensor to numpy array
            # shared_torques.append(torques)

            # print("*"*10)
            # print(shared_torques)

            shared_commands.put(commands)
            print(shared_commands.get(timeout=0.1))
         
            test_env.save_to_csv('torques.csv', test_env.shared_torques)
            # print("Size of shared_torques: {}".format(len(test_env.shared_torques)))s

            test_env.save_to_csv('pos.csv', test_env.positions)
            # print("Size of positions: {}".format(len(test_env.positions)))
            print("#"*20)
            print("save file")
        


        except:
            
            pass