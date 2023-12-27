import os
import pandas as pd
import math
import numpy as np

class DataAnalyser:
    def __init__(self) -> None:
        self.experiment_folder = "2023-12-27_23-58-20"
        self.experiment_data_dir = "experiment_data"
        self.experiment_dir = os.path.join(self.experiment_data_dir, self.experiment_folder)
        self.env_ids = list(range(50))
        self.time_step = 0.05
        

    def read_and_calculate_data(self, experiment_dir, env_ids, target_env):
        for env_id in env_ids:
            if target_env is not None and env_id != target_env:
                continue

            filename_joint_torque = os.path.join(experiment_dir, f'joint_torque_env{env_id}.csv')
            filename_joint_vel = os.path.join(experiment_dir, f'joint_vel_env{env_id}.csv')
            filename_base_vel = os.path.join(experiment_dir, f'base_vel_env{env_id}.csv')

            if os.path.exists(filename_joint_torque):
                df_joint_torques = pd.read_csv(filename_joint_torque)
            if os.path.exists(filename_joint_vel):
                df_joint_vel = pd.read_csv(filename_joint_vel)
            if os.path.exists(filename_base_vel):
                df_base_vel = pd.read_csv(filename_base_vel)

            distance = self.calculate_distance(df_base_vel)
            E_m = self.calculate_mechanical_energy_consumption(df_joint_torques, df_joint_vel)
            m = 3.578209936079509+0.7011939191100404+2.4365544485369104+0.2812248770831842+0.2812427717615869+0.7011939191100404+2.4365544485369104+0.2812248770831842+0.2812427717615869+0.7011939191100404+2.4365544485369104+0.28122487708301536+0.2812427717616486+0.7011939191100404+2.4365544485369104+0.28122487708301536+0.2812427717616486
            g = 9.81
            COT = E_m/m/g/distance



            print("*"*20)
            print("Robot number: ", target_env)
            # print("distance: ", distance)
            # print("E_m: ", E_m)
            # print("mass: ", m)
            print("COT: ", COT)
            print("*"*20)

            return COT




    def calculate_distance(self, df):
        total_distance = 0
        for index, row in df.iterrows():
        # Calculate the distance for each time step
            if index < df.shape[0]-130 or index > df.shape[0]-30:
                continue
            vel_x = row['base_vel_x']
            vel_y = row['base_vel_y']
            vel_z = row['base_vel_z']
            distance = self.time_step*math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)


            # print("*"*20)
            # print("vel_x: ", vel_x)
            # print("vel_y: ", vel_y)
            # print("vel_z: ", vel_z)
            # print("*"*20)

            # Add the distance to the total
            total_distance += distance
        return total_distance

    def calculate_mechanical_energy_consumption(self, df_joint_torques, df_joint_vel):
        energy = 0
        for index, row in df_joint_torques.iterrows():
        # Calculate the energy consumption for each time stepc
            if index < df_joint_torques.shape[0]-130 or index > df_joint_torques.shape[0]-30:
                continue
            power = 0
            for column in df_joint_torques.columns:
                if column.startswith('Joint_'):
                    torque = df_joint_torques.loc[index,column]
                    vel = df_joint_vel.loc[index,column]
                    power = power + abs(torque*vel)
                    # print("*"*20)
                    # print("Torque: ", torque)
                    # print("Vel: ", vel)
                    # print("Power: ",abs(torque*vel))
                    # print("Total power: ",power)
                    # print("*"*20)

            energy = energy+power*self.time_step
            # print("*"*20)
            # print("energy: ", energy)
            # print("*"*20)

        return energy

    def cal_CoT(self):
        total_COT = 0
        cot_values = []

        for target_env in self.env_ids:
            cot_value = self.read_and_calculate_data(self.experiment_dir, self.env_ids, target_env)
            cot_values.append(cot_value)

        cot_values = np.array(cot_values)
        filtered_values = self.exclude_outliers(cot_values)

        average_COT = np.mean(cot_values)
        average_COT_process = np.mean(filtered_values)

        print("average_COT (before excluding outliers): ", average_COT)
        print("average_COT (after excluding outliers): ", average_COT_process)
        return average_COT

    def exclude_outliers(self, data, sigma_threshold=3):
        z_scores = (data - np.mean(data)) / np.std(data)
        filtered_data = data[abs(z_scores) < sigma_threshold]
        return filtered_data


if __name__ == '__main__':
    d = DataAnalyser()
    d.cal_CoT()
    
