import os
import pandas as pd
import math
import numpy as np

class DataAnalyser:
    def __init__(self) -> None:
        self.experiment_folder = "2023-12-28_00-02-06"
        self.experiment_data_dir = "experiment_data"
        self.experiment_dir = os.path.join(self.experiment_data_dir, self.experiment_folder)
        self.env_ids = list(range(50))
        self.time_step = 0.05
        

    def read_and_calculate_data(self, experiment_dir, env_ids, target_env):

        # Calculate MCOT of the target robot
        for env_id in env_ids:
            if target_env is not None and env_id != target_env:
                continue

            # find the data file path
            filename_joint_torque = os.path.join(experiment_dir, f'joint_torque_env{env_id}.csv')
            filename_joint_vel = os.path.join(experiment_dir, f'joint_vel_env{env_id}.csv')
            filename_base_vel = os.path.join(experiment_dir, f'base_vel_env{env_id}.csv')

            # Confirm the existence of data files and read data
            if os.path.exists(filename_joint_torque):
                df_joint_torques = pd.read_csv(filename_joint_torque)
            if os.path.exists(filename_joint_vel):
                df_joint_vel = pd.read_csv(filename_joint_vel)
            if os.path.exists(filename_base_vel):
                df_base_vel = pd.read_csv(filename_base_vel)

            # MCOT = E/(mgd)
            distance = self.calculate_distance(df_base_vel)
            E_m = self.calculate_mechanical_energy_consumption(df_joint_torques, df_joint_vel)
            m = 3.578209936079509+0.7011939191100404+2.4365544485369104+0.2812248770831842+0.2812427717615869+0.7011939191100404+2.4365544485369104+0.2812248770831842+0.2812427717615869+0.7011939191100404+2.4365544485369104+0.28122487708301536+0.2812427717616486+0.7011939191100404+2.4365544485369104+0.28122487708301536+0.2812427717616486
            g = 9.81
            MCOT = E_m/m/g/distance

            # RCOT = P/(mgwzd)
            L = 0.511/2
            W = 0.125/2
            D = math.sqrt(L**2 + W**2)
            total_RCOT = 0
            for index, row in df_base_vel.iterrows():
                omega = np.abs(row['base_vel_yaw'])
                power = 0
                for column in df_joint_torques.columns:
                    if column.startswith('Joint_'):
                        torque = df_joint_torques.loc[index,column]
                        vel = df_joint_vel.loc[index,column]
                        power = power + abs(torque*vel)
                total_RCOT = total_RCOT + power / (m*g*omega*D)

            RCOT = total_RCOT/df_base_vel.shape[0]

            print("*"*20)
            print("Robot number: ", target_env)
            # print("distance: ", distance)
            # print("E_m: ", E_m)
            # print("mass: ", m)
            print("MCOT: ", MCOT)
            print("RCOT: ", RCOT)

            print("*"*20)

            return MCOT, RCOT
        
  




    def calculate_distance(self, df):
        total_distance = 0

        for index, row in df.iterrows():
        # Calculate the distance for each time step
            
            # Select a suitable piece of data to calculate
            # if index < df.shape[0]-200 or index > df.shape[0]-100:
            if index < 230 or index > 330:
                
                continue
            vel_x = row['base_vel_x']
            vel_y = row['base_vel_y']
            vel_z = row['base_vel_z']

            # Distance = time_step * velocity
            distance = self.time_step*math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)


            # print("*"*20)
            # print("vel_x: ", vel_x)
            # print("vel_y: ", vel_y)
            # print("vel_z: ", vel_z)
            # print("*"*20)

            # Add the distance to the total
            total_distance += distance
        return total_distance

    # Calculate mechanical energy consumption with torques and velocities of all joints
    def calculate_mechanical_energy_consumption(self, df_joint_torques, df_joint_vel):
        energy = 0

        
        for index, row in df_joint_torques.iterrows():
        # Calculate the energy consumption for each time stepc
            
            # Select a suitable piece of data to calculate
            # if index < df_joint_torques.shape[0]-200 or index > df_joint_torques.shape[0]-100:
            if index < 230 or index > 330:
                continue
            power = 0

            # Set the absolute value of the product of torque and velocity for every joint and sum them up
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
            
            # Get the mechanical energy consumption
            energy = energy+power*self.time_step
            # print("*"*20)
            # print("energy: ", energy)
            # print("*"*20)

        return energy
    
    # Calculate MCOT
    def cal_COT(self):
        MCOT_values = []
        RCOT_values = []


        # Calculate MCOT and RCOT for all robots and get the average value
        for target_env in self.env_ids:
            MCOT_value, RCOT_value = self.read_and_calculate_data(self.experiment_dir, self.env_ids, target_env)
            MCOT_values.append(MCOT_value)
            RCOT_values.append(RCOT_value)


        MCOT_values = np.array(MCOT_values)
        RCOT_values = np.array(RCOT_values)


        # Exclude the outliers of MCOT data
        MCOT_filtered_values = self.exclude_outliers(MCOT_values)        
        
        # Exclude the outliers of RCOT dataiers(RCOT_values)
        RCOT_filtered_values = self.exclude_outliers(RCOT_values)

        # Get the average value of MCOT
        average_MCOT = np.mean(MCOT_values)
        average_MCOT_process = np.mean(MCOT_filtered_values)

        # Get the average value of RCOT
        average_RCOT = np.mean(RCOT_values)
        average_RCOT_process = np.mean(RCOT_filtered_values)


        print("average_MCOT (before excluding outliers): ", average_MCOT)
        print("average_MCOT (after excluding outliers): ", average_MCOT_process)

        print("average_RCOT (before excluding outliers): ", average_RCOT)
        print("average_RCOT (after excluding outliers): ", average_RCOT_process)

        return average_MCOT, average_RCOT

    # Exclude the outliers of data
    def exclude_outliers(self, data, sigma_threshold=3):
        z_scores = (data - np.mean(data)) / np.std(data)
        filtered_data = data[abs(z_scores) < sigma_threshold]
        return filtered_data


if __name__ == '__main__':
    d = DataAnalyser()
    d.cal_COT()
    
