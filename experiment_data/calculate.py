import os
import pandas as pd
import math

def read_and_calculate_data(experiment_dir, env_ids, target_env):
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

        distance = calculate_distance(df_base_vel)
        E_m = calculate_mechanical_energy_consumption(df_joint_torques, df_joint_vel)
        m = 3.578209936079509+0.7011939191100404+2.4365544485369104+0.2812248770831842+0.2812427717615869+0.7011939191100404+2.4365544485369104+0.2812248770831842+0.2812427717615869+0.7011939191100404+2.4365544485369104+0.28122487708301536+0.2812427717616486+0.7011939191100404+2.4365544485369104+0.28122487708301536+0.2812427717616486
        g = 9.81
        COT = E_m/m/g/distance


        
        print("*"*20)
        print("Robot number: ", target_env)
        print("distance: ", distance)
        print("E_m: ", E_m)
        print("mass: ", m)
        print("COT: ", COT)
        print("*"*20)

        return COT




def calculate_distance(df):
    total_distance = 0
    time_step = 0.01
    for index, row in df.iterrows():
    # Calculate the distance for each time step
        if index < 5 or index > df.shape[0]-5:
            continue
        vel_x = row['base_vel_x']
        vel_y = row['base_vel_y']
        vel_z = row['base_vel_z']
        distance = time_step*math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        

        # print("*"*20)
        # print("vel_x: ", vel_x)
        # print("vel_y: ", vel_y)
        # print("vel_z: ", vel_z)
        # print("*"*20)

        # Add the distance to the total
        total_distance += distance
    return total_distance
           
def calculate_mechanical_energy_consumption(df_joint_torques, df_joint_vel):
    time_step = 0.01
    energy = 0
    for index, row in df_joint_torques.iterrows():
    # Calculate the energy consumption for each time stepc
        if index < 5 or index > df_joint_torques.shape[0]-5:
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

        energy = energy+power*time_step
        # print("*"*20)
        # print("energy: ", energy)
        # print("*"*20)

    return energy

if __name__ == '__main__':
    env_ids = [0, 1]
    experiment_data_dir = "experiment_data"
    experiment_folder = "2023-12-27_22-40-32"
    experiment_dir = os.path.join(experiment_data_dir, experiment_folder)

    total_COT = 0
    for target_env in env_ids:
        total_COT = total_COT + read_and_calculate_data(experiment_dir, env_ids, target_env)

    average_COT = total_COT/len(env_ids)

    print("average_COT: ", average_COT)

    
    
    
