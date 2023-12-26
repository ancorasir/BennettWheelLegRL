import os
import pandas as pd
import matplotlib.pyplot as plt

def read_and_plot_data(experiment_dir, data_type, env_ids, target_env):
    for env_id in env_ids:
        if target_env is not None and env_id != target_env:
            continue
            
        filename = os.path.join(experiment_dir, f'{data_type}_env{env_id}.csv')
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            plt.plot(df, label=df.columns)

    plt.title(f'{data_type} - Environment {target_env if target_env is not None else "All"}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    figures_dir = os.path.join(experiment_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    experiment_time = os.path.basename(experiment_dir)
    target_env_str = f'_env{target_env}' if target_env is not None else ''
    fig_filename = f'{experiment_time}_{data_type}{target_env_str}.png'
    fig_filepath = os.path.join(figures_dir, fig_filename)
    plt.savefig(fig_filepath)
    plt.close()
    print(f'Figure saved: {fig_filename}')




if __name__ == '__main__':
    env_ids = [0, 1]
    target_env = 0
    experiment_data_dir = "experiment_data"
    experiment_folder = "2023-12-26_01-24-23"
    experiment_dir = os.path.join(experiment_data_dir, experiment_folder)
    data_types = ['base_vel', 'command', 'contact_forces_z', 'joint_position', 'joint_torque', 'joint_vel']
    for data_type in data_types:
        read_and_plot_data(experiment_dir, data_type, env_ids, target_env)
    
