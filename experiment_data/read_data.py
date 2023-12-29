import os
import pandas as pd
import matplotlib.pyplot as plt


# Read the specific type of data of a specific robot from a specific data folder and plot it.
def read_and_plot_data(experiment_dir, data_type, env_ids, target_env):

    for env_id in env_ids:

        # Find the target robot numer
        if target_env is not None and env_id != target_env:
            continue

        # Find the data file(.csv) for the target robot   
        filename = os.path.join(experiment_dir, f'{data_type}_env{env_id}.csv')

        # Confirm the existence of file
        if os.path.exists(filename):

            # Read data
            df = pd.read_csv(filename)

            # Plot every column of data labeled with its header
            plt.plot(df, label=df.columns)

    # Set title, xlabel, ylabel and legend
    plt.title(f'{data_type} - Environment {target_env if target_env is not None else "All"}')
    plt.xlabel('Time Steps(frequency: 20Hz)')
    plt.ylabel('Values')
    plt.legend()

    # Save the plots named after experiment time, data type and target robot number in the 'figures' folder
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
    # list robot numbers
    env_ids = [0, 1]

    # set the number of robot for the data we want to read 
    target_env = 0

    # set the data file path
    experiment_data_dir = "experiment_data"
    experiment_folder = "2023-12-28_01-43-31"
    experiment_dir = os.path.join(experiment_data_dir, experiment_folder)

    # set the data type we want
    data_types = ['base_vel', 'command', 'contact_forces_z', 'joint_position', 'joint_torque', 'joint_vel']

    # read all types of data
    for data_type in data_types:
        read_and_plot_data(experiment_dir, data_type, env_ids, target_env)
    
