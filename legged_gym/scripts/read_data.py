import pandas as pd
import matplotlib.pyplot as plt

def read_and_plot_torques(csv_filename):
    # Read torques from CSV file
    df = pd.read_csv(csv_filename)

    # Plot each torque column
    for column in df.columns:
        if column.startswith('Torque_'):
            plt.plot(df[column], label=column)

    plt.xlabel('Time Step')
    plt.ylabel('Torques')
    plt.title('Torques Over Time')
    plt.legend()
    plt.show()

# Usage example:
read_and_plot_torques('torques.csv')