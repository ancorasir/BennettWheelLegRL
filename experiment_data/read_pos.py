import pandas as pd
import matplotlib.pyplot as plt

def read_and_plot_pos(csv_filename):
    # Read torques from CSV file
    df = pd.read_csv(csv_filename)

    # Plot each torque column
    i = 0
    for column in df.columns:
        i=i+1
        if column.startswith('Torque_'):
            plt.subplot(4,3,i)
            plt.plot(df[column])
            plt.xlabel('Time Step')
            plt.ylabel('Position')
            plt.title('Positions Over Time')

    plt.show()

# Usage example:
read_and_plot_pos('pos.csv')