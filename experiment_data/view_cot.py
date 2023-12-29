import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data
actions = ['Forward', 'Left', 'Backward', 'Right']
speeds = ['1m/s', '0.5m/s']
average_COT_values = [
    [0.5046531633060511, 0.587507780037245, 0.5125283801821855, 0.546663401107746],
    [0.5499807027729322, 0.5672266012902617, 0.5494145357522562, 0.6604792018974952]
]

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Loop through speeds and actions to create bars
for i, speed in enumerate(speeds):
    for j, action in enumerate(actions):
        x = j  # x-coordinate (action)
        y = i  # y-coordinate (speed)
        z = average_COT_values[i][j]  # z-coordinate (average COT)

        # Plot the bar
        bar = ax.bar3d(x - 0.2, y - 0.2, 0, 0.4, 0.4, z, shade=True)

        # Add data label to the center of the bar
        ax.text(x + 0.1, y + 0.1, z + 0.4, f'{z:.2f}', color='black', ha='center', va='center', fontsize=8)

# Set labels and title
ax.set_xticks(range(len(actions)))
ax.set_xticklabels(actions)
ax.set_yticks(range(len(speeds)))
ax.set_yticklabels(speeds)
ax.set_zlabel('Average COT')
# ax.set_zticks(range(int(np.floor(np.max(average_COT_values)))+1))
ax.set_title('Average Cost of Transport (COT) for Different Actions and Speeds')

# ax.set_aspect('equal')

# Display the plot
plt.show()
