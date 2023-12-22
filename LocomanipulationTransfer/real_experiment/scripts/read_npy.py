import numpy as np
import os
current_dir = os.path.dirname(__file__)
import matplotlib.pyplot as plt # plotting library

npy_file = os.path.join(current_dir, "230626-planar66134-5.npy")
array =  np.load(npy_file)

position_y = array[:, 2]
num_array = array.shape[0]
  
plt.plot(range(num_array), position_y)
plt.show()

