import os
current_dir = os.path.dirname(__file__)

obs_diff_file = os.path.join(current_dir, "0530-yaw_90-obs_diff")
output_diff_file = os.path.join(current_dir, "obs_diff")

import pickle
import numpy as np

if __name__ == "__main__":
    with open(obs_diff_file, "rb") as f:
        obs_diff = pickle.load(f)

    
    diff_sum = np.zeros(82)

    with open(output_diff_file, 'w+') as f:
        for diff_vec in obs_diff:
            rounded_diff = np.round(diff_vec, 2)
            f.write(str(rounded_diff) + "\n")
            diff_sum += diff_vec

    diff_avg = diff_sum / len(obs_diff)
    print(diff_avg)

    