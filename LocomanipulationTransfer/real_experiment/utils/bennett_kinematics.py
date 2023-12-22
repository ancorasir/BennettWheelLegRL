import os, sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

import numpy as np

from leg_kinematics_v2 import BennettLegKin3D
import torch

b = 0.1
alpha = np.pi/4.0
beta = 3*np.pi/4.0
L_o = 0.0568
L_s = 0.027

Tbs = np.matrix([[0,  0, 1, 0],
                 [0, -1, 0, 0],
                 [1,  0, 0, -0.1329],
                 [0,  0, 0, 1]])
# Tts = np.matrix([[1,  0, 0, 0],
#                  [0,  1, 0, 0],
#                  [0,  0, 1, 0.006],
#                  [0,  0, 0, 1]])
Tts = np.matrix([[1,  0, 0, 0],
                 [0,  1, 0, 0],
                 [0,  0, 1, 0.006],
                 [0,  0, 0, 1]])

bennett_leg = BennettLegKin3D(b, alpha, beta, L_o, L_s)

def forward_kinematics(q1, q2, q3):
    # Transform the q1, q2, q3 from omni
    if q1 >= np.pi/-2.0:
        q1_s = q1 + np.pi/2.0
    else:
        q1_s = q1 + 5.0*np.pi/2.0

    q2_s = q2 + np.pi
    q3_s = q3 + np.pi

    x_s, y_s, z_s = bennett_leg.forward_kin_threedof_Bennett(q1_s, q3_s, q2_s)

    p_s = np.matrix([[x_s], [y_s], [z_s], [1]])
    #print(p_s)
    # p_t = np.matmul(Tts, p_s)
    # p_t = p_s

    p_b = np.matmul(Tbs, p_s)[0:3]

    return p_b.transpose()

if __name__ == "__main__":
    q1 = 0
    q2 = np.pi/2.0
    q3 = 0

    pos = forward_kinematics(q1, q2, q3)

    print(pos)