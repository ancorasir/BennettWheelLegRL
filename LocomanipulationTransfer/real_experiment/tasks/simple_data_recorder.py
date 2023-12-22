from inspect import stack
import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "controllers"))
from imu_controller import IMUController
from marker_detector import ArucoDetector

import numpy as np
from pynput import keyboard
import time

IMU_FREQ = 200
MARKER_SIZE = 0.039 # in meters

IMU_CALI_MATRIX = np.array([[-0.59262,     0.805464,    0.00538908],
                        [-0.805481,   -0.592619,   -0.00187102],
                        [ 0.00168663, -0.0054496,   0.999984  ]])

CALI_MATRIX = np.array([[ 0.9987751,   0.04906245, -0.00640792,  0.22232355],
                        [ 0.03097743, -0.72102314, -0.6922182,   0.35085803],
                        [-0.03858218,  0.6911718,  -0.72165984,  0.61024565],
                        [ 0.,          0.,          0.,          1.        ]])

OUTPUT_FILE_NAME = '221014_bennett_100_100_forward_v10.csv'

def keyboardCallback(key):
    global key_pressed
    key_pressed = True

def startKeyboardListener():
    global key_pressed
    keyboard_listener = keyboard.Listener(on_press=keyboardCallback)
    keyboard_listener.start()
    key_pressed = False

if __name__ == "__main__":
    global key_pressed

    output_file = os.path.join(root_dir, "data", OUTPUT_FILE_NAME)

    imu = IMUController(sampling_freq=IMU_FREQ, transform_rot_mat=IMU_CALI_MATRIX)
    aruco_detector = ArucoDetector(MARKER_SIZE, MARKER_SIZE, cali_matrix=CALI_MATRIX)

    imu.initialize()
    aruco_detector.initialize()

    print("Enter s to start. ")
    key_input = input()
    if key_input[-1] != "s":
        print("exiting...")
        exit(1)

    print("Gathering data...Press any key to stop recording.")
    startKeyboardListener()
    with open(output_file, 'w') as f:
        while not key_pressed:
            timestamp = round(time.time()*1000)
            euler_angles = imu.retrieveInfo()

            transform_mat = aruco_detector.retrieveInfo()
            # Transform the data into one-dimensional array
            if transform_mat is not None:
                tvec = transform_mat[0:3, 3]
            else:
                tvec = [0,0,0]
            #flattened_vector = tvec.flattened()
            
            stacked_array = np.append(euler_angles, tvec)
            stacked_array = [str(i) for i in stacked_array]
            array_str = ",".join(stacked_array)

            # Convert stacked_array to comma-seperated string
            output_str = str(timestamp) + "," + array_str + "\n"
            
            f.write(output_str)
    
    imu.close()
    aruco_detector.close()
    