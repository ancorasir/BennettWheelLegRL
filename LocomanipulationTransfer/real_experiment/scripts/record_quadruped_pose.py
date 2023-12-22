import numpy as np
import cv2
import os
current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)
import sys
sys.path.append(root_dir)

from controllers.quadruped_pose_reader import QuadrupedVispPoseReader
import time

FILE_NAME = "230626-planar66134-6.npy"

T_MARKER_BASE = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, -0.0775],
                          [0, 0, 0, 1]])
ORIGIN_CAMERA_TRANFORM = np.array([[ 0.9988251 , -0.008975  ,  0.0476222 ,  0.05184546],
       [ 0.01393173, -0.8880351 , -0.45956457,  0.08186829],
       [ 0.04641477,  0.45968807, -0.8868667 ,  0.82554126],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
MARKER_LEN = 0.08

pose_reader = QuadrupedVispPoseReader(base_transform=T_MARKER_BASE,
                                          origin_camera_transform=ORIGIN_CAMERA_TRANFORM,
                                          marker_len=MARKER_LEN,
                                          server_ip="localhost",
                                          server_port=8081)

if __name__ == "__main__":
    cv2.namedWindow("control")
    pose_reader.initialize()

    file_path = os.path.join(current_dir, FILE_NAME)

    pose_array = []

    key = cv2.waitKey(1)
    first_frame = True
    while key != ord("s"):
       position, quaternion = pose_reader.get_pose_in_origin()
       pose = [time.time(), position[0], position[1], position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
       pose_array.append(pose)
       key = cv2.waitKey(1)

    print(np.array(pose_array))

    np.save(file_path, np.array(pose_array))