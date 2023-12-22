import time
import socket 
import numpy as np

import cv2.aruco as aruco
import cv2
from scipy.spatial.transform import Rotation as R

class QuadrupedMocapPoseReader:
    def __init__(self, 
                 server_ip="localhost", 
                 server_port=8081) -> None:
        self.server_ip = server_ip
        self.server_port = server_port

        self.socket = socket.socket()
        self.socket.connect((self.server_ip, self.server_port))

    def get_pose(self):
        self.socket.send(b"test")
        data = self.socket.recv(1024).decode('utf-8')

        # Convert data to array
        data_array = eval(data)

        position = data_array[0:3]
        quaternion = data_array[3:7]
        linear_vel = data_array[7:10]
        angular_vel = data_array[10:13]

        return position, quaternion, linear_vel, angular_vel
           
    def close(self):
        self.socket.close()

class QuadrupedVispPoseReader:

    def __init__(self, 
                 base_transform=None,
                 origin_camera_transform=None,
                 marker_len=0.08,
                 server_ip="localhost", 
                 server_port=8081) -> None:
        """
            base_transform: transformation from the marker origin, to the actual origin of robot base;
            origin_camera_transform: the transformation from the camera to the world frame origin
        """
        self.T_marker_base = base_transform
        if base_transform is None:
            # Default to identity matrix, meaning that the actual base origin is just the marker frame origin
            self.T_marker_base = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], dtype=np.float32)

        self.origin_camera_transform = origin_camera_transform
        if self.origin_camera_transform is not None:
            # Get inverse
            self.T_origin_camera = np.linalg.inv(self.origin_camera_transform)

        self.marker_len = marker_len
        self.server_ip = server_ip
        self.server_port = server_port

        self.socket = socket.socket()

        from controllers.realsense_controller import RealsenseRGBController
        self.__camera_width = 960
        self.__camera_height = 540
        self.__fake_realsense = RealsenseRGBController(self.__camera_width,
                                                       self.__camera_height)
        self.__camera_intrinsics = None
        self.__intrinsic_matrix = None
        self.__camera_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # A queue used to approximate velocity
        from collections import deque
        self.__deque_size = 5
        # [eldest, ..., lastest]
        self.__timestamp_deque = deque(maxlen=self.__deque_size)
        self.__position_deque = deque(maxlen=self.__deque_size)
        self.__quaternion_deque = deque(maxlen=self.__deque_size)

    def initialize(self):
        # Read camera intrinsic
        self.__fake_realsense.initialize()
        self.__camera_intrinsics = self.__fake_realsense.get_intrinsics()
        self.__fake_realsense.close()
        fx = self.__camera_intrinsics.fx
        fy = self.__camera_intrinsics.fy
        ppx = self.__camera_intrinsics.ppx
        ppy = self.__camera_intrinsics.ppy

        self.__intrinsic_matrix = np.array([[fx, 0, ppx],
                                          [0,  fy, ppy],
                                          [0,  0,  1]])
        
        # Waiting for the server to start
        start = input("Start the server, and input s after initializing the server1...\n")
        if start[-1] != 's':
            print("exiting...")
            exit(0)
        print("Connecting to the server...")
        try:
            self.socket.connect((self.server_ip, self.server_port))
        except:
            print("Failed to connect to the server...")
            exit(0)

    def get_corners(self):
        self.socket.send(b"test")
        data = self.socket.recv(1024).decode('utf-8')
        
        eval_data = eval(data)

        timestamp_ms = eval_data[0]
        corners = eval_data[1]

        return timestamp_ms, corners

    def get_pose_in_camera(self):
        import cv2.aruco as aruco
        import cv2
        timestamp_ms, corners = self.get_corners()
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(np.array([corners]), 
                                                        self.marker_len, 
                                                        self.__intrinsic_matrix, 
                                                        self.__camera_dist) 
        
        if rvec is not None:
            (rvec-tvec).any() # get rid of that nasty numpy value array error
            R = cv2.Rodrigues(rvec[0])[0]
            T = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,1]], dtype=np.float32)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec[0]
            return T, timestamp_ms
        else:
            return None, None
        
    def get_pose_in_origin(self):
        T_camera_marker, timestamp = self.get_pose_in_camera()
        if T_camera_marker is not None:
            # Get pose in origin frame
            T_origin_marker = np.matmul(self.T_origin_camera, T_camera_marker)
            T_origin_base = np.matmul(T_origin_marker, self.T_marker_base)

            position_vec = T_origin_base[0:3, 3]
            rot_mat = T_origin_base[0:3, 0:3]
            rotation = R.from_matrix(rot_mat)
            quaternion = rotation.as_quat()

            self.__timestamp_deque.append(timestamp)
            self.__position_deque.append(position_vec)
            self.__quaternion_deque.append(quaternion)
            
            return position_vec, quaternion
        else:
            return None, None

    def get_velocities_in_origin(self):
        if len(self.__timestamp_deque) == self.__deque_size:
            time_elasped_s = (self.__timestamp_deque[-1] - self.__timestamp_deque[0])/1000.0

            # Estimate linear velocities
            position_change = self.__position_deque[-1] - self.__position_deque[0]
            linear_vel = position_change/time_elasped_s

            # Estimate angular velocities
            quat_change = self.__quaternion_deque[-1] - self.__quaternion_deque[0]
            eldest_rot = R.from_quat(self.__quaternion_deque[-1])
            eldest_rot_inv = eldest_rot.inv().as_quat()
            angular_vel = 2.0 * (quat_change/time_elasped_s) * eldest_rot_inv

            return linear_vel, angular_vel
        else:
            return np.zeros(3), np.zeros(3)

    def close(self):
        self.socket.close()

if __name__ == "__main__":
    import os, sys
    root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(root_dir)

    origin_camera_transform = np.array([[ 0.9984185,   0.04003535,  0.03946701,  0.080394  ],
                                        [ 0.04888278, -0.96497655, -0.25774157, -0.07278471],
                                        [ 0.02776596,  0.25926322, -0.9654075,   0.855662  ],
                                        [ 0.,         0.,         0.,          1.        ]])
    visp_reader = QuadrupedVispPoseReader(origin_camera_transform=origin_camera_transform,
                                          marker_len=0.08)
    visp_reader.initialize()
    while True:
        # visp_reader.get_pose_in_origin()
        # linear_vel, angular_vel = visp_reader.get_velocities_in_origin()
        # print(np.round(angular_vel, 1))
        # print(time.time()*1000.0)
        print(visp_reader.get_pose_in_camera())
        time.sleep(0.01)