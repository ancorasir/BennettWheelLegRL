import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
sys.path.append(os.path.join(root_dir, "controllers"))
from controller import Controller
from realsense_controller import RealsenseRGBController

import numpy as np
import cv2.aruco as aruco
import time
from collections import deque
import socket
import ast
import cv2
import subprocess

SERVER_CMD = "/home/bionicdl/SHR/packages/visp-ws/blob/build/tutorial-blob-tracker-live-realsense"
#SERVER_SH_PATH = os.path.join(os.path.dirname(__file__), "run_blob_tracking_server.sh")

class BlobTrackerClient(Controller):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        self.marker_len = self.cfg.get('marker_length')
        self.time_interval_ms = self.cfg.get('time_interval_ms')
        self.ip_addr = self.cfg.get('ip_address')
        self.port = self.cfg.get('port')
        self.cali_matrix = self.cfg.get('cali_matrix', None)
        if self.cali_matrix is None:
            self.cali_matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        self.origin_xyz_offset = self.cfg.get('xyz_offset', np.array([0,0,0]))
        # Camera infos
        self.camera_width = self.cfg.get('camera_width', 960)
        self.camera_height = self.cfg.get('camera_height', 540)
        self.camera_fps = self.cfg.get('camera_fps', 60)
        self.fake_realsense = RealsenseRGBController(self.camera_width, self.camera_height, self.camera_fps)
        
        self._camera_intrinsics = None
        self._intrinsic_matrix = None
        self._camera_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.settimeout(0.05)
        self.server_process = None

    def initialize(self):
        # Read camera intrinsic
        self.fake_realsense.initialize()
        self._camera_intrinsics = self.fake_realsense.get_intrinsics()
        self.fake_realsense.close()
        fx = self._camera_intrinsics.fx
        fy = self._camera_intrinsics.fy
        ppx = self._camera_intrinsics.ppx
        ppy = self._camera_intrinsics.ppy

        self._intrinsic_matrix = np.array([[fx, 0, ppx],
                                          [0,  fy, ppy],
                                          [0,  0,  1]])

        # Start the server
        self.logInfo("[Tracker Client] Starting blob tracking server...")
        self.server_process = subprocess.Popen(SERVER_CMD)

        # Waiting for the server to start
        start = input("Start the server, and input s after initializing the server1...\n")
        if start[-1] != 's':
            print("exiting...")
            exit(0)
        self.logInfo("Connecting to server...")
        try:
            self._client.connect((self.ip_addr, self.port))
            time.sleep(5)
        except:
            self.logCritical("Failed to connect to the server...")
            exit(0)
        
        self.logInfo("[Tracker Client] Start calibration. ")
        self.logInfo("[Tracker Client] Press s to save current configuration; press q to use default value; press any other key to exit program. ")
        self.logInfo("Current Position(x,y,z in meters): ")
        T_bc = None
        self.startKeyboardListener()
        while not self.key_pressed:
            corners = self.getCorners()
            T_bc = self.getPose(corners)
            tvec = T_bc[0:3, 3]
            tvec_str = ""
            for t in tvec:
                rounded_t = round(t, 3)
                if rounded_t > 0:
                    tvec_str += " " + str(rounded_t) + "\t"
                else:
                    tvec_str += str(rounded_t) + "\t"
            print(tvec_str, end='\r')
        
        if self.pressed_key == 's':
            if T_bc is None:
                self.logCritical("No markers detected; exiting...")
                exit(0)
            self.cali_matrix = np.linalg.inv(T_bc)
            self.logInfo("Calibration information is saved; transformation matrix: ")
            self.logInfo(self.cali_matrix)
        elif self.pressed_key == 'q':
            self.logInfo("No calibration is done; default value provided will be used. ")
        else:
            self.logWarning("Exiting...")
            exit(0)

    def getCorners(self):
        # try:
        #     self._client.sendall(b"INFO")
        #     data = self._client.recv(1024).decode()
        #     data_array = ast.literal_eval(data)
        # except:
        #     self.logWarning("Failed to fetch corners. ")
        #     return None

        self._client.sendall(b"INFO")
        data = self._client.recv(1024).decode()
        data_array = ast.literal_eval(data)

        return np.array([data_array])

    def getPose(self, corners):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 
                                                        self.marker_len, 
                                                        self._intrinsic_matrix, 
                                                        self._camera_dist) 
        if rvec is not None:
            # Tracking not failed
            (rvec-tvec).any() # get rid of that nasty numpy value array error
            R = cv2.Rodrigues(rvec[0])[0]
            T = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,1]], dtype=np.float32)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec[0]
            return T
        else:
            return None

    def retrieveInfo(self):
        corners = self.getCorners()
        T_cr = self.getPose(corners)
        if T_cr is None:
            return None
        else:
            raw_T = np.matmul(self.cali_matrix, T_cr)
            raw_T[0:3, 3] += self.origin_xyz_offset
            return raw_T

    def close(self):
        self.server_process.kill()

class BlobTranslationTracker(BlobTrackerClient):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._vel_cal_interval = self.cfg.get("velocity_calculation_interval", 3)
        self._pos_buf = deque(maxlen=self._vel_cal_interval) # [most_recent,...,least_recent]

    def retrieveInfo(self):
        raw_T = super().retrieveInfo()
        if raw_T is None:
            return None

        # Get position
        pos = raw_T[0:3, 3]

        # Estimate velocity from position buffer
        if len(self._pos_buf) == self._vel_cal_interval:
            least_recent_pos = self._pos_buf[-1]
            vel = (pos-least_recent_pos)/(self.time_interval_ms*self._vel_cal_interval/1000.0) # Pos_diff/total_time
        else:
            # Not enough history to calculate velocity, output [0,0,0]
            vel = np.array([0,0,0])
        
        # Update the buffer
        self._pos_buf.appendleft(pos)

        return np.append(pos, vel)

if __name__ == "__main__":
    tracker_cfg = {}
    tracker_cfg['marker_length'] = 0.0825
    tracker_cfg['time_interval_ms'] = 33
    tracker_cfg['xyz_offset'] = np.array([0,0,0.227])
    tracker_cfg['ip_address'] = '127.0.0.1'
    tracker_cfg['port'] = 8081
    tracker_cfg['velocity_calculation_interval'] = 5
    tracker = BlobTranslationTracker(tracker_cfg)
    tracker.initialize()
    #for i in range(100):
    while True:
        time_start = time.time()
        info = tracker.retrieveInfo()
        if info is None:
            break
        print(info[0:3])
        print(time.time()-time_start)
        time.sleep(0.1)
    tracker.close()
    
