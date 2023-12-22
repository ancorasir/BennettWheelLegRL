import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
sys.path.append(os.path.join(root_dir, "controllers"))
from controller import Controller
from realsense_controller import RealsenseRGBController

import numpy as np
import cv2
import cv2.aruco as aruco
import time
from collections import deque

class ArucoDetector(Controller):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.base_marker_length = self.cfg.get('base_marker_length')
        self.base_marker_id = self.cfg.get('base_marker_id', 0)
        self.goal_marker_length = self.cfg.get('goal_marker_length')
        self.goal_marker_id = self.cfg.get('goal_marker_id', 0)
        self.cali_matrix = self.cfg.get('cali_matrix', None)
        self.camera_width = self.cfg.get('camera_width', 960)
        self.camera_height = self.cfg.get('camera_height', 540)
        self.camera_fps = self.cfg.get('camera_fps', 60)
        if self.cali_matrix is None:
            self.cali_matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        self.origin_xyz_offset = self.cfg.get('xyz_offset', np.array([0,0,0]))

        self.realsense = RealsenseRGBController(self.camera_width, self.camera_height, self.camera_fps)
        
        # Intrinsic infos
        self.intrinsic_matrix = None

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_params.maxMarkerPerimeterRate = 0.5
        self.aruco_params.minMarkerPerimeterRate = 0.0625
        # self.aruco_params.adaptiveThreshWinSizeMin = 3
        # self.aruco_params.adaptiveThreshWinSizeMax = 16
        # self.aruco_params.adaptiveThreshWinSizeStep = 3

    def initialize(self):
        # Initial camera
        self.realsense.initialize()

        # Getting camera intrinsics
        intrinsics = self.realsense.get_intrinsics()
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy

        self.intrinsic_matrix = np.array([[fx, 0, ppx],
                                          [0,  fy, ppy],
                                          [0,  0,  1]])
        self.dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    
        self.logInfo("[ArucoDetector] Finished reading camera intrinsic information. ")
        self.logInfo("Intrinsic matrix: ")
        self.logInfo(self.intrinsic_matrix)
        self.logInfo("[ArucoDetector] Base Marker Calibration: ")
        self.logInfo("Press s to save current configuration; Press q to use default value.")
        # Begin calibration
        key = -1
        T_bc = None
        while key != ord('q') and key != ord('s'):
            img = self.realsense.retrieveInfo()
            T_bc = self.detectMarker(img, self.base_marker_length, self.base_marker_id)
            # Draw markers 
            if T_bc is not None:
                self.drawMarker(img, T_bc)
            cv2.imshow("Base Marker", img) 
            key = cv2.waitKey(1)
        
        if key == ord('q'):
            self.logInfo("No calibration is done; default value provided will be used. ")
        elif key == ord('s'):
            if T_bc is None:
                self.logCritical("No marker is detected in this frame; exiting...")
                exit(0)
            print(T_bc)
            #T_bc[0:3, 3] = T_bc[0:3, 3] + self.origin_xyz_offset # Offset the base marker
            print(T_bc)
            self.cali_matrix = np.linalg.inv(T_bc)
            # Do calibration here
            self.logInfo("Calibration information is saved; transformation matrix: ")
            self.logInfo(self.cali_matrix)
        
        #time.sleep(5)
        cv2.destroyAllWindows()

    def retrieveInfo(self):
        img = self.realsense.retrieveInfo()
        T_cr = self.detectMarker(img, self.goal_marker_length, self.goal_marker_id)
        if T_cr is None:
            return None
        else:
            raw_T = np.matmul(self.cali_matrix, T_cr)
            raw_T[0:3, 3] -= self.origin_xyz_offset
            return raw_T

    def detectMarker(self, img, marker_length, marker_id):
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(img, 
                                              self.aruco_dict, 
                                              parameters=self.aruco_params)
        print(rejected)
        print(corners)
        if ids is not None:
            del_indice = []
            for m_index, m_id in enumerate(ids):
                if m_id[0] != marker_id:
                    # Exclude all other ids
                    del_indice.append(m_index)
            sorted_del_indices = sorted(del_indice, reverse=True)
            for del_index in sorted_del_indices:
                del corners[del_index]
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 
                                                            marker_length, 
                                                            self.intrinsic_matrix, 
                                                            self.dist) 
            # Estimate pose of each marker and return the values rvet and tvec---different 
            # from camera coeficcients  
            if rvec is not None:
                (rvec-tvec).any() # get rid of that nasty numpy value array error
                # only read the first marker in the list
                R = cv2.Rodrigues(rvec[0])[0]
                T = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,1]], dtype=np.float32)
                T[0:3, 0:3] = R
                T[0:3, 3] = tvec[0]
                return T
            else:
                return None
        else:
            return None   

    def drawMarker(self, img, T):
        R = T[0:3, 0:3]
        tvec = np.array([T[0:3, 3]])
        rvec = cv2.Rodrigues(R)[0]
        aruco.drawAxis(img, self.intrinsic_matrix, self.dist, rvec, tvec, 0.03)

    def close(self):
        self.realsense.close()

class ArucoTranslationDetector(ArucoDetector):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._time_interval_ms = self.cfg.get('time_interval_ms')
        self._pos_buf = deque(maxlen=2) # [last, second_last]
        self._vel_buf = deque(maxlen=2) # [last, second_last]
        self._lost_buf = deque(maxlen=2) # The number of intervals without detecting the markers
        self._lost_buf.appendleft(0) # initialized value 0

    def retrieveInfo(self):
        '''
            Buffer logic (| indicates the marker is detected, ! indicates lost detection):
                |------!------!------|------!------|
              buf[1]               buf[0]         tvec
                        [2 lost]           [1 lost]  
            in the above case, the self._lost_buf should be [2, 1] since 2 detections lost
            between the second_last and last detected tvec;

            Interpolation (actually prediction) logic:
            (1) Position interpolation (Linear):
                suppose t is the current time; t0 being the time for the last detection;
                t1 being the second last; tvec is the current translational vector to be interpolated;
                tvec0 is the last; tvec1 is the second last;

                (t - t0)/(t0 - t1) = (tvec - tvec0)/(tvec0 - tvec1)
                and t-t0 = (self._lost_buf[0] + 1) * interval_length; t0-t1 = (self._lost_buf[1] + 1) * interval_length
                tvec = [(t-t0)/(t0-t1)] * (tvec0 - tvec1) + tvec0
        '''
        T =  super().retrieveInfo()
        if T is not None:
            # If T is a valid matrix
            tvec = T[0:3, 3]
            if len(self._pos_buf) == 0:
                # if no previous pos for reference,
                # the velocity is set to 0
                tvel = np.array([0,0,0])
            else:
                # If there exist at least one previous pos
                # Vel = (current_pos - last_pos)/time_interval
                # Unit: m/s
                total_interval = (self._lost_buf[0] + 1) * self._time_interval_ms/1000.0
                tvel = (tvec-self._pos_buf[0])/total_interval

            # Save the current position to the buffer
            self._pos_buf.appendleft(tvec)
            # Save the current velocity to the buffer
            self._vel_buf.appendleft(tvel)
            # Update lost interval buffer
            self._lost_buf.appendleft(0)
            
            return np.append(tvec, tvel) 
        else:
            # T is None, interpolation is needed; raise the warning
            self.logWarning("[ArucoDetector] Lost detection for the current frame")
            if len(self._pos_buf) != 2:
                # If not enough pos in buffer
                tvec = np.array([0,0,0])
                tvel = np.array([0,0,0])
            else:
                # Interpolate the position
                t_t0 = (self._lost_buf[0] + 1) * self._time_interval_ms*1000.0
                t0_t1 = (self._lost_buf[1] + 1) * self._time_interval_ms*1000.0
                tvec1 = self._pos_buf[1]
                tvec0 = self._pos_buf[0]

                tvec = (t_t0/t0_t1)*(tvec0-tvec1) + tvec0
                # Estimate the velocity
                total_interval = (self._lost_buf[0] + 1) * self._time_interval_ms/1000.0
                tvel = (tvec-self._pos_buf[0])/total_interval

                # Update buffer; lost + 1
                self._lost_buf[0] += 1

            return np.append(tvec, tvel)

from pynput import keyboard

def keyboardCallback(key):
    global key_pressed
    key_pressed = True

def startKeyboardListener():
    global key_pressed
    keyboard_listener = keyboard.Listener(on_press=keyboardCallback)
    keyboard_listener.start()
    key_pressed = False

def recordData(detector, f):
    info = detector.retrieveInfo()
    info_str = ""
    for i, elem in enumerate(info):
        if i == len(info)-1:
            # not the last one
            info_str += str(elem) + '\n'
        else:
            info_str += str(elem) + ","
    f.write(info_str)


if __name__ == "__main__":
    aruco_cfg = {}
    aruco_cfg['base_marker_length'] = 0.0975
    aruco_cfg['goal_marker_length'] = 0.0975
    aruco_cfg['time_interval_ms'] = 33
    aruco_cfg['xyz_offset'] = np.array([0,0,0])
    aruco_detector = ArucoTranslationDetector(aruco_cfg)
    aruco_detector.initialize()
 
    # for i in range(10):
    #     time_start = time.time()
    #     print(aruco_detector.retrieveInfo())
    #     time_end = time.time()
    #     print((time_end - time_start)*1000)

    # Save xyz to a file
    from time_slicer import TimeSlicer
    file_path = os.path.join(root_dir, "data", '221020-test_xyz_1.csv')
    with open(file_path, 'w') as f:
        #ts = TimeSlicer(33, recordData, (aruco_detector, f))
        #ts.start()
        startKeyboardListener()
        global key_pressed
        while not key_pressed:
            recordData(aruco_detector, f)
        #ts.stop()
        time.sleep(1)
    aruco_detector.close()


    
