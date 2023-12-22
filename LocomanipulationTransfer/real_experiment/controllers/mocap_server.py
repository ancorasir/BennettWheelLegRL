from nokov.nokovsdk import *
import time
import sys, getopt
import threading

from copy import deepcopy

class MocapServer:
    def __init__(self, 
                 server_ip='10.1.1.198') -> None:
        self.server_ip = server_ip
        
        self.position = []
        self.quaternion = []
        self.linear_vel = []
        self.angular_vel = []

        self.mocap_server_thread = threading.Thread(target=self.mocap_process, args=())

    def run_mocap(self):
        self.mocap_process.start()

    def py_data_func(self, pFrameOfMocapData, pUserData):
        if pFrameOfMocapData == None:  
            state_info = [0] * 13
        else:
            frameData = pFrameOfMocapData.contents

            # Get timestamp
            timestamp = frameData.iTimeStamp
            # Get number of markersets
            num_marker_sets = frameData.nMarkerSets
            # Get the number of rigid bodies
            num_rigid_bodies = frameData.nRigidBodies
            
            if num_marker_sets != 1 or num_rigid_bodies != 1:
                # there should only be a single markerset/rigidbody
                state_info = [0] * 13
            else:
                # There is valid data

                # Get the name of marker set
                marker_set_name = markerset = frameData.MocapData[0].szName

                if marker_set_name != "quadruped_pose_marker":
                    state_info = [0] * 13
                else:
                    # It is a valid quadruped pose marker

                    # Get rigid body position and quaternions
                    marker_rb = frameData.RigidBodies[0]

                    x = marker_rb.x
                    y = marker_rb.y
                    z = marker_rb.z
                    qw = marker_rb.qw
                    qx = marker_rb.qx
                    qy = marker_rb.qy
                    qz = marker_rb.qz

                    # Check if the pose data is valid
                    if z > 10000:
                        state_info = [0] * 13
                    else:
                        state_info[0:7] = [x, y, z, qw, qx, qy, qz]
                        self.position = [x, y, z]

    def mocap_process(self):
        mocap_client = PySDKClient()
        ver = mocap_client.PySeekerVersion()
        print('SeekerSDK Sample Client 2.4.0.3142(SeekerSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

        mocap_client.PySetVerbosityLevel(0)
        mocap_client.PySetMessageCallback(py_msg_func)
        mocap_client.PySetDataCallback(py_data_func, None)

        print("Begin to init the SDK Client")

        ret = mocap_client.Initialize(bytes(self.server_ip, encoding = "utf8"))
        if ret == 0:
            print("Connect to the Seeker Succeed")
        else:
            print("Connect Failed: [%d]" % ret)
            exit(0)

        while True:
            pass
