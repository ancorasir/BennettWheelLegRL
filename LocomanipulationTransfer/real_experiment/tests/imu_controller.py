import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
sys.path.append("/usr/share/python3-mscl/")
import mscl
from copy import deepcopy
import keyboard
import time

import numpy as np
from controller import Controller

class IMUController(Controller):
    valid_datatypes = ['euler_angle', 'quaternion', 'rotation_matrix']

    def __init__(self, port_name, baudrate, data_type='euler_angle', sampling_freq=60) -> None:
        super().__init__()

        self.port_name = port_name
        self.baudrate = baudrate
        self.sampling_freq = sampling_freq

        assert data_type in self.valid_datatypes, \
            "Invalid data type {}; available data types: {}".format(data_type, self.valid_datatypes)
        self.data_type = data_type

        #create the connection object with port and baud rate
        self.connection = mscl.Connection.Serial(self.port_name, self.baudrate)
        self.node = mscl.InertialNode(self.connection)
        self.ahrsImuChs = None

        self.euler_yaw_offset = 0
        self.matrix_yaw_offset = 1

    def initialize(self):
        ping_success = self.node.ping()
        if ping_success:
            self.logInfo("[IMUController] IMU ping success! ")
        else:
            self.logCritical("[IMUController] Failed to ping IMU. ")
            exit(0)
        self.node.setToIdle()
        time.sleep(0.5)

        # Orientation calibration
        _cali_chs = mscl.MipChannels()
        _cali_chs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_EULER_ANGLES, mscl.SampleRate.Hertz(self.sampling_freq)))
        self.node.setActiveChannelFields(mscl.MipTypes.CLASS_AHRS_IMU, _cali_chs)
        self.node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)

        self.logInfo("[IMUController] IMU initial orientation calibration started;")
        self.logInfo("Press s to set current orientation as the initial configuration; press q if you do not want to make any changes.")
        self.logInfo("Be sure to keep IMU still when pressing s.")
        # Set data type to euler angle for clearer inteqrpretation
        _origin_datatype = deepcopy(self.data_type)
        self.data_type = "euler_angle"
        print("Euler angles ([roll, pitch, yaw] in rad): ")

        # Start keyboard listener
        self.startKeyboardListener()
        # Keep printing euler angles when no keys pressed
        origin_yaw = 0
        while not self.key_pressed:
            euler_angles = self.retrieveInfo()
            origin_yaw = euler_angles[2] # Update yaw to the lastest
            # Format the output
            euler_angle_str = ""
            for euler_angle in euler_angles:
                rounded_angle = round(euler_angle, 2)
                if rounded_angle > 0:
                    euler_angle_str += " " + str(rounded_angle) + "\t"
                else:
                    euler_angle_str += str(rounded_angle) + "\t"
            print(euler_angle_str, end="\r")

        if self.pressed_key == 'q':
            self.logWarning("IMU Calibration Finished; Nothing has changed.")
        elif self.pressed_key == 's':
            # Reset the headings
            
            self.logInfo("IMU Calibration Finished; New yaw offset : {}. ".format(self.yaw_offset))
        else:
            self.logWarning("Unrecognized command; Nothing has changed")

        self.data_type = _origin_datatype
        self.node.setToIdle()
        time.sleep(0.5)
        self.ahrsImuChs = mscl.MipChannels()

        # Config data type
        if self.data_type == "quaternion":
            self.ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_ORIENTATION_QUATERNION, mscl.SampleRate.Hertz(self.sampling_freq)))
        elif self.data_type == "euler_angle":
            self.ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_EULER_ANGLES, mscl.SampleRate.Hertz(self.sampling_freq)))
        elif self.data_type == "rotation_matrix":
            self.ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_ORIENTATION_MATRIX, mscl.SampleRate.Hertz(self.sampling_freq)))
    
        self.node.setActiveChannelFields(mscl.MipTypes.CLASS_AHRS_IMU, self.ahrsImuChs)
        self.node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)
        #self.node.resume()

    
    def retrieveInfo(self):
        packets = self.node.getDataPackets(50, 1)
        last_packet = packets[-1]
        points = last_packet.data()

        if self.data_type == "quaternion":
            for point in points:
                return np.array(point.as_Vector())
        elif self.data_type == "euler_angle":
            return np.array([points[0].as_float(), points[1].as_float(), points[2].as_float()+self.euler_yaw_offset])
        elif self.data_type == "rotation_matrix":
            return np.array(points[0].as_Matrix())

    def close(self):
        self.node.setToIdle()
        time.sleep(0.5)
        self.node.setToIdle()
        self.connection.disconnect()

# Test IMU
if __name__ == "__main__":
    imu = IMUController("/dev/ttyACM0", 921600, 'rotation_matrix', 100)
    imu.initialize()
    for i in range(10):
        time_start = time.time_ns()
        imu.retrieveInfo()
        time_end = time.time_ns()
        print("Time elasped : ", (time_end - time_start)/10e6)
    imu.close()
