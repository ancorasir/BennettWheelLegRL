import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
sys.path.append("/usr/share/python3-mscl/")
import mscl
import time

import numpy as np
from controller import Controller
import ast
import torch
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665

def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def rot_matrices_to_quats(rotation_matrices: np.ndarray, device=None) -> np.ndarray:
    """Vectorized version of converting rotation matrices to quaternions

    Args:
        rotation_matrices (np.ndarray): N Rotation matrices with shape (N, 3, 3) or (3, 3)

    Returns:
        np.ndarray: quaternion representation of the rotation matrices (N, 4) or (4,) - scalar first
    """
    rot = Rotation.from_matrix(rotation_matrices)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    return result

class IMUController(Controller):
    valid_datatypes = ['euler_angle', 'rotation_matrix']

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.port_name = self.cfg.get('port_name', "/dev/ttyACM0")
        self.baudrate = self.cfg.get('baudrate', 921600)
        self.sampling_freq = self.cfg.get('sampling_freq', 60)
        self.data_type = self.cfg.get('data_type', 'euler_angle')
        self.transform_rot_mat = self.cfg.get('transform_rot_mat', None)


        assert self.data_type in self.valid_datatypes, \
            "Invalid data type {}; available data types: {}".format(self.data_type, self.valid_datatypes)

        #create the connection object with port and baud rate
        self.connection = mscl.Connection.Serial(self.port_name, self.baudrate)
        self.node = mscl.InertialNode(self.connection)
        self.ahrsImuChs = None

        # Set default transformation matrix to I(3x3)
        if self.transform_rot_mat is None:
            self.transform_rot_mat = np.array([[1,0,0], [0,1,0], [0,0,1]])

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
        print("Euler angles ([roll, pitch, yaw] in rad): ")

        # Start keyboard listener
        self.startKeyboardListener()
        # Keep printing euler angles when no keys pressed
        while not self.key_pressed:
            euler_angles = self.retrieveRawInfo("euler_angle")
            # Format the output
            euler_angle_str = ""
            for euler_angle in euler_angles:
                rounded_angle = round(euler_angle, 3)
                if rounded_angle > 0:
                    euler_angle_str += " " + str(rounded_angle) + "\t"
                else:
                    euler_angle_str += str(rounded_angle) + "\t"
            print(euler_angle_str, end="\r")

        self.node.setToIdle()
        time.sleep(0.5)
        self.ahrsImuChs = mscl.MipChannels()

        # Config data type to rotation matrix
        self.ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_ORIENTATION_MATRIX, mscl.SampleRate.Hertz(self.sampling_freq)))
        self.ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_SCALED_GYRO_VEC, mscl.SampleRate.Hertz(self.sampling_freq)))
        #self.ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_DELTA_THETA_VEC, mscl.SampleRate.Hertz(100))) # seems that only 100hz works
        self.node.setActiveChannelFields(mscl.MipTypes.CLASS_AHRS_IMU, self.ahrsImuChs)
        self.node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)

        if self.pressed_key == 'q':
            self.logWarning("IMU Calibration Finished; Nothing has changed. Using default matrix: ")
            self.logWarning(self.transform_rot_mat)
        elif self.pressed_key == 's':
            # Reset the headings
            time.sleep(0.5)
            R_or = self.retrieveRawInfo("rotation_matrix")
            self.transform_rot_mat = np.transpose(R_or)
            self.logInfo("IMU Calibration Finished; Transformation matrix: {}. ".format(self.transform_rot_mat))
        else:
            self.logWarning("Unrecognized command; Nothing has changed")

    
    def retrieveInfo(self):
        raw_rot_vel = np.zeros(3)
        raw_rot_mat = np.zeros((3,3))
        #raw_vel = np.zeros(3)

        packets = self.node.getDataPackets(50, 0)
        last_packet = packets[-1]
        points = last_packet.data()
        for point in points:
            if point.channelName() == "scaledGyroX":
                raw_rot_vel[0] = point.as_float()
            elif point.channelName() == "scaledGyroY":
                raw_rot_vel[1] = point.as_float()
            elif point.channelName() == "scaledGyroZ":
                raw_rot_vel[2] = point.as_float()
            elif point.channelName() == "orientMatrix":
                raw_rot_mat = np.transpose(np.array(ast.literal_eval(point.as_string())))
            # elif point.channelName() == "deltaThetaX":
            #     raw_vel[0] = point.as_float() * GRAVITY
            # elif point.channelName() == "deltaThetaY":
            #     raw_vel[1] = point.as_float() * GRAVITY
            # elif point.channelName() == "deltaThetaZ":
            #     raw_vel[2] = point.as_float() * GRAVITY             
        reference_rot_mat = np.matmul(self.transform_rot_mat, raw_rot_mat)
        reference_rot_vel = np.matmul(self.transform_rot_mat, raw_rot_vel)

        # transform the rotation matrix to the custom defined reference frame
        if self.data_type == "euler_angle":
            # Transform from rotation matrix to euler angles
            return np.append(self.rotationMatrixToEuler(reference_rot_mat), reference_rot_vel)
        elif self.data_type == "rotation_matrix":
            return reference_rot_mat

    def retrieveRawInfo(self, datatype):
        packets = self.node.getDataPackets(50, 0)
        last_packet = packets[-1]
        points = last_packet.data()
        if datatype == "euler_angle":
            return np.array([points[0].as_float(), points[1].as_float(), points[2].as_float()])
        elif datatype == "rotation_matrix":
            return np.transpose(np.array(ast.literal_eval(points[0].as_string())))

    def close(self):
        self.node.setToIdle()
        time.sleep(0.5)
        self.node.setToIdle()
        self.connection.disconnect()

    def rotationMatrixToEuler(self, R):
        '''
            Code copied from 
            https://learnopencv.com/rotation-matrix-to-euler-angles/
        '''
        import math
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

class CalibratedIMUController(IMUController):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.data_type = self.cfg.get('data_type', 'rotation_matrix')
        self.init_rot_transform = self.cfg.get('init_rot_transform', None)
        if self.init_rot_transform is None:
            self.init_rot_transform = np.array([[0, -1, 0],
                                                [-1, 0, 0],
                                                [0,  0, -1]])

    # def initialize(self):
    #     super().initialize()
    #     # transform the rotational transformation matrix to the desired initial configuration
    #     self.transform_rot_mat = np.matmul(self.init_rot_transform, self.transform_rot_mat)


    def getCalibratedInfo(self):
        raw_rot_vel = np.zeros(3)
        raw_rot_mat = np.zeros((3,3))
        #raw_vel = np.zeros(3)

        packets = self.node.getDataPackets(50, 0)
        last_packet = packets[-1]
        points = last_packet.data()
        for point in points:
            if point.channelName() == "scaledGyroX":
                raw_rot_vel[0] = point.as_float()
            elif point.channelName() == "scaledGyroY":
                raw_rot_vel[1] = point.as_float()
            elif point.channelName() == "scaledGyroZ":
                raw_rot_vel[2] = point.as_float()
            elif point.channelName() == "orientMatrix":
                raw_rot_mat = np.transpose(np.array(ast.literal_eval(point.as_string())))
            # elif point.channelName() == "deltaThetaX":
            #     raw_vel[0] = point.as_float() * GRAVITY
            # elif point.channelName() == "deltaThetaY":
            #     raw_vel[1] = point.as_float() * GRAVITY
            # elif point.channelName() == "deltaThetaZ":
            #     raw_vel[2] = point.as_float() * GRAVITY            
        reference_rot_mat = np.matmul(self.transform_rot_mat, raw_rot_mat)
        global_rot_mat = self.transformToGlobal(reference_rot_mat)
        reference_rot_vel = np.matmul(reference_rot_mat, raw_rot_vel)
        global_rot_vel = np.matmul(self.init_rot_transform, reference_rot_vel)

        return global_rot_mat, global_rot_vel

    def retrieveInfo(self):
        '''
            Return:
                An array of dimension (6,1), which consists projected gravity vector
                and base angular velocities relative to fixed global frame defined
                from calibration and init_rot_transform;
        '''
        global_rot_mat, global_rot_vel = self.getCalibratedInfo()

        # Calculate projected gravity vector from rotation matrix
        g_b = np.array([0,0,-1.0]) # Gravity in fixed global frame {b}
        g_i = np.matmul(np.transpose(global_rot_mat), g_b) # g_i = R_ib * g_b = R_bi^T * g_b

        # quaternion = rot_matrices_to_quats(global_rot_mat)
        # gravity_vector = torch.tensor([0, 0, -1], dtype=torch.double, device='cpu')
        # quaternion = torch.from_numpy(np.array([quaternion])).to('cpu')
        # g_i = quat_rotate(quaternion, gravity_vector).numpy()

        return np.append(g_i, global_rot_vel)


    def  transformToGlobal(self, reference_rot_mat):
        '''
            R_b'i' = R_b'b * R_bi * R_ii'

            where:
            b is the fixed base-frame defined by the calibration result; 
            (zero rotation reference frame) ;
            i is the rotation relative to b;
            b' is the global reference frame;
            i' is the rotation relative to b';

            and R_b'b, R_ii' are just fixed transformations, with the following relationship:
            R_b'b = transpose(R_ii'), and R_b'b is self.init_rot_transform.
        '''
        interm_global_rot_mat = np.matmul(self.init_rot_transform, reference_rot_mat)
        global_rot = np.matmul(interm_global_rot_mat, np.transpose(self.init_rot_transform))
        return global_rot

# Test IMU
if __name__ == "__main__":
    imu_cfg = {}
    imu_cfg['port_name'] = "/dev/ttyACM0"
    imu_cfg['baudrate'] = 921600
    imu_cfg['sampling_freq'] = 500
    imu = QuadrupedIMUController(imu_cfg)
    imu.initialize()
    # for i in range(10):
    #     time_start = time.time()
    #     print(imu.retrieveInfo())
    #     time_end = time.time()
    #     print("Time elasped : ", (time_end - time_start)*1000)
    while True:
        info = imu.retrieveInfo()
        gyro_str = ""
        for gyro in info[0:6]:
            rounded_gyro = round(gyro, 4)
            if gyro < 0:
                gyro_str += " " + str(rounded_gyro)
            else:
                gyro_str += "  " + str(rounded_gyro)
        #print(gyro_str, end='\r')
        print(gyro_str)
        time.sleep(0.05)
    # while True:
    #     rot_matrix, angular_vel = imu.getCalibratedInfo()
    #     print(np.round(rot_matrix,1))
    #     time.sleep(0.05)
    imu.close()
