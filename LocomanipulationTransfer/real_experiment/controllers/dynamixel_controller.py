import os, sys
from turtle import position
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))

import numpy as np
from controller import Controller
from dynamixel_sdk import * # Uses Dynamixel SDK library

class DynamixelBaseController(Controller):

    # Address
    ADDR_TORQUE_ON = 64
    LEN_TORQUE_ON = 1
    ADDR_GOAL_POSITION = 116
    LEN_GOAL_POSITION = 4
    ADDR_GOAL_CURRENT = 102
    LEN_GOAL_CURRENT = 2

    ADDR_PRES_POSITION = 132
    LEN_PRES_POSITION = 4
    ADDR_PRES_VELOCITY = 128
    LEN_PRES_VELOCITY = 4
    ADDR_PRES_CURRENT = 126
    LEN_PRES_CURRENT = 2

    ADDR_INFO_START = 126
    LEN_INFO = 10

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.port_name = self.cfg.get('port_name')
        self.motor_ids = self.cfg.get('motor_ids')
        self.protocol = self.cfg.get('protocol', 2.0)
        self.baudrate = self.cfg.get('baudrate', 4000000)

        # Dynamixel Classes
        self.portHandler = PortHandler(self.port_name)
        self.portHandler.setPacketTimeoutMillis(6)

        self.packetHandler = PacketHandler(self.protocol)

        self.torque_on_writer = None
        self.position_writer = None
        self.current_writer = None
        self.info_reader = None

    def initialize(self):
        # Open port
        if self.portHandler.openPort():
            self.logInfo("[DynamixelController] Successfully open serial port.")
        else:
            self.logCritical("[DynamixelController] Failed to open serial port of name {}.".format(self.port_name))
            exit(0)

        # Set port baudrate
        if self.portHandler.setBaudRate(self.baudrate):
            self.logInfo("[DynamixelController] Succeeded to change the baudrate")
        else:
            self.logCritical("[DynamixelController] Failed to change the baudrate")
            exit(0)

        # Ping the motors
        for motor_id in self.motor_ids:
            dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, motor_id)
            if dxl_comm_result != COMM_SUCCESS:
                self.logCritical("[DynamixelController] Communication failed when trying to ping motor {}; {}".format(motor_id, self.packetHandler.getTxRxResult(dxl_comm_result)))
            elif dxl_error != 0:
                self.logCritical("[DynamixelController] Failed to ping motor %d %s" % (motor_id, self.packetHandler.getRxPacketError(dxl_error)))
            else:
                self.logInfo("[DynamixelController] ID:%03d ping succeeded. Dynamixel model number : %d" % (motor_id, dxl_model_number))

        # Add motor id to param lists
        # Init sync writers
        self.torque_on_writer = GroupSyncWrite(self.portHandler,
                                             self.packetHandler,
                                             self.ADDR_TORQUE_ON,
                                             self.LEN_TORQUE_ON)

        self.position_writer = GroupSyncWrite(self.portHandler,
                                               self.packetHandler,
                                               self.ADDR_GOAL_POSITION,
                                               self.LEN_GOAL_POSITION)

        self.current_writer = GroupSyncWrite(self.portHandler,
                                             self.packetHandler,
                                             self.ADDR_GOAL_CURRENT,
                                             self.LEN_GOAL_CURRENT)

        self.info_reader = GroupSyncRead(self.portHandler,
                                         self.packetHandler,
                                         self.ADDR_INFO_START,
                                         self.LEN_INFO)

        for motor_id in self.motor_ids:
            self.info_reader.addParam(motor_id)

        # Torque on
        self.logWarning("[DynamixelController] Motor torques has been activated!")
        self.torqueOn()

    def close(self):
        # Torque off
        self.torqueOff()
        self.portHandler.closePort()
        self.logWarning("[DynamixelController] Motor torques has been inactivated. Closing serial port...")

    def torqueOn(self):
        self.torque_on_writer.clearParam()
        for motor_id in self.motor_ids:
            self.torque_on_writer.addParam(motor_id, [0x1])
        return self.torque_on_writer.txPacket()

    def torqueOff(self):
        self.torque_on_writer.clearParam()
        for motor_id in self.motor_ids:
            self.torque_on_writer.addParam(motor_id, [0x0])
        return self.torque_on_writer.txPacket()

    def setGoalCurrent(self, current_list):
        self.current_writer.clearParam()
        for i, motor_id in enumerate(self.motor_ids):
            current_value = [DXL_LOBYTE(DXL_LOWORD(current_list[i])), 
                             DXL_HIBYTE(DXL_LOWORD(current_list[i]))]
            self.current_writer.addParam(motor_id, current_value)
        return self.current_writer.txPacket()

    def setGoalPosition(self, position_list):
        self.position_writer.clearParam()
        for i, motor_id in enumerate(self.motor_ids):
            position_value = [DXL_LOBYTE(DXL_LOWORD(position_list[i])), 
                              DXL_HIBYTE(DXL_LOWORD(position_list[i])), 
                              DXL_LOBYTE(DXL_HIWORD(position_list[i])), 
                              DXL_HIBYTE(DXL_HIWORD(position_list[i]))]
            self.position_writer.addParam(motor_id, position_value)
        return self.position_writer.txPacket()

    def readInfo(self):
        '''
            Read the present position, velocity and current for 
            all the motors;

            We did not use the default api to parse the read data, because the 
            original Dynamixel SDK does not support parsing data composed of multiple
            memory blocks;
            Instead we read data_dict, which will be in the form of {motor_id: [byte1, byte2, ...], ...}

            return type:
            (position_list, velocity_list, current_list)
        '''
        dxl_comm_result = self.info_reader.txRxPacket()
        data_arrays = list(self.info_reader.data_dict.values())

        # In case of communication error, read again
        max_repeat_time = 3
        repeat_time = 0
        while dxl_comm_result != COMM_SUCCESS and repeat_time < max_repeat_time:
            #print(dxl_comm_result)
            dxl_comm_result = self.info_reader.txRxPacket()
            data_arrays = list(self.info_reader.data_dict.values())
            self.logWarning("[Dynamixel Controller] Lost packet while reading data from motors. ")
            repeat_time += 1

        data_stack = np.stack(data_arrays) # Size (num_motors, 10)

        current_list = DXL_MAKEWORD(data_stack[:, 0], data_stack[:, 1])
        velocity_list = DXL_MAKEDWORD(DXL_MAKEWORD(data_stack[:, 2], data_stack[:, 3]),
                                      DXL_MAKEWORD(data_stack[:, 4], data_stack[:, 5]))
        position_list = DXL_MAKEDWORD(DXL_MAKEWORD(data_stack[:, 6], data_stack[:, 7]),
                                      DXL_MAKEWORD(data_stack[:, 8], data_stack[:, 9]))

        # handle negative values
        offset_vel_list = (velocity_list > 0x7fffffff).astype(int) * 4294967296
        velocity_list -= offset_vel_list
        offset_cur_list = (current_list > 0x7fff).astype(int) * 65536
        current_list -= offset_cur_list

        return (position_list, velocity_list, current_list)

class QuadrupedPositionController(DynamixelBaseController):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.defualt_joint_positions = np.array(self.cfg.get("default_joint_positions"))

    def initialize(self):
        super().initialize()

        # Move joints to default positions
        self.logWarning("[Dynamixel Controller] Moving joints to default positions in 5 seconds...")
        time.sleep(5)
        self.command(self.defualt_joint_positions)
        self.logInfo("Started moving to default joint positions. Wait for 5 seconds for execution...")
        time.sleep(5)

    def retrieveInfo(self):
        '''
            Return: np.array([joint_positions, joint_velocities]) of 
            size num_motors*2; the unit is rad, rad/s respectively.
        '''
        position_list, velocity_list, _ = self.readInfo()
        # Transform the position unit to rad
        position_list = self.positionEncoder2Rad(position_list)
        # Transform the velocity unit to rad/s
        velocity_list = self.velocityEncoder2Rad(velocity_list)

        return np.append(position_list, velocity_list)

    def positionEncoder2Rad(self, position_list):
        return (-1*position_list + 2048) * 0.001534

    def velocityEncoder2Rad(self, velocity_list):
        # The positive direction of the velocity should be reversed
        return velocity_list * -0.02398

    def positionRad2Encoder(self, rad_position_list):
        return (-1*rad_position_list/0.001534 + 2048).astype(int)

    def command(self, cmd):
        '''
            cmd should be a numpy array consisting the position command
            for each motor, i.e. [motor1_pos, motor2_pos, ...], the cmd position
            should be in rad.
        '''
        encoder_position_list = self.positionRad2Encoder(cmd)
        self.setGoalPosition(encoder_position_list)

if __name__ == "__main__":
    motor_ids = [10,11,12,
                 20,21,22,
                 30,31,32,
                 40,41,42]
    cfg = {}
    cfg['port_name'] = "/dev/ttyUSB0"
    cfg['motor_ids'] = motor_ids
    dynamixel_controller = QuadrupedPositionController(cfg)
    dynamixel_controller.initialize()
    # while True:
    #     positions = dynamixel_controller.retrieveInfo()[0:12]
    #     # To degrees
    #     pos_degrees = positions * (180/np.pi)
    #     print(pos_degrees)
    #     time.sleep(0.05)
    time.sleep(2)
    for i in range(30):
        time_start = time.time()
        dynamixel_controller.readInfo()
        # init_pos = np.array([-1.57, -1.05, -2.09,
        #                     1.57,   2.09,  1.05,
        #                     1.57,   2.09,  1.05,
        #                     -1.57, -1.05, -2.09])
        # dynamixel_controller.command(init_pos)
        print(time.time()-time_start)
    # test_cmd = np.array([-90.00112719,  -77.34471868, -127.26721892,   
    #                      67.41295367,  118.56593807, 55.19600379,   
    #                      57.21751348,  116.80810355,   49.30725816,  
    #                      -75.41110071, -58.79956454 -127.61878582])
    # test_cmd = (test_cmd/180.0) * np.pi
    # print(dynamixel_controller.positionRad2Encoder(test_cmd))

    # Test sending position cmd

    # time.sleep(1)
    dynamixel_controller.close()


