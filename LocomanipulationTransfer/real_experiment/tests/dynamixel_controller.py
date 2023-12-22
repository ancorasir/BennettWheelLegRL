import numpy as np
from dynamixel_sdk import *                    # Uses Dynamixel SDK library

class DynamixelController(object):
    BAUDRATE = 4000000
    PROTOCOL = 2.0

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

    def __init__(self, port_name, motor_ids) -> None:
        self.port_name = port_name
        self.portHandler = PortHandler(self.port_name)
        self.packetHandler = PacketHandler(self.PROTOCOL)
        self.motor_ids = motor_ids

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            exit(0)

        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            exit(1)

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
        self.info_reader.txRxPacket()
        data_arrays = list(self.info_reader.data_dict.values())
        data_stack = np.stack(data_arrays) # Size (num_motors, 10)

        print(data_stack)

        current_list = DXL_MAKEWORD(data_stack[:, 0], data_stack[:, 1])
        velocity_list = DXL_MAKEDWORD(DXL_MAKEWORD(data_stack[:, 2], data_stack[:, 3]),
                                      DXL_MAKEWORD(data_stack[:, 4], data_stack[:, 5]))
        position_list = DXL_MAKEDWORD(DXL_MAKEWORD(data_stack[:, 6], data_stack[:, 7]),
                                      DXL_MAKEWORD(data_stack[:, 8], data_stack[:, 9]))

        return (position_list, velocity_list, current_list)

    def parseData(self):
        '''
            Parse a np array consisting of position, velocity and 
            current information for the motors
        '''

if __name__ == "__main__":
    import time
    port_name = '/dev/ttyUSB0'
    #motor_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]
    motor_ids = [30]

    dc = DynamixelController(port_name, motor_ids)

    time_start = time.time()
    dc.torqueOn()
    #dc.setGoalPosition([2000, 3500, 1400])
    dc.setGoalCurrent([5])
    print(dc.readInfo())
    time_end = time.time()

    print((time_end - time_start)*1000)