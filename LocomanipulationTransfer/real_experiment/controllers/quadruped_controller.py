from dynamixel_controller import DynamixelController, XM430W210
import time

class QuadrupedPositionController:
    def __init__(self, 
                 port_name,
                 baudrate,
                 motor_ids,
                 init_joint_pos,
                 ) -> None:
        self.port_name = port_name
        self.baudrate = baudrate
        self.motor_ids = motor_ids
        self.init_joint_pos = init_joint_pos

        self.motor_list = []
        for motor_id in motor_ids:
            self.motor_list.append(XM430W210(motor_id))

        self.dc = DynamixelController(self.port_name,
                                      self.motor_list,
                                      baudrate=self.baudrate,
                                      reverse_direction=True)
        
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_current = []
        self.joint_pwm = []

        # Activate the robot
        self.dc.activate_controller()
        
    def init_quadruped(self, 
                       operating_mode,
                       profile_velocity,
                       profile_acceleration,
                       goal_current,
                       position_p_gain,
                       position_d_gain):
        '''
            operating_mode: "position_control" or "current_based_position_control"

        '''
        assert len(profile_velocity) == len(self.motor_ids)
        assert len(profile_acceleration) == len(self.motor_ids)
        assert len(goal_current) == len(self.motor_ids)
        assert len(position_p_gain) == len(self.motor_ids)
        assert len(position_d_gain) == len(self.motor_ids) 

        # De-activate joint torque
        s = input("Enter to deactivate joint torque. ")
        self.dc.torque_off()
        time.sleep(1.0)

        # Set configs
        print("Setting motor configs...")
        ## Set operating mode
        self.dc.set_operating_mode_all(operating_mode)
        ## Set profile velocity
        self.dc.set_profile_velocity(profile_velocity)
        ## Set profile acceleration
        self.dc.set_profile_acceleration(profile_acceleration)
        ## Set goal current
        self.dc.set_goal_current(goal_current)
        ## Set position p gain
        self.dc.set_position_p_gain(position_p_gain)
        ## Set position d gain
        self.dc.set_position_d_gain(position_d_gain)

        time.sleep(1.0)
        # Torque on
        s = input("Enter to activate joint torque. ")
        self.dc.torque_on()
        time.sleep(1.0)

        # Move to init joint positions
        s = input("Enter to move joint to default joint positions. ")
        self.set_joint_position_targets(self.init_joint_pos)
        time.sleep(2.0)
    
    def update_joint_states(self):
        self.joint_positions, \
        self.joint_velocities, \
        self.joint_current, \
        self.joint_pwm = self.dc.read_info_with_unit(angle_unit="rad", pwm_unit="", current_unit="")

    def set_joint_position_targets(self, joint_position_targets):
        return self.dc.set_goal_position_rad(joint_position_targets)
    
if __name__ == "__main__":
    port_name = "/dev/ttyUSB0"
    baudrate = 2000000
    motor_ids = [40]
    init_joint_pos = [0.5]
    profile_vel = [125]
    profile_acc = [2]
    goal_current = [557]
    position_p_gain = [900]
    position_d_gain = [10]
    
    quadruped_controller = QuadrupedPositionController(port_name=port_name,
                                                       baudrate=baudrate,
                                                       motor_ids=motor_ids,
                                                       init_joint_pos=init_joint_pos)
    quadruped_controller.init_quadruped("position_control",
                                        profile_velocity=profile_vel,
                                        profile_acceleration=profile_acc,
                                        goal_current=goal_current,
                                        position_p_gain=position_p_gain,
                                        position_d_gain=position_d_gain)
