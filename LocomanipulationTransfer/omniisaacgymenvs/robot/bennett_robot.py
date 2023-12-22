from typing import Optional
import numpy as np
import torch
from utils.path import *
from utils.math import transform_vectors, inverse_transform_vectors, rand_quaternions, inverse_rotate_orientations, rotate_orientations

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.torch.maths import unscale_transform
from omni.isaac.core.utils.prims import get_prim_at_path

from pxr import PhysxSchema

available_module_configs = ["horizontal", "vertical"]
num_modules = 4
module_prefix_list = ['FL_leg', 'FR_leg', 'RL_leg', 'RR_leg'] #changed
module_prefix_list1 = ['FL','FR','RL','RR']
# float_horizontal_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_v1_symmetric", "overconstrained_ov_v1_symmetric_test.usd")
float_horizontal_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "quadruped_isaacsim_with_foot","Quadruped-IsaacSim-with-foot","Quadruped-IsaacSim-with-foot.usd")
# float_horizontal_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "planar_symmetric_horizontal", "planar_symmetric_horizontal_test.usd")
# float_horizontal_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_v1_symmetric", "overconstrained_ov_v1_symmetric_fixed.usd")
float_horizontal_robot_usd_path_simple_collision = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_v1_symmetric_tip_collision_only", "overconstrained_ov_v1_symmetric_test.usd")
fixed_horizontal_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_v1_symmetric", "overconstrained_ov_v1_symmetric_fixed.usd")
fixed_horizontal_robot_usd_path_simple_collision = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_v1_symmetric_tip_collision_only", "overconstrained_ov_v1_symmetric_fixed.usd")
float_vertical_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_vertical_symmetric_cog_center", "overconstrained_ov_vertical_symmetric_cog_center_test.usd")
fixed_vertical_robot_usd_path = os.path.join(root_ws_dir, "design", "RobotUSD", "overconstrained_ov_vertical_symmetric_cog_center", "overconstrained_ov_vertical_symmetric_cog_center_fixed.usd")

class OmniverseRobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Module",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            position=translation,
            orientation=orientation,
            articulation_controller=None,
        )

    def set_rigid_body_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)
    
    #TODO: not exist yet
    def prepare_tip_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "fingertip_frame1" in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)

    def prepare_base_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "Body1" in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)

    def prepare_link_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "link" in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)
    
    #TODO: define wheel contacts
    def prepare_wheel_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "wheel" in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)

class BennettRobot:
    def __init__(self, 
                 default_joint_positions,
                 actual_control_decimal=1,
                 control_kp=4.5,
                 control_kd=0.2,
                 action_scale=0.15,
                 max_torque=1.5,
                 joint_friction=0.007,
                 joint_damping=0.008,
                 default_base_position=[0.0, 0.0, 0.0],
                 default_base_quaternion=[1.0, 0.0, 0.0, 0.0],
                 fixed=False,
                 module_config="horizontal",
                 enable_tip_contact_sensor=False,
                 enable_contact_termination=False
                 ) -> None:
        '''
            args:
                fixed: True for fixed base, False for floating base
        '''
        self.default_joint_positions = default_joint_positions
        self.control_kp = control_kp
        self.control_kd = control_kd
        self.action_scale = action_scale
        self.max_torque = max_torque
        self.joint_friction = joint_friction
        self.joint_damping = joint_damping
        self.actual_control_decimal = actual_control_decimal

        self.fixed = fixed
        self.module_config = module_config
        assert self.module_config in available_module_configs
        self.default_base_position = default_base_position
        self.default_base_quaternion = default_base_quaternion
        self.robot_env_ids = None
        self.enable_tip_contact_sensor = enable_tip_contact_sensor
        self.enable_contact_termination = enable_contact_termination

        # if self.fixed:
        #     # For fixed base robot
        #     if self.module_config == "horizontal":
        #         # Fixed base horizontal configuration
        #         if not self.enable_contact_termination:
        #             self.usd_path = fixed_horizontal_robot_usd_path_simple_collision
        #         else:
        #             self.usd_path = fixed_horizontal_robot_usd_path
        #         self.robot_name = "robot_horizontal_fixed"
        #     elif self.module_config == "vertical":
        #         self.usd_path = fixed_vertical_robot_usd_path
        #         self.robot_name = "robot_vertical_fixed"
        # else:
        #     if self.module_config == "horizontal":
        #         if not self.enable_contact_termination:
        #             self.usd_path = float_horizontal_robot_usd_path_simple_collision
        #         else:
        #             self.usd_path = float_horizontal_robot_usd_path
        #         self.robot_name = "robot_horizontal"
        #     elif self.module_config == "vertical":
        #         self.usd_path = float_vertical_robot_usd_path
        #         self.robot_name = "robot_vertical"

        ## we only consider float horizontal model in bennett case
        self.usd_path = float_horizontal_robot_usd_path
        

        #TODO: set initial states here
        if self.module_config == "horizontal":
            # For horizontal configuration, swing extension control
            self.min_joint_pos_swing_ext = [-2.35, -0.78, -0.78, -2.35,
                                        -2.09, 0.52, # Swing, extension
                                        1.05, 0.52,
                                        1.05, 0.52,
                                        -2.09, 0.52]
            self.max_joint_pos_swing_ext = [0.78,  2.35,  2.35,  0.78,
                                        -1.05,   2.09, # Swing, extension
                                        2.09, 2.09,
                                        2.09, 2.09,
                                        -1.05, 2.09]
            self.default_joint_position_targets_se = [-1.57, 1.57, 1.57, -1.57,
                                                    -1.57, 1.05,
                                                    1.57, 1.05,
                                                    1.57, 1.05,
                                                    -1.57, 1.05]
            
            # For horizontal configuration, normal joint control
            self.min_joint_pos = [-2.35, -0.78, -0.78, -2.35,
                                -1.83, -3.14,
                                1.31, 0.00,
                                1.31, 0.00,
                                -1.83, -3.14]
            
            self.max_joint_pos = [0.78,  2.35,  2.35,  0.78,
                                0.00, -1.31,
                                3.14, 1.83,
                                3.14, 1.83,
                                0.00, -1.31,]
            
            self.default_joint_position_targets = self.default_joint_positions[0:12]

            self.reset_min_joint_23_diff = 0.384 # 22 degree
            self.reset_max_joint_23_diff = 2.61 # 150 degree
            self.reset_min_joint_1_pos = -2.44 # For a1, 140 degree
            self.reset_max_joint_1_pos = 0.87 # For a1, 50 degree

            self.__corner_pos_robot = None
            
        elif self.module_config == "vertical":
            # For horizontal configuration
            self.min_joint_pos_swing_ext = [-1.57, -1.57, -1.57, -1.57,
                                        -0.785, 0.52, # Swing, extension
                                        -0.785, 0.52,
                                        -0.785, 0.52,
                                        -0.785, 0.52]
            self.max_joint_pos_swing_ext = [1.57,  1.57,  1.57,  1.57,
                                        0.785,   2.09, # Swing, extension
                                        0.785, 2.09,
                                        0.785, 2.09,
                                        0.785, 2.09]
            self.default_joint_position_targets_se = [0.0, 0.0, 0.0, 0.0,
                                                    0.35, -0.35, 
                                                    0.35, -0.35,
                                                    0.35, -0.35,
                                                    0.35, -0.35,]

            # For vertical configuration, normal joint control
            self.min_joint_pos = [-1.57, -1.57, -1.57, -1.57,
                                -1.33, -1.33,
                                -1.33, -1.33,
                                -1.33, -1.33,
                                -1.33, -1.33]
            
            self.max_joint_pos = [1.57,  1.57,  1.57,  1.57,
                                1.33, 1.33,
                                1.33, 1.33,
                                1.33, 1.33,
                                1.33, 1.33]
            
            self.default_joint_position_targets = self.default_joint_positions[0:12]

            self.reset_min_joint_23_diff = 0.384 # 22 degree
            self.reset_max_joint_23_diff = 2.61 # 150 degree
            self.reset_min_joint_1_pos = -1.75 # -2.26 # For a1, 100 degree
            self.reset_max_joint_1_pos = 1.75 # 2.26 # For a1, 100 degree

            self.__corner_pos_robot = None

        self._omniverse_robot = None
        self._robot_articulation = None
        # Some auxilary prim views
        self._base_view = None
        self._knee_view = None
        self._tip_view = None
        self._link_view = None

        self._omni_dof_names = []
        self._omni_dof_indices = []

        self._local_origin_positions = None

        self.control_kp_tensor = None 
        self.control_kd_tensor = None
        self.action_scale_tensor = None
        self.max_torque_tensor = None

        self.last_base_in_contact = None
        self.last_link_in_contact = None
        self.robot_name = "Quadruped_IsaacSim_with_foot"


        
    def init_omniverse_robot(self, default_zero_env_path="/World/envs/env_0", sim_config=None): #TODO:change the default path when referencing to this method
        """
            Initialize the Omniverse Robot instance; Add the first Omniverse Robot to the stage
            for cloning;
        """
        stage = get_current_stage()

        prim_path = default_zero_env_path + "/" + self.robot_name
        # print("/World/bennett_new/Quadruped_IsaacSim/Body1")

        
        self._omniverse_robot = OmniverseRobot(prim_path,
                                               self.robot_name,
                                               self.usd_path,
                                               self.default_base_position,
                                               self.default_base_quaternion)
        
        if sim_config is not None:
            sim_config.apply_articulation_settings(self.robot_name, get_prim_at_path(self._omniverse_robot.prim_path), sim_config.parse_actor_config(self.robot_name))
        
        self._omniverse_robot.set_rigid_body_properties(stage=stage,
                                                        prim=self._omniverse_robot.prim)
        
        self._omniverse_robot.prepare_tip_contacts(stage=stage,
                                                   prim=self._omniverse_robot.prim)
        self._omniverse_robot.prepare_base_contacts(stage=stage,
                                                   prim=self._omniverse_robot.prim)
        self._omniverse_robot.prepare_link_contacts(stage=stage,
                                                   prim=self._omniverse_robot.prim)
        self._omniverse_robot.prepare_wheel_contacts(stage=stage,
                                                     prim=self._omniverse_robot.prim)
    def init_robot_views(self, scene):
        self.init_robot_articulation(scene)
        self.init_base_view(scene)
        # self.init_knee_view(scene)
        self.init_tip_view(scene)
        self.init_link_view(scene)

    #TODO: set articulation here
    def init_robot_articulation(self, scene):
        """
            Create the robot articulation; Add the articulation to the scene;
        """
        arti_root_name = "/World/envs/.*/" + self.robot_name  + "/Body1" #changed
        self._robot_articulation = ArticulationView(prim_paths_expr=arti_root_name, name=self.robot_name)
        scene.add(self._robot_articulation)

    #TODO: tell the program what is base, knee, link, tip, wheel
    def init_base_view(self, scene):
        self._base_view = RigidPrimView("/World/envs/.*/" + self.robot_name + "/Body1", #changed
                                        self.robot_name + "_base_view", 
                                        reset_xform_properties=False,
                                        track_contact_forces=self.enable_contact_termination)
        
        # self._base_view = RigidPrimView("/World/envs/.*/" + self.robot_name + "/base", 
        #                                 self.robot_name + "_base_view", 
        #                                 reset_xform_properties=False,
        #                                 track_contact_forces=False)
        
        scene.add(self._base_view)

    #TODO: the structure is different from our bennett_new model, it may should be changed
    def init_knee_view(self, scene):
        prefix_names = ""
        for i, module_prefix in enumerate(module_prefix_list):
            if i == 0:
                prefix_names += module_prefix
            else:
                prefix_names += "|" + module_prefix
        knee_prim_path_expr = "/World/envs/.*/" + self.robot_name + "/({})_link[21,31]".format(prefix_names) #changed
        
        self._knee_view = RigidPrimView(knee_prim_path_expr, 
                                        self.robot_name + "_knee_view", 
                                        reset_xform_properties=False,
                                        track_contact_forces=False)

        scene.add(self._knee_view)

    def init_link_view(self, scene):
        link_prim_path_expr = "/World/envs/.*/" + self.robot_name + "/.*_link.*"
        
        self._link_view = RigidPrimView(link_prim_path_expr, 
                                        self.robot_name + "_link_view", 
                                        reset_xform_properties=False,
                                        track_contact_forces=self.enable_contact_termination)
        
        # self._link_view = RigidPrimView(link_prim_path_expr, 
        #                                 self.robot_name + "_link_view", 
        #                                 reset_xform_properties=False,
        #                                 track_contact_forces=False)

        scene.add(self._link_view)

    def init_tip_view(self, scene):
        prefix_names = ""
        for i, module_prefix in enumerate(module_prefix_list1):
            if i == 0:
                prefix_names += module_prefix
            else:
                prefix_names += "|" + module_prefix

        tip_prim_path_expr = "/World/envs/.*/" + self.robot_name + "/({})_fingertip_frame1".format(prefix_names) #TODO: change act tipname 
        
        self._tip_view = RigidPrimView(tip_prim_path_expr, 
                                       self.robot_name + "_tip_view", 
                                       reset_xform_properties=False, 
                                       track_contact_forces=self.enable_tip_contact_sensor)
        
        # self._tip_view = RigidPrimView(tip_prim_path_expr, 
        #                                self.robot_name + "_tip_view", 
        #                                reset_xform_properties=False, 
        #                                track_contact_forces=False)
        
        scene.add(self._tip_view)

    def post_reset_robot(self, env_pos, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = device

        # for prefix in module_prefix_list:
        #     for i in range(1, 4):
        #         self._omni_dof_names.append("{}_dof{}".format(prefix, i))
        self._omni_dof_names.append("FL_Single_Motor1_FL_Single_Double_Motor")
        self._omni_dof_names.append("FR_Single_Motor1_FR_Single_Double_Motor")
        self._omni_dof_names.append("RL_Single_Motor1_RL_Single_Double_Motor")
        self._omni_dof_names.append("RR_Single_Motor1_RR_Single_Double_Motor")
        self._omni_dof_names.append("FL_Double_Motor1_FL_Double_Motor_Link1")
        self._omni_dof_names.append("FR_Double_Motor1_FR_Double_Motor_Link1")
        self._omni_dof_names.append("RL_Double_Motor1_RL_Double_Motor_Link1")
        self._omni_dof_names.append("RR_Double_Motor1_RR_Double_Motor_Link1")
        self._omni_dof_names.append("FL_Double_Motor1_FL_Double_Motor_Link4")       
        self._omni_dof_names.append("FR_Double_Motor1_FR_Double_Motor_Link4")
        self._omni_dof_names.append("RL_Double_Motor1_RL_Double_Motor_Link4")
        self._omni_dof_names.append("RR_Double_Motor1_RR_Double_Motor_Link4")
        # Read the dof indices
        if self._omni_dof_names:
            # If dof_names has been defined
            for dof_name in self._omni_dof_names:
                self._omni_dof_indices.append(self._robot_articulation.get_dof_index(dof_name))

            # Re-arrange dof_name by dof_indices
            self._omni_dof_names = sorted(self._omni_dof_names, key=lambda dof_name: self._omni_dof_indices[self._omni_dof_names.index(dof_name)])
            # Sort dof_indices
            self._omni_dof_indices = sorted(self._omni_dof_indices)

            # Convert dof indices to torch tensor
            self._omni_dof_indices = torch.tensor(self._omni_dof_indices, device=self.device, dtype=torch.long)

        print("Dof names: ", self._omni_dof_names)
        print("Dof indices: ", self._omni_dof_indices)

        self._local_origin_positions = env_pos

        # Set control mode to torque control
        self._robot_articulation.switch_control_mode("effort")
        # set joint damping
        damping = torch.tensor(self.joint_damping, dtype=torch.float32, device=self.device).repeat(self.num_envs, 12)
        self._robot_articulation.set_gains(kds=damping, joint_indices=self._omni_dof_indices)
        # Set joint friction
        friction = torch.tensor(self.joint_friction, dtype=torch.float32, device=self.device).repeat(self.num_envs, 12)
        self._robot_articulation.set_friction_coefficients(values=friction, joint_indices=self._omni_dof_indices)

        # Get gains
        kps, kds = self._robot_articulation.get_gains()
        print("Dof KP: ", kps[0, :])
        print("Dof Kds: ", kds[0, :])
        # Get friciton
        friction = self._robot_articulation.get_friction_coefficients()
        print("Dof friction: ", friction[0, :])

        self.init_state_buffers(device, num_envs)

    def init_state_buffers(self, device, num_envs):
        # Default robot states
        self.default_joint_positions = torch.tensor(self.default_joint_positions, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.default_base_positions = torch.tensor(self.default_base_position, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.default_base_quaternions = torch.tensor(self.default_base_quaternion, dtype=torch.float32, device=device).repeat(num_envs, 1)

        # Control parameters
        self.control_kp_tensor = torch.tensor([self.control_kp], dtype=torch.float32, device=device).repeat(num_envs, 12)
        self.control_kd_tensor = torch.tensor([self.control_kd], dtype=torch.float32, device=device).repeat(num_envs, 12)
        self.action_scale_tensor = torch.tensor([self.action_scale], dtype=torch.float32, device=device).repeat(num_envs, 12)
        self.max_torque_tensor = torch.tensor([self.max_torque], dtype=torch.float32, device=device).repeat(num_envs, 12)

        self.current_joint_position_targets_se = torch.tensor(self.default_joint_position_targets_se, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.current_joint_position_targets = self.default_joint_positions[:, 0:12].clone()
        self.last_joint_position_targets_se = torch.tensor(self.default_joint_position_targets_se, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.last_joint_position_targets = self.default_joint_positions[:, 0:12].clone()
        self.joint_position_target_se_upper = torch.tensor(self.max_joint_pos_swing_ext, dtype=torch.float32, device=device)
        self.joint_position_target_se_lower = torch.tensor(self.min_joint_pos_swing_ext, dtype=torch.float32, device=device)
        self.joint_position_target_lower = torch.tensor(self.min_joint_pos, dtype=torch.float32, device=device)
        self.joint_position_target_upper = torch.tensor(self.max_joint_pos, dtype=torch.float32, device=device)

        self.current_joint_positions = self.default_joint_positions[:, 0:12].clone()
        self.current_joint_velocities = torch.zeros(num_envs, 12, dtype=torch.float32, device=device)
        self.current_joint_torques = torch.zeros(num_envs, 12, dtype=torch.float32, device=device)

        self.last_base_in_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_link_in_contact = torch.zeros(num_envs, 16, dtype=torch.bool, device=device)

        if self.module_config == "horizontal":
            # For horizontal configuration
            self.__corner_pos_robot = torch.cat((
                torch.tensor([0.075, 0.1835, -0.04]).repeat(self._num_envs, 1),
                torch.tensor([-0.075, 0.1835, -0.04]).repeat(self._num_envs, 1),
                torch.tensor([0.075, -0.1835, -0.04]).repeat(self._num_envs, 1),
                torch.tensor([-0.075, -0.1835, -0.04]).repeat(self._num_envs, 1)
            ), dim=-1).view(self._num_envs, 4, 3).to(torch.float32).to(self._device)
            
        elif self.module_config == "vertical":
            self.__corner_pos_robot = torch.cat((
                torch.tensor([0.0, -0.115, -0.1853]).repeat(self._num_envs, 1),
                torch.tensor([0.0, 0.115, -0.1853]).repeat(self._num_envs, 1),
                torch.tensor([-0.115, 0.0, -0.1853]).repeat(self._num_envs, 1),
                torch.tensor([0.115, 0.0, -0.1853]).repeat(self._num_envs, 1)
            ), dim=-1).view(self._num_envs, 4, 3).to(torch.float32).to(self._device)  

    def get_joint_positions(self):
        return self._robot_articulation.get_joint_positions(joint_indices=self._omni_dof_indices, clone=False)
    
    def get_joint_velocities(self):
        return self._robot_articulation.get_joint_velocities(joint_indices=self._omni_dof_indices, clone=False)
    
    def get_joint_torques(self):
        return self.current_joint_torques
    
    def get_base_poses(self):
        base_position_world, base_quaternions = self._robot_articulation.get_world_poses(clone=False)
        return base_position_world - self._local_origin_positions, base_quaternions
    
    def get_base_angular_velocities(self):
        return self._robot_articulation.get_angular_velocities(clone=False)
    
    def get_base_linear_velocities(self):
        return self._robot_articulation.get_linear_velocities(clone=False)
    
    def get_tip_positions(self):
        # update tip positions
        tip_positions, _ = self._tip_view.get_world_poses()
        # Transform to shape (num_envs, num_modules, 3)
        tip_positions_world = tip_positions.view(self.num_envs, num_modules, 3)
        local_tip_origin_positions = self._local_origin_positions.repeat(1, num_modules).view(self.num_envs, num_modules, 3)
        return tip_positions_world - local_tip_origin_positions
    
    def get_tip_velocities(self):
        return self._tip_view.get_linear_velocities()
    
    def get_knee_positions(self):
        raw_positions = self._knee_view.get_world_poses(clone=False)[0].view(self.num_envs, num_modules*2, 3)
        local_origins = self._local_origin_positions.repeat(1, num_modules*2).view(self.num_envs, num_modules*2, 3)
        return raw_positions - local_origins
    
    def get_corner_positions(self):
        base_positions, base_quaternions = self.get_base_poses()
        return transform_vectors(base_quaternions,
                                 base_positions,
                                 self.__corner_pos_robot,
                                 self._device)
    
    def set_joint_positions(self, positions, indices=None, full_joint_indices=False):
        """
            Args:
                positions: the positions of shape (num_envs, num_targets)
                indices: the env indices to be set
                full_joint_indices: whether or not the target positions include passive joints
        """
        if full_joint_indices:
            joint_indices = None
        else:
            joint_indices = self._omni_dof_indices
        self._robot_articulation.set_joint_positions(positions, indices=indices, joint_indices=joint_indices)

    def set_joint_velocities(self, velocities, indices=None, full_joint_indices=False):
        """
            Args:
                velocities: the velocities of shape (num_envs, num_targets)
                indices: the env indices to be set
                full_joint_indices: whether or not the target velocities include passive joints
        """
        if full_joint_indices:
            joint_indices = None
        else:
            joint_indices = self._omni_dof_indices
        self._robot_articulation.set_joint_velocities(velocities, indices=indices, joint_indices=joint_indices)

    def set_robot_pose(self, positions=None, quaternions=None, indices=None):
        '''
            Set robot positions and orientations in their local frames
        '''
        # Calculate global positions
        global_positions = None
        if positions is not None:
            if indices is None:
                global_positions = positions + self._local_origin_positions
            else:
                # indices is specified
                global_positions = positions + self._local_origin_positions[indices, :]

        self._robot_articulation.set_world_poses(global_positions, quaternions, indices)

    def set_robot_velocities(self, velocities, indices=None):
        self._robot_articulation.set_velocities(velocities, indices=indices)
    
    def take_action(self, 
                    actions, 
                    world, 
                    control_mode="increment", 
                    swing_extension_control=True,
                    render_subframes=False):
        '''
            control_mode:
                (1) increment: the actions are increments to the current joint position targets
                (2) direct: the actions are joint position targets scaled to joint lower and upper bounds
        '''
        # actions[:] = 0.0
        # Get joint_position target changes

        if control_mode == "increment":
            delta_joint_position_targets = torch.mul(actions, self.action_scale_tensor)
            if swing_extension_control:
                # Dof2 and Dof3 will be converted to swing extension values
                self.current_joint_position_targets_se = self.current_joint_position_targets_se + delta_joint_position_targets
                # Clamp the joint position targets
                self.current_joint_position_targets_se = torch.clamp(self.current_joint_position_targets_se, 
                                                                    min=self.joint_position_target_se_lower, 
                                                                    max=self.joint_position_target_se_upper)
                
                # Convert swing extension angles to dof2 and dof3 angles
                joint_swing = self.current_joint_position_targets_se[:, [4,6,8,10]]
                joint_extension = self.current_joint_position_targets_se[:, [5,7,9,11]]
                dof2_joint_positions = joint_swing + joint_extension/2.0
                dof3_joint_positions = joint_swing - joint_extension/2.0
                self.current_joint_position_targets[:, 0:4] = self.current_joint_position_targets_se[:, 0:4] # For dof1, they are the same
                self.current_joint_position_targets[:, [4,6,8,10]] = dof2_joint_positions # For dof2 joint positions
                self.current_joint_position_targets[:, [5,7,9,11]] = dof3_joint_positions # For dof3 joint positions
            else:
                # The actions are just goal joint positions
                self.current_joint_position_targets = self.current_joint_position_targets + delta_joint_position_targets
                # Clamp the joint position targets
                self.current_joint_position_targets = torch.clamp(self.current_joint_position_targets,
                                                                min=self.joint_position_target_lower,
                                                                max=self.joint_position_target_upper)
        elif control_mode == "direct":
            if swing_extension_control:
                # Project action to joint lower and upper bounds
                self.current_joint_position_targets_se = unscale_transform(actions, 
                                                                        self.joint_position_target_se_lower.repeat(self.num_envs, 1),
                                                                        self.joint_position_target_se_upper.repeat(self.num_envs, 1))
                # Convert swing extension angles to dof2 and dof3 angles
                joint_swing = self.current_joint_position_targets_se[:, [4,6,8,10]]
                joint_extension = self.current_joint_position_targets_se[:, [5,7,9,11]]
                dof2_joint_positions = joint_swing + joint_extension/2.0
                dof3_joint_positions = joint_swing - joint_extension/2.0
                self.current_joint_position_targets[:, 0:4] = self.current_joint_position_targets_se[:, 0:4] # For dof1, they are the same
                self.current_joint_position_targets[:, [4,6,8,10]] = dof2_joint_positions # For dof2 joint positions
                self.current_joint_position_targets[:, [5,7,9,11]] = dof3_joint_positions # For dof3 joint positions
            else:
                self.current_joint_position_targets = unscale_transform(actions, 
                                                                        self.joint_position_target_lower.repeat(self.num_envs, 1),
                                                                        self.joint_position_target_upper.repeat(self.num_envs, 1))

        # # a1 dof2, dof3
        # self.current_joint_position_targets[:, 4:6] += swing_offset
        # # a2 dof2, dof3
        # self.current_joint_position_targets[:, 6:8] -= swing_offset
        # # a3 dof2, dof3
        # self.current_joint_position_targets[:, 8:10] -= swing_offset
        # # a4 dof2, dof3
        # self.current_joint_position_targets[:, 10:12] += swing_offset

        for _ in range(self.actual_control_decimal):
            # apply torque to the joints
            torque = torch.mul(self.control_kp_tensor, (self.current_joint_position_targets-self.current_joint_positions)) - \
                    torch.mul(self.control_kd_tensor, self.current_joint_velocities)
            torque = torch.clamp(torque, min=-self.max_torque_tensor, max=self.max_torque_tensor)
            self._robot_articulation.set_joint_efforts(torque, joint_indices=self._omni_dof_indices)

            from omni.isaac.core.simulation_context import SimulationContext
            SimulationContext.step(world, render=render_subframes)

            # Get current joint positions and velocities
            self.current_joint_positions = self.get_joint_positions()
            self.current_joint_velocities = self.get_joint_velocities()

        # apply torque to the joints
        torque = torch.mul(self.control_kp_tensor, (self.current_joint_position_targets-self.current_joint_positions)) - \
                torch.mul(self.control_kd_tensor, self.current_joint_velocities)
        torque = torch.clamp(torque, min=-self.max_torque_tensor, max=self.max_torque_tensor)
        self._robot_articulation.set_joint_efforts(torque, joint_indices=self._omni_dof_indices)

        # Update joint torques
        self.current_joint_torques = torque

    def reset_robot_joint_positions(self, indices, rand_deviation=0.0):
        '''
            Args:
                indices: the env ids to reset joint positions
                rand_deviation: this will randomly set a deviation angle from -rand_deviation to rand_deviation

            deviation_angle = rand_deviation * rand(-1, 1)
            reset_joint_angle = default_joint_angle + deviation_angle
        '''
        num_reset = len(indices)

        rand_tensor = 2.0*(torch.rand(num_reset, 12, dtype=torch.float32, device=self._device)-0.5) # From -1 to 1
        rand_deviation = rand_deviation * rand_tensor
        deviation = torch.zeros(num_reset, 20, dtype=torch.float32, device=self._device)
        deviation[:, 0:12] = rand_deviation

        joint_positions = self.default_joint_positions[indices] + deviation
        # Set joint positions
        self.set_joint_positions(joint_positions, indices=indices, full_joint_indices=True)

        # Reset current_joint_positions
        self.current_joint_positions[indices] = joint_positions[:, 0:12]
        # Reset current_joint_position_targets
        self.current_joint_position_targets[indices] = joint_positions[:, 0:12]
        self.last_joint_position_targets[indices] = joint_positions[:, 0:12]
        # Calculate swing extension angles
        dof2_joint_positions = joint_positions[:, [4,6,8,10]]
        dof3_joint_positions = joint_positions[:, [5,7,9,11]]
        joint_swing = (dof2_joint_positions + dof3_joint_positions)/2.0
        joint_extension = dof2_joint_positions - dof3_joint_positions
        new_swing_extension_values = torch.zeros(num_reset, 12, dtype=torch.float32, device=self._device)
        new_swing_extension_values[:, 0:4] = self.current_joint_position_targets[indices, 0:4]
        new_swing_extension_values[:, [4,6,8,10]] = joint_swing
        new_swing_extension_values[:, [5,7,9,11]] = joint_extension
        self.current_joint_position_targets_se[indices, :] = new_swing_extension_values
        self.last_joint_position_targets_se[indices, :] = new_swing_extension_values

    def reset_robot_joint_velocities(self, indices, rand_max_joint_velocities=0.0):
        '''
            Args:
                indices: the env ids to reset joint velocities
                rand_deviation: this will randomly set joint velocities from -rand_max_joint_velocities to rand_max_joint_velocities
        '''
        num_reset = len(indices)
        rand_tensor = 2*(torch.rand(num_reset, 20, dtype=torch.float32, device=self._device)-0.5) # from -1 ~ 1

        rand_joint_velocities = rand_max_joint_velocities * rand_tensor

        self.current_joint_velocities[indices, :] = rand_joint_velocities[:, 0:12]

        self.set_joint_velocities(rand_joint_velocities, indices=indices, full_joint_indices=True)

    def reset_robot_poses(self, indices, 
                              rand_max_xy_deviation=0.0, 
                              rand_max_z_deviation=0.0,
                              rand_max_roll=0.0,
                              rand_max_pitch=0.0,
                              rand_max_yaw=0.0):
        '''
            Args:
                indices: the env ids to reset robot base positions and quaternions
                rand_max_xy_deviation: this will add default base x and y position with random values from -rand_max_xy_deviation to rand_max_xy_deviation
                rand_max_z_deviation: this will add default base z position with random values from -rand_max_z_deviation to rand_max_z_deviation
        '''
        num_reset = len(indices)

        # Get reset robot position
        random_tensor = 2.0*(torch.rand(num_reset, 3, dtype=torch.float32, device=self._device)-0.5)
        
        xy_deviation = rand_max_xy_deviation*random_tensor[:, 0:2]
        z_deviation = rand_max_z_deviation*random_tensor[:, 2].view(num_reset, 1)     

        xyz_deviation = torch.cat((
            xy_deviation,
            z_deviation
        ), dim=-1)

        robot_position = self.default_base_positions[indices]+xyz_deviation

        # Get reset robot quaternion
        rand_init_quaternion = rand_quaternions(num_reset,
                                                min_roll=-rand_max_roll,
                                                max_roll=rand_max_roll,
                                                min_pitch=-rand_max_pitch,
                                                max_pitch=rand_max_pitch,
                                                min_yaw=-rand_max_yaw,
                                                max_yaw=rand_max_yaw,
                                                device=self.device)
        # Apply the rand init quaternion to the default quaternion
        robot_quaternion = rotate_orientations(rand_init_quaternion, 
                                               self.default_base_quaternions[indices],
                                               device=self.device)
        
        # Set robot poses
        self.set_robot_pose(positions=robot_position,
                            quaternions=robot_quaternion,
                            indices=indices)
        
    def reset_robot_velocities(self, 
                               indices,
                               rand_max_linear_velocity=0.0, 
                               rand_max_angular_velocity=0.0):
        '''
            Args:
                indices: the env ids to reset robot base linear and angular velocities
                rand_max_linear_velocity: the max random velocities to be set to the base linear velocities
                rand_max_angular_velocity: the max random velocities to be set to the base angular velocities
        '''
        num_reset = len(indices)

        rand_tensor = 2.0*(torch.rand(num_reset, 6, dtype=torch.float32, device=self._device)-0.5)

        rand_linear_vel = rand_max_linear_velocity * rand_tensor[:, 0:3]
        rand_angular_vel = rand_max_angular_velocity * rand_tensor[:, 3:6]

        rand_velocities = torch.cat((
            rand_linear_vel,
            rand_angular_vel
        ), dim=-1)

        self.set_robot_velocities(rand_velocities, indices=indices)
        
    def randomize_control_params(self,
                                 control_kp_scale_range,
                                 control_kd_scale_range,
                                 action_scale_range,
                                 max_torque_scale_range):
        '''
            Randomize control parameters

            The range specifies the range of a multiplier to the original parameters, i.e., 
            if control_kp_scale_range is [0.9, 1.1], this means every motor in the evironment will
            be randomly be multipled by a value from 0.9~1.1
        '''
        rand_float = torch.rand(4, self._num_envs, 12, device=self._device)
        rand_control_kp_range = control_kp_scale_range[0] + rand_float[0, :, :] * (control_kp_scale_range[1] - control_kp_scale_range[0])
        rand_control_kd_range = control_kd_scale_range[0] + rand_float[1, :, :] * (control_kd_scale_range[1] - control_kd_scale_range[0])
        rand_action_scale_range = action_scale_range[0] + rand_float[2, :, :] * (action_scale_range[1] - action_scale_range[0])
        rand_max_torque_range = max_torque_scale_range[0] + rand_float[3, :, :] * (max_torque_scale_range[1] - max_torque_scale_range[0])

        self.control_kp_tensor = rand_control_kp_range * self.control_kp
        self.control_kd_tensor = rand_control_kd_range * self.control_kd
        self.action_scale_tensor = rand_action_scale_range * self.action_scale
        self.max_torque_tensor = rand_max_torque_range * self.max_torque

    def check_links_self_collision(self):
        if self.module_config == "horizontal":
            '''joint limit penalty'''
            joint_positions = self.current_joint_positions
            a14_dof1_positions = joint_positions[:, [0,3]]
            a23_dof1_positions = joint_positions[:, [1,2]]
            dof2_positions = joint_positions[:, [4,6,8,10]]
            dof3_positions = joint_positions[:, [5,7,9,11]]
            # Check if position difference between DoF2&3 break the reset limit
            joint23_pos_diffs = torch.abs(dof3_positions - dof2_positions)
            joint23_pos_diff_low_reset = joint23_pos_diffs < self.reset_min_joint_23_diff
            joint23_pos_diff_high_reset = joint23_pos_diffs > self.reset_max_joint_23_diff
            joint23_pos_reset_sumup = joint23_pos_diff_low_reset + joint23_pos_diff_high_reset # shape (N, 4)
            joint23_pos_reset = torch.sum(joint23_pos_reset_sumup, dim=-1) # Shape (N, 1)
            # Check if DoF 1 break the reset limit
            a14_dof1_reset_min_pos = self.reset_min_joint_1_pos
            a14_dof1_reset_max_pos = self.reset_max_joint_1_pos
            a23_dof1_reset_min_pos = -self.reset_max_joint_1_pos
            a23_dof1_reset_max_pos = -self.reset_min_joint_1_pos
            a14_dof_pos_low_reset = a14_dof1_positions < a14_dof1_reset_min_pos
            a14_dof_pos_high_reset = a14_dof1_positions > a14_dof1_reset_max_pos
            a23_dof_pos_low_reset = a23_dof1_positions < a23_dof1_reset_min_pos
            a23_dof_pos_high_reset = a23_dof1_positions > a23_dof1_reset_max_pos
            joint1_pos_reset_sumup = a14_dof_pos_low_reset + a14_dof_pos_high_reset + \
                                    a23_dof_pos_low_reset + a23_dof_pos_high_reset # Shape (N, 2)
            joint1_pos_reset = torch.sum(joint1_pos_reset_sumup, dim=-1) # Shape (N, 1)

            joint_reset_num = joint1_pos_reset + joint23_pos_reset

        elif self.module_config == "vertical":
            joint_positions = self.current_joint_positions
            dof1_positions = joint_positions[:, 0:4]
            dof2_positions = joint_positions[:, [4,6,8,10]]
            dof3_positions = joint_positions[:, [5,7,9,11]]
            # Check if position difference between DoF2&3 break the reset limit
            joint23_pos_diffs = torch.abs(dof3_positions - dof2_positions)
            joint23_pos_diff_low_reset = joint23_pos_diffs < self.reset_min_joint_23_diff
            joint23_pos_diff_high_reset = joint23_pos_diffs > self.reset_max_joint_23_diff
            joint23_pos_reset_sumup = joint23_pos_diff_low_reset + joint23_pos_diff_high_reset # shape (N, 4)
            joint23_pos_reset = torch.sum(joint23_pos_reset_sumup, dim=-1) # Shape (N, 1)
            # Check if DoF 1 break the reset limit
            dof1_too_low_reset = dof1_positions < self.reset_min_joint_1_pos
            dof1_too_high_reset = dof1_positions > self.reset_max_joint_1_pos
            joint1_pos_reset_sumup = dof1_too_low_reset + dof1_too_high_reset # Shape (N, 4)
            joint1_pos_reset = torch.sum(joint1_pos_reset_sumup, dim=-1) # Shape (N, 1)

            joint_reset_num = joint1_pos_reset + joint23_pos_reset
        
        return joint_reset_num > 0
    
    def check_base_contact(self, force_threshold=1.0, dt=0.005):
        current_base_contact_forces = self._base_view.get_net_contact_forces(clone=False, dt=dt)
        force_norm = torch.norm(current_base_contact_forces, dim=-1)
        force_above_threshold = force_norm > force_threshold

        # The base is seen to have contact only when last is not in contact
        # and the current is in contact
        last_not_in_contact = torch.logical_not(self.last_base_in_contact)
        actual_in_contact = torch.logical_and(last_not_in_contact, force_above_threshold)

        # Update last contact 
        self.last_base_in_contact = force_above_threshold

        return actual_in_contact
    
    def check_link_contact(self, force_threshold=1.0, dt=0.005):
        current_link_contact_forces = self._link_view.get_net_contact_forces(clone=False, dt=dt)
        force_norm = torch.norm(current_link_contact_forces, dim=-1).view(self.num_envs, 16)
        force_above_threshold = force_norm > force_threshold

        # The knee is seen to have contact only when last is not in contact
        # and the current is in contact
        last_not_in_contact = torch.logical_not(self.last_link_in_contact)
        actual_in_contact = torch.logical_and(last_not_in_contact, force_above_threshold)

        # Update last contact
        self.last_link_in_contact = force_above_threshold

        return torch.sum(actual_in_contact, dim=-1) > 0
    
    def reset_last_contact_buffers(self, indices):
        num_reset = len(indices)
        self.last_base_in_contact[indices] = 0
        self.last_link_in_contact[indices, :] = torch.zeros(num_reset, 8, dtype=torch.bool, device=self.device)

    @property
    def _device(self):
        return self.device
    
    @property
    def _num_envs(self):
        return self.num_envs
