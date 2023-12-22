import os
import torch

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

class Ball(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0,
                 fixed=False) -> None:
        if fixed:
            object_name = "ball_fixed"
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "ball", "ball_fixed.usd")
            # object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "ball", "ball.usd")
        else:
            object_name = "ball"
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "ball", "ball.usd")
        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        
        object_prim_name = "/ball" # "/ball"
        self.radius = scale * 0.1
        self.scale = [scale, scale, scale]

        super().__init__(object_name, 
                         object_usd_path, 
                         object_prim_name, 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor(self.scale))
        

from robot.omni_robots import OmniverseRobot

class BallWithFrame(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0,
                 mass=None,
                 inertia=None,
                 com=None,
                 object_name="ball",
                 fixed=False) -> None:
        
        if not fixed:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "ball_with_frame", "ball_with_frame.usd")
        else:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "ball_with_frame", "ball_with_frame_fixed.usd")

        self.scale = scale
        self.mass = mass
        self.inertia = inertia

        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        object_prim_name = "/ball"
        super().__init__(object_name, 
                         object_usd_path, 
                         object_prim_name, 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor([self.scale, self.scale, self.scale]),
                         mass=self.mass,
                         inertia=self.inertia,
                         com=com)

class BallWithFrameLegacy(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0, 
                 mass=None, 
                 inertia=None, 
                 as_goal=False,
                 object_name="") -> None:
        self.default_position = default_position
        self.default_quaternion = default_quaternion
        self.scale = scale
        self.mass = mass
        self.inertia = inertia
        self.as_goal = as_goal
        self.object_name = object_name

        if not self.as_goal:
            self.usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "ball_with_frame", "ball_with_frame_backup.usd")
        else:
            self.usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "ball_with_frame", "ball_with_frame_backup.usd")

        if self.as_goal:
            self.object_name = object_name + "goal"

        super().__init__(self.object_name, 
                         self.usd_path, 
                         "/" + self.object_name + "/ball", 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor(self.scale))

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        if not self.as_goal:
            add_reference_to_stage(self.usd_path, default_zero_env_path + "/" + self.object_name)
            obj = XFormPrim(
                prim_path=default_zero_env_path + "/" + self.object_name + "/ball",
                name=self.object_name,
                translation=self.default_position,
                orientation=self.default_quaternion,
                scale=[self.scale,self.scale,self.scale],
            )
            # stage = get_current_stage()
            # prim_path = default_zero_env_path + "/" + "ball"
            
            # obj = OmniverseRobot(prim_path,
            #                     "ball",
            #                     self.usd_path,
            #                     self.default_position,
            #                     self.default_quaternion)
            sim_config.apply_articulation_settings(self.object_name, get_prim_at_path(obj.prim_path), sim_config.parse_actor_config(self.object_name))
        else:
            add_reference_to_stage(self.usd_path, default_zero_env_path + "/goal")
            goal = XFormPrim(
                prim_path=default_zero_env_path + "/goal",
                name="goal",
                translation=self.default_position,
                orientation=self.default_quaternion,
            )
            sim_config.apply_articulation_settings("goal", get_prim_at_path(goal.prim_path), sim_config.parse_actor_config("goal"))

    def init_object_view(self, scene):
        if not self.as_goal:
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/" + self.object_name + "/ball",
                                    name="object_view", 
                                    reset_xform_properties=False)
            # self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/ball",
            #                         name="object_view", 
            #                         reset_xform_properties=False)
            # self.object_view = ArticulationView(prim_paths_expr="/World/envs/env_.*/ball",
            #                         name="object_view", 
            #                         reset_xform_properties=False)
            scene.add(self.object_view)
        else:
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/goal/ball", 
                                    name="goal_view", 
                                    reset_xform_properties=False)
            self.object_view._non_root_link = True # hack to ignore kinematics
            scene.add(self.object_view)

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        super().post_reset_object(env_pos, num_envs, device)

        # Set the mass of the ball
        if self.mass is not None:
            ball_mass_tensor = torch.tensor([self.mass], dtype=torch.float32, device=self._device).repeat(self._num_envs)
            self.object_view.set_masses(ball_mass_tensor)

        if self.inertia is not None:
            object_inertias = torch.tensor(self.inertia, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            self.object_view.set_inertias(object_inertias)

class BallLockXY(RigidObjectOmni):
    def __init__(self, default_object_position, default_object_quaternion) -> None:
        self.usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "ball_lock_xy", "ball_lock_xy.usd")
        self.object_name = "ball"
        self.object_prim_name = ""
        self.default_position = default_object_position
        self.default_quaternion = default_object_quaternion

        self._omniverse_robot = None
        self._robot_articulation = None
        self._local_origin_positions = None

        self._ball_view = None

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        """
            Initialize the Omniverse Robot instance; Add the first Omniverse Robot to the stage
            for cloning;
        """
        prim_path = default_zero_env_path + "/" + self.object_name
        
        self._omniverse_robot = OmniverseRobot(prim_path,
                                               self.object_name,
                                               self.usd_path,
                                               self.default_position,
                                               [1.0, 0.0, 0.0, 0.0])
        
        if sim_config is not None:
            sim_config.apply_articulation_settings(self.object_name, get_prim_at_path(self._omniverse_robot.prim_path), sim_config.parse_actor_config(self.object_name))

    def init_object_view(self, scene):
        arti_root_name = "/World/envs/.*/" + self.object_name + "/rack_base"
        self._robot_articulation = ArticulationView(prim_paths_expr=arti_root_name, name=self.object_name)
        scene.add(self._robot_articulation)

        ball_path = "/World/envs/.*/" + self.object_name + "/ball"
        self._ball_view = RigidPrimView(prim_paths_expr=ball_path, name="ball_view")
        scene.add(self._ball_view)

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self._local_origin_positions = env_pos

    def get_object_poses(self):
        base_position_world, base_quaternions = self._ball_view.get_world_poses(clone=False)
        return base_position_world - self._local_origin_positions, base_quaternions
    
    def get_object_velocities(self):
        return self._ball_view.get_linear_velocities(clone=False), self._ball_view.get_angular_velocities(clone=False)
    
    def reset_object_poses(self, env_ids):
        num_reset = len(env_ids)
        default_joint_positions = torch.tensor([0.0, torch.pi, -torch.pi/2.0, 0.0], dtype=torch.float32, device=self.device).repeat(num_reset, 1)
        self._robot_articulation.set_joint_positions(default_joint_positions, indices=env_ids)

    def reset_object_velocities(self, env_ids):
        num_reset = len(env_ids)
        default_joint_velocities = torch.zeros(num_reset, 4, dtype=torch.float32, device=self.device)
        self._robot_articulation.set_joint_velocities(default_joint_velocities, indices=env_ids)


class SemisphereWithFrame(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0,
                 mass=None,
                 inertia=None,
                 com=None,
                 object_name="ball",
                 fixed=False) -> None:
        
        if not fixed:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "semisphere_with_frame", "semisphere_with_frame.usd")
        else:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "ball_with_frame", "semisphere_with_frame_fixed.usd")

        self.scale = scale
        self.mass = mass
        self.inertia = inertia

        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        object_prim_name = "/semisphere"
        super().__init__(object_name, 
                         object_usd_path, 
                         object_prim_name, 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor([self.scale, self.scale, self.scale]),
                         mass=self.mass,
                         inertia=self.inertia,
                         com=com)
