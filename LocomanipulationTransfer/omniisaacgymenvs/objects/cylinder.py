import os
import torch

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path

class Cylinder(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=[1.0, 1.0, 1.0], 
                 mass=None, 
                 inertia=None, 
                 as_goal=False) -> None:
        self.default_position = default_position
        self.default_quaternion = default_quaternion
        self.scale = scale
        self.mass = mass
        self.inertia = inertia
        self.as_goal = as_goal

        self.usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "cylinder_aligned", "cylinder_aligned.usd")

        if self.as_goal:
            object_name = "goal"
        else:
            object_name = "cylinder"

        super().__init__(object_name, 
                         self.usd_path, 
                         "/cylinder/cylinder", 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor(self.scale))

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        if not self.as_goal:
            add_reference_to_stage(self.usd_path, default_zero_env_path + "/cylinder")
            obj = XFormPrim(
                prim_path=default_zero_env_path + "/cylinder/cylinder",
                name="cylinder",
                translation=self.default_position,
                orientation=self.default_quaternion,
                scale=self.scale,
            )
            sim_config.apply_articulation_settings("cylinder", get_prim_at_path(obj.prim_path), sim_config.parse_actor_config("cylinder"))
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
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/cylinder/cylinder",
                                    name="object_view", 
                                    reset_xform_properties=False)
            scene.add(self.object_view)
        else:
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/goal/cylinder", 
                                    name="goal_view", 
                                    reset_xform_properties=False)
            self.object_view._non_root_link = True # hack to ignore kinematics
            scene.add(self.object_view)

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        super().post_reset_object(env_pos, num_envs, device)

        # Set the mass of the ball
        if self.mass is not None:
            mass_tensor = torch.tensor([self.mass], dtype=torch.float32, device=self._device).repeat(self._num_envs)
            self.object_view.set_masses(mass_tensor)

        if self.inertia is not None:
            inertias_tensor = torch.tensor(self.inertia, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            self.object_view.set_inertias(inertias_tensor)

class CylinderWithLoad(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0, 
                 mass=None, 
                 inertia=None, 
                 as_goal=False) -> None:
        self.default_position = default_position
        self.default_quaternion = default_quaternion
        self.scale = scale
        self.mass = mass
        self.inertia = inertia
        self.as_goal = as_goal

        self.usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "cylinder_with_load", "cylinder_with_load.usd")

        if self.as_goal:
            object_name = "goal"
        else:
            object_name = "cylinder"

        super().__init__(object_name, 
                         self.usd_path, 
                         "/cylinder/cylinder", 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor(self.scale))

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        if not self.as_goal:
            add_reference_to_stage(self.usd_path, default_zero_env_path + "/cylinder")
            obj = XFormPrim(
                prim_path=default_zero_env_path + "/cylinder/cylinder",
                name="cylinder",
                translation=self.default_position,
                orientation=self.default_quaternion,
                scale=[self.scale,self.scale,self.scale],
            )
            sim_config.apply_articulation_settings("cylinder", get_prim_at_path(obj.prim_path), sim_config.parse_actor_config("cylinder"))
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
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/cylinder/cylinder",
                                    name="object_view", 
                                    reset_xform_properties=False)
            scene.add(self.object_view)
        else:
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/goal/cylinder", 
                                    name="goal_view", 
                                    reset_xform_properties=False)
            self.object_view._non_root_link = True # hack to ignore kinematics
            scene.add(self.object_view)

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        super().post_reset_object(env_pos, num_envs, device)

        # Just do not manually set the inertia

        # Set the mass of the ball
        # if self.mass is not None:
        #     mass_tensor = torch.tensor([self.mass], dtype=torch.float32, device=self._device).repeat(self._num_envs)
        #     self.object_view.set_masses(mass_tensor)

        # if self.inertia is not None:
        #     inertias_tensor = torch.tensor(self.inertia, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        #     self.object_view.set_inertias(inertias_tensor)

class CylinderOneDof(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0, 
                 mass=None, 
                 inertia=None, 
                 as_goal=False) -> None:
        self.default_position = default_position
        self.default_quaternion = default_quaternion
        self.scale = scale
        self.mass = mass
        self.inertia = inertia
        self.as_goal = as_goal

        self.usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "cylinder_aligned", "cylinder_aligned_lock_xy.usd")

        if self.as_goal:
            object_name = "goal"
        else:
            object_name = "cylinder"

        super().__init__(object_name, 
                         self.usd_path, 
                         "/cylinder/cylinder", 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor(self.scale))

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        if not self.as_goal:
            add_reference_to_stage(self.usd_path, default_zero_env_path + "/cylinder")
            obj = XFormPrim(
                prim_path=default_zero_env_path + "/cylinder/cylinder",
                name="cylinder",
                translation=self.default_position,
                orientation=self.default_quaternion,
                scale=[self.scale,self.scale,self.scale],
            )
            sim_config.apply_articulation_settings("cylinder", get_prim_at_path(obj.prim_path), sim_config.parse_actor_config("cylinder"))
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
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/cylinder/cylinder",
                                    name="object_view", 
                                    reset_xform_properties=False)
            scene.add(self.object_view)
        else:
            self.object_view = RigidPrimView(prim_paths_expr="/World/envs/env_.*/goal/cylinder", 
                                    name="goal_view", 
                                    reset_xform_properties=False)
            self.object_view._non_root_link = True # hack to ignore kinematics
            scene.add(self.object_view)

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        super().post_reset_object(env_pos, num_envs, device)

        # Set the mass of the ball
        if self.mass is not None:
            mass_tensor = torch.tensor([self.mass], dtype=torch.float32, device=self._device).repeat(self._num_envs)
            self.object_view.set_masses(mass_tensor)

        if self.inertia is not None:
            inertias_tensor = torch.tensor(self.inertia, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            self.object_view.set_inertias(inertias_tensor)