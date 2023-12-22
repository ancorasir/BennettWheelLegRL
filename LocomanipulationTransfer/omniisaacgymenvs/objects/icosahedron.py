import os

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

import torch

class Icosahedron(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0,
                 mass=None,
                 inertia=None,
                 object_name="icosahedron") -> None:
        
        object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "Icosahedron", "Icosahedron.usd")

        self.scale = scale
        self.mass = mass
        self.inertia = inertia

        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        object_prim_name = "/Icosahedron"
        super().__init__(object_name, 
                         object_usd_path, 
                         object_prim_name, 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor([self.scale, self.scale, self.scale]))

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        super().post_reset_object(env_pos, num_envs, device)

        # Set the mass of the ball
        if self.mass is not None:
            ball_mass_tensor = torch.tensor([self.mass], dtype=torch.float32, device=self._device).repeat(self._num_envs)
            self.object_view.set_masses(ball_mass_tensor)

        if self.inertia is not None:
            object_inertias = torch.tensor(self.inertia, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            self.object_view.set_inertias(object_inertias)

class HalfIcosahedron(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0,
                 mass=None,
                 inertia=None,
                 com=None,
                 object_name="icosahedron") -> None:
        
        object_usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "half_icosahedron", "half_icosahedron.usd")

        if scale == 1.0:
            self.scale = None
        else:
            self.scale = torch.tensor([scale, scale, scale])

        self.mass = mass
        self.inertia = inertia

        object_prim_name = "/half_icosahedron"
        super().__init__(object_name, 
                         object_usd_path, 
                         object_prim_name, 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=self.scale,
                         mass=self.mass,
                         inertia=self.inertia,
                         com=com)