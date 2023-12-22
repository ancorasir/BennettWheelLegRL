import os

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

import torch

class Semidodecahedron(RigidObjectOmni):
    def __init__(self, 
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=1.0,
                 mass=None,
                 inertia=None,
                 com=None,
                 object_name="semidodecahedron",
                 fixed=False) -> None:
        
        if not fixed:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "semidodecahedron", "semidodecahedron.usd")
        else:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "semidodecahedron", "semidodecahedron_fixed.usd")

        self.scale = scale
        self.mass = mass
        self.inertia = inertia
        self.com = com

        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        object_prim_name = "/semidodecahedron"
        super().__init__(object_name, 
                         object_usd_path, 
                         object_prim_name, 
                         default_object_position=default_position, 
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor([self.scale, self.scale, self.scale]),
                         mass=self.mass,
                         inertia=self.inertia,
                         com=self.com)