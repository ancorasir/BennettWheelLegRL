import os

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *

class Plate50cmOmni(RigidObjectOmni):
    def __init__(self, 
                 object_name="plate",
                 default_position=[0.0, 0.0, 0.0],
                 default_quaternion=[1.0, 0.0, 0.0, 0.0],
                 fixed=False) -> None:
        # object_usd_path = "/home/bionicdl/Downloads/plate_40cm/plate/plate.usd" # os.path.join(root_ws_dir, "Design", "ObjectUSD", "plate_50cm.usd")
        if not fixed:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "plate", "plate.usd")
        else:
            object_usd_path =  os.path.join(root_ws_dir, "design", "ObjectUSD", "plate", "plate_fixed.usd")

        object_prim_name = "/plate"
        super().__init__(object_name, object_usd_path, object_prim_name, default_object_position=default_position, default_object_quaternion=default_quaternion)