import os

import torch

from objects.base.rigid_object import RigidObjectOmni
from utils.path import *
from omni.isaac.core.utils.nucleus import get_assets_root_path

class PoseIndicator(RigidObjectOmni):
    def __init__(self, object_name="pose_indicator", object_prim_name="/pose_indicator", default_position=[0.0, 0.0, 0.0], default_quaternion=[1.0, 0.0, 0.0, 0.0], scale=1.0) -> None:
        # object_usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "pose_indicator.usd")
        object_usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "pose_indicator_xyz", "pose_indicator_xyz.usd")
        super().__init__(object_name, object_usd_path, object_prim_name,
                         default_object_position=default_position,
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor([scale, scale, scale]))
        
class PoseIndicatorXYZ(RigidObjectOmni):
    def __init__(self, object_name="pose_indicator_xyz", object_prim_name="/pose_indicator_xyz", default_position=[0.0, 0.0, 0.0], default_quaternion=[1.0, 0.0, 0.0, 0.0], scale=1.0) -> None:
        object_usd_path = os.path.join(root_ws_dir, "design", "ObjectUSD", "pose_indicator_xyz", "pose_indicator_xyz.usd")
        super().__init__(object_name, object_usd_path, object_prim_name,
                         default_object_position=default_position,
                         default_object_quaternion=default_quaternion,
                         scale=torch.tensor([scale, scale, scale]))
        
class PoseIndicatorOmni(RigidObjectOmni):
    def __init__(self, object_name="pose_indicator_omni", object_prim_name="", default_position=[0.0, 0.0, 0.0], default_quaternion=[1.0, 0.0, 0.0, 0.0]) -> None:
        assets_root_path = get_assets_root_path()
        object_usd_path = assets_root_path + "/Isaac/Props/UIElements/frame_prim.usd"
        super().__init__(object_name, object_usd_path, object_prim_name, default_position, default_quaternion)