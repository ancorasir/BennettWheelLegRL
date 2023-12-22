import torch

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView, XFormPrim, XFormPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView

from utils.math import transform_vectors, rand_quaternions, rotate_orientations

class RigidObjectOmni(object):
    def __init__(self, 
                 object_name, 
                 object_usd_path,
                 object_prim_name,
                 default_object_position=[0.0, 0.0, 0.0],
                 default_object_quaternion=[1.0, 0.0, 0.0, 0.0],
                 scale=torch.tensor([1.0, 1.0, 1.0]),
                 mass=None,
                 inertia=None,
                 com=None) -> None:
        self.object_name = object_name
        self.object_usd_path = object_usd_path
        self.object_prim_name = object_prim_name
        self.env_pos = None

        self.default_object_position = default_object_position
        self.default_object_quaternion = default_object_quaternion
        self.scale = scale
        self.mass = mass
        self.inertia = inertia
        self.com=com

        self.object_xform_prim = None
        self.object_view = None

        # Default values
        self._num_envs = 1
        self._device = "cuda"

    def init_stage_object(self, default_zero_env_path="/World/envs/env_0", sim_config=None):
        add_reference_to_stage(self.object_usd_path, 
                               default_zero_env_path + "/" + self.object_name)
        self.object_xform_prim = XFormPrim(prim_path=default_zero_env_path + "/" + self.object_name + self.object_prim_name,
                                           name=self.object_name,
                                           scale=self.scale)
        
        if sim_config is not None:
            sim_config.apply_articulation_settings(self.object_name, 
                                                   get_prim_at_path(self.object_xform_prim.prim_path), 
                                                   sim_config.parse_actor_config(self.object_name))

    def init_object_view(self, scene):
        self.object_view = RigidPrimView(prim_paths_expr="/World/envs/.*/{}{}".format(self.object_name, self.object_prim_name), 
                                         name=self.object_name + "_view",
                                         reset_xform_properties=False)

        # self.object_view = ArticulationView(prim_paths_expr="/World/envs/.*/{}{}".format(self.object_name, self.object_prim_name), 
        #                                  name=self.object_name + "_view",
        #                                  reset_xform_properties=False)
        scene.add(self.object_view)
    
    def set_env_pos(self, env_pos):
        self.env_pos = env_pos

    def post_reset_object(self, env_pos, num_envs, device="cuda"):
        self._num_envs = num_envs
        self._device = device
        self.env_pos = env_pos

        self.default_object_positions = torch.tensor(self.default_object_position, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
        self.default_object_quaternions = torch.tensor(self.default_object_quaternion, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)

        # Set the mass of the ball
        if self.mass is not None:
            mass_tensor = torch.tensor([self.mass], dtype=torch.float32, device=self._device).repeat(self._num_envs)
            self.object_view.set_masses(mass_tensor)

        if self.inertia is not None:
            inertia_tensor = torch.tensor(self.inertia, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            self.object_view.set_inertias(inertia_tensor)

        if self.com is not None:
            com_tensor = torch.tensor(self.com, dtype=torch.float32, device=self._device).repeat(self._num_envs, 1)
            self.object_view.set_coms(positions=com_tensor)

    def set_object_poses(self, positions=None, quaternions=None, indices=None):
        if positions is not None:
            # If positions are going to be reset
            if indices is not None:
                local_positions = self.env_pos[indices]
            else:
                local_positions = self.env_pos

            new_world_positions = local_positions + positions
            self.object_view.set_world_poses(new_world_positions, quaternions, indices=indices)
        else:
            self.object_view.set_world_poses(positions, quaternions, indices=indices)

    def set_object_velocities(self, velocities, indices=None):
        self.object_view.set_velocities(velocities, indices)

    def get_object_poses(self):
        world_position, quaternion = self.object_view.get_world_poses(clone=False)
        local_positions = world_position - self.env_pos
        return local_positions, quaternion

    def get_object_velocities(self):
        return self.object_view.get_linear_velocities(clone=False), self.object_view.get_angular_velocities(clone=False)
    
    def reset_object_poses(self, indices, 
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

        object_position = self.default_object_positions[indices]+xyz_deviation

        # Get reset robot quaternion
        rand_init_quaternion = rand_quaternions(num_reset,
                                                min_roll=-rand_max_roll,
                                                max_roll=rand_max_roll,
                                                min_pitch=-rand_max_pitch,
                                                max_pitch=rand_max_pitch,
                                                min_yaw=-rand_max_yaw,
                                                max_yaw=rand_max_yaw,
                                                device=self._device)
        # Apply the rand init quaternion to the default quaternion
        object_quaternion = rotate_orientations(rand_init_quaternion, 
                                               self.default_object_quaternions[indices],
                                               device=self._device)
        
        # Set robot poses
        self.set_object_poses(positions=object_position,
                              quaternions=object_quaternion,
                              indices=indices)
        
    def reset_object_velocities(self,
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

        self.set_object_velocities(rand_velocities, indices=indices)