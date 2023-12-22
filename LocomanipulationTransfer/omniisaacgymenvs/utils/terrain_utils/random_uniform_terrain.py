import numpy as np

from utils.terrain_utils.terrain_utils import random_uniform_terrain, SubTerrain, convert_heightfield_to_trimesh, add_terrain_to_stage
from omni.isaac.core.utils.stage import get_current_stage

class RandomUniformTerrain:
    def __init__(self, task_cfg, num_envs) -> None:
        '''
            task_cfg: the task configuration dict
            num_envs: the number of envs

            special cfg:
                side_length_per_env: this class will create a terrain in square shape for each env; 
                this parameter determines the side length of this square.
        '''
        self.terrain_cfg:dict = task_cfg["env"]["terrain"]

        self.num_envs = num_envs

        self.side_len_per_env = self.terrain_cfg.get("side_length_per_env", 2.0)
        self.horizontal_scale = self.terrain_cfg.get("horizontal_scale", 0.25)
        self.vertical_scale = self.terrain_cfg.get("vertical_scale", 0.005)
        self.min_height = self.terrain_cfg.get("min_height", -0.05)
        self.max_height = self.terrain_cfg.get("max_height", 0.05)
        self.step = self.terrain_cfg.get("step", 0.05)
        self.downsampled_scale = self.terrain_cfg.get("downsampled_scale", 0.5)
        self.slope_threshold = self.terrain_cfg.get("slope_threshold", 1.5)

        self.num_terrains = self.num_envs # One terrain for one env
        # Compute the number of rows and columns (the calculation is the same as grid cloner)
        num_clones = self.num_envs
        self._num_per_row = int(np.sqrt(num_clones))
        num_rows = np.ceil(num_clones / self._num_per_row)
        num_cols = np.ceil(num_clones / num_rows)

        self.terrain_width = num_rows * self.side_len_per_env
        self.terrain_length = num_cols * self.side_len_per_env

        self.num_row_pixels = int(num_rows * (self.side_len_per_env / self.horizontal_scale))
        self.num_col_pixels = int(num_cols * (self.side_len_per_env / self.horizontal_scale))
        self.heightfield = np.zeros((self.num_row_pixels, self.num_col_pixels), dtype=np.int16)

    
    def get_terrain(self):
        def new_sub_terrain(): 
            return SubTerrain(width=self.num_row_pixels, length=self.num_col_pixels, vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
        
        self.heightfield[:] = random_uniform_terrain(new_sub_terrain(), 
                                                     min_height=self.min_height, 
                                                     max_height=self.max_height, 
                                                     step=self.step, 
                                                     downsampled_scale=self.downsampled_scale).height_field_raw
        
        vertices, triangles = convert_heightfield_to_trimesh(self.heightfield, 
                                                             horizontal_scale=self.horizontal_scale, 
                                                             vertical_scale=self.vertical_scale, 
                                                             slope_threshold=self.slope_threshold)
        
        position = np.array([-self.terrain_width/2.0, -self.terrain_length/2.0, 0])
        add_terrain_to_stage(stage=get_current_stage(), vertices=vertices, triangles=triangles, position=position)

