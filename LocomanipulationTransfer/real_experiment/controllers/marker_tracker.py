import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
sys.path.append(os.path.join(root_dir, "controllers"))
from controller import Controller
from realsense_controller import RealsenseRGBController

import cv2

class MarkerTracker(Controller):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        self.marker_len = cfg.get('marker_length')

        
