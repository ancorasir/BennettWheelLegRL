# !/usr/bin/python
# coding=utf-8
import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
import numpy as np
import pyrealsense2 as rs
from controller import Controller

class RealsenseRGBController(Controller):
    def __init__(self, width, height, fps=60):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.align = None

    def initialize(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.logInfo("[RealsenseRGBController] Camera Initialized. ")

    def retrieveInfo(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(aligned_color_frame.get_data())

        return color_image

    def get_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color)
        return color_stream.as_video_stream_profile().get_intrinsics()

    def close(self):
        self.pipeline.stop()

if __name__ == "__main__":
    realsense = RealsenseRGBController(848, 480)
    realsense.initialize()
    import cv2
    key = cv2.waitKey(1)
    while key != ord('q'):
        cv2.imshow("Realsense Test", realsense.retrieveInfo())
        key = cv2.waitKey(1)
    realsense.close()