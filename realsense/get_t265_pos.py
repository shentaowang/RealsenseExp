import pyrealsense2 as rs
import numpy as np


# init the pipeline
pipeline = rs.pipeline()
conf = rs.config()
conf.enable_stream(rs.stream.pose)
profile = pipeline.start(conf)

try:
    while True:
        frames = pipeline.wait_for_frames()
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            if pose.frame_number % 100 == 0:
                print("Frame #{}".format(pose.frame_number))
                print("Position: {}".format(data.translation))
                # print("Velocity: {}".format(data.velocity))
                # print("Acceleration: {}\n".format(data.acceleration))
        else:
            print("lost")
finally:
    pipeline.stop()
