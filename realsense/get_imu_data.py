import pyrealsense2 as rs
import numpy as np


# init the pipeline
pipeline = rs.pipeline()
conf = rs.config()
conf.enable_stream(rs.stream.accel)
conf.enable_stream(rs.stream.gyro)
profile = pipeline.start(conf)

try:
    while True:
        frames = pipeline.wait_for_frames()
        accel = frames[0].as_motion_frame().get_motion_data()
        accel = np.array([accel.x, accel.y, accel.z])
        gyro = frames[1].as_motion_frame().get_motion_data()
        gyro = np.array([gyro.x, gyro.y, gyro.z])
        print("accel", accel)
        print("gyro", gyro)
finally:
    pipeline.stop()
