import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()

cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("深度比例系数为：", depth_scale)
# 深度比例系数为： 0.0010000000474974513
# 测试了数个摄像头，发现深度比例系数都相同，甚至D435i的也一样。

align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = depth_image * depth_scale * 1000

        width = aligned_depth_frame.get_width()
        height = aligned_depth_frame.get_height()

        dist_to_center = aligned_depth_frame.get_distance(int(width / 2), int(height / 2))
        print(dist_to_center)

finally:
    pipeline.stop()
