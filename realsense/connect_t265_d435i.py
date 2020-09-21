import pyrealsense2 as rs
import numpy as np
import os

# cannot work

ctx = rs.context()
devices = ctx.query_devices()
align = rs.align(rs.stream.color)
depth_scale = None
pipelines = {}
for dev in devices:
    # get the name
    dev_name = dev.get_info(rs.camera_info.name)
    if "D435I" in dev_name:
        pipeline = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        print(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(cfg)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        pipelines["D435I"] = pipeline
    elif "T265" in dev_name:
        pipeline = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        print(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.pose)
        profile = pipeline.start(cfg)
        pipelines["T265"] = pipeline
    else:
        print("detect extra camera")

try:
    while True:
        # D453i info
        frames = pipelines["D435I"].wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        # get the distance to image center
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = depth_image * depth_scale * 1000
        width = aligned_depth_frame.get_width()
        height = aligned_depth_frame.get_height()
        dist_to_center = aligned_depth_frame.get_distance(int(width / 2), int(height / 2))

        # T265 info
        frames = pipelines["T265"].wait_for_frames()
        pose = frames.get_pose_frame()

        if pose and dist_to_center:
            data = pose.get_pose_data()
            if pose.frame_number % 100 == 0:
                print("Frame #{}".format(pose.frame_number))
                print("Position: {}".format(data.translation))
                print("Distance: {}".format(dist_to_center))
finally:
    for name in pipelines:
        pipelines[name].stop()

