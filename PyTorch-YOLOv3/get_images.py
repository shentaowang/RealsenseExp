import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import pickle
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, action='store_true', help="if save the video")
    opt = parser.parse_args()
    # init the camerae
    pipeline = rs.pipeline()

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    profile = pipeline.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("深度比例系数为：", depth_scale)

    align = rs.align(rs.stream.color)

    # init the video streaming
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    out_dir = os.path.join("data/exp", time_str)
    if not os.path.exists(out_dir) and opt.save:
        os.makedirs(out_dir)

    cnt = 0

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
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imshow("RGB", color_image)
            cv2.imshow("depth", depth_colormap)
            if opt.save:
                cv2.imwrite(os.path.join(out_dir, "{:0>6d}.jpg".format(cnt)), color_image)
                with open(os.path.join(out_dir, "{:0>6d}.pickle".format(cnt)), 'wb') as fout:
                    pickle.dump(depth_image, fout)

            key = cv2.waitKey(1)
            # Press esc or "q" to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
            cnt += 1
    finally:
        pipeline.stop()

