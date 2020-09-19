import pyrealsense2 as rs
import numpy as np
import argparse
import os
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=True, action='store_true', help="if save the pos")
    parser.add_argument("--out_dir", default="data/t265_pos", help="dir to save the data")
    opt = parser.parse_args()

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    if not os.path.exists(opt.out_dir) and opt.save:
        os.makedirs(opt.out_dir)
    if opt.save:
        fout = open(os.path.join(opt.out_dir, time_str + ".txt"), "w")

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
                if opt.save:
                    save_time = time.time()
                    pos = data.translation
                    fout.write("{:.4f}\tx\t{:.4f}\ty\t{:.4f}\tz\t{:.4f}\n".format(save_time, pos.x, pos.y, pos.z))
                if pose.frame_number % 100 == 0:
                    print("Frame #{}".format(pose.frame_number))
                    print("Position: {}".format(data.translation))
            else:
                print("lost")
    finally:
        pipeline.stop()
