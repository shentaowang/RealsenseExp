import sys
import time
import os
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import Imu as msg_Imu
from nav_msgs.msg import Odometry as msg_Odometry
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import argparse
import message_filters
import cv2
import matplotlib.pyplot as plt


class CWaitForMessage:
    def __init__(self, opt):
        self.result = None
        self.timeout = opt.timeout
        self.seq = opt.seq
        self.node_name = "rs2_listener"
        self.bridge = CvBridge()
        self.listener = None
        self.prev_msg_time = 0
        self.fout = None
        self.func_data = dict()

    def callback(self, img_data, depth_data, gyro_data, accel_data, pos_data):
        cv_color_img = self.bridge.imgmsg_to_cv2(img_data, img_data.encoding)
        cv2.imshow("RGB", cv2.cvtColor(cv_color_img, cv2.COLOR_RGB2BGR))
        cv_depth_img = self.bridge.imgmsg_to_cv2(depth_data, depth_data.encoding)
        cv2.imshow("depth", cv_depth_img)
        key = cv2.waitKey(1)
        # Press esc or "q" to close the image window
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            os._exit(0)
        print("gryo angular_velocity")
        print(gyro_data.angular_velocity)
        print("accel linear_acceleration")
        print(accel_data.linear_acceleration)
        print("pos")
        print(pos_data.pose)

    def wait_for_messages(self):
        print("connect to ROS with name: %s" % self.node_name)
        rospy.init_node(self.node_name, anonymous=True)
        img_sub = message_filters.Subscriber("/d400/color/image_raw", msg_Image)
        depth_sub = message_filters.Subscriber("/d400/aligned_depth_to_color/image_raw", msg_Image)
        gyro_sub = message_filters.Subscriber("/d400/gyro/sample", msg_Imu)
        accel_sub = message_filters.Subscriber("/d400/accel/sample", msg_Imu)
        pos_sub = message_filters.Subscriber("/t265/odom/sample", msg_Odometry)
        # topic synchronization by time
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, depth_sub, gyro_sub, accel_sub, pos_sub],
                                                         queue_size=10, slop=0.5, allow_headerless=True)
        ts.registerCallback(self.callback)
        rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wanted_topic", type=str, default="colorStream", help="the topic to listen")
    parser.add_argument("--seq", type=str, default="", help="the sequential number for device")
    parser.add_argument("--timeout", type=float, default=1, help="timeout value")
    opt = parser.parse_args()

    msg_retriever = CWaitForMessage(opt)
    msg_retriever.wait_for_messages()
