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
        # x->pitch, y->yaw, z->roll
        self.theta = {}
        self.theta["pitch"], self.theta["yaw"], self.theta["roll"] = None, None, None

    def rotation_estimator(self, pos_data):
        w = pos_data.pose.pose.orientation.w
        x = pos_data.pose.pose.orientation.x
        y = pos_data.pose.pose.orientation.y
        z = pos_data.pose.pose.orientation.z
        self.theta["pitch"] = -np.arcsin(2.0 * (x*z - w*y)) * 180.0 / np.pi
        self.theta["roll"] = np.arctan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) * 180.0 / np.pi
        self.theta["yaw"] = np.arctan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / np.pi
        return

    def callback(self, img_data, depth_data, gyro_data, accel_data, pos_data):
        self.rotation_estimator(pos_data)
        # print(pos_data)
        print(self.theta)
        cv_color_img = self.bridge.imgmsg_to_cv2(img_data, img_data.encoding)
        cv2.imshow("RGB", cv2.cvtColor(cv_color_img, cv2.COLOR_RGB2BGR))
        cv_depth_img = self.bridge.imgmsg_to_cv2(depth_data, depth_data.encoding)
        cv2.imshow("depth", cv_depth_img)
        key = cv2.waitKey(1)
        # Press esc or "q" to close the image window
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            os._exit(0)

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
