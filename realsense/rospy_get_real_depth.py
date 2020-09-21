import sys
import time
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Imu as msg_Imu
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import inspect
import ctypes
import struct
import tf
import argparse

try:
    from theora_image_transport.msg import Packet as msg_theora
except Exception:
    pass


def pc2_to_xyzrgb(point):
    # Thanks to Panos for his code used in this function.
    x, y, z = point[:3]
    rgb = point[3]

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', rgb)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
    return x, y, z, r, g, b


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

        self.themes = {'depthStream': {'topic': '/camera/depth/image_rect_raw',
                                       'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'colorStream': {'topic': '/camera/color/image_raw',
                                       'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'alignedDepthInfra1': {'topic': '/camera/aligned_depth_to_infra1/image_raw',
                                              'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'alignedDepthColor': {'topic': '/camera/aligned_depth_to_color/image_raw',
                                             'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'static_tf': {'topic': '/camera/color/image_raw',
                                     'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'accelStream': {'topic': '/camera/accel/sample',
                                       'callback': self.imuCallback, 'msg_type': msg_Imu},
                       }

    def imuCallback(self, theme_name):
        def _imuCallback(data):
            if self.listener is None:
                self.listener = tf.TransformListener()
            self.prev_time = time.time()
            self.func_data[theme_name].setdefault('value', [])
            self.func_data[theme_name].setdefault('ros_value', [])
            try:
                frame_id = data.header.frame_id
                value = data.linear_acceleration
                (trans, rot) = self.listener.lookupTransform('/camera_link', frame_id, rospy.Time(0))
                quat = tf.transformations.quaternion_matrix(rot)
                point = np.matrix([value.x, value.y, value.z, 1], dtype='float32')
                point.resize((4, 1))
                rotated = quat * point
                rotated.resize(1, 4)
                rotated = np.array(rotated)[0][:3]
            except Exception as e:
                print(e)
                return
            self.func_data[theme_name]['value'].append(value)
            self.func_data[theme_name]['ros_value'].append(rotated)

        return _imuCallback

    def imageColorCallback(self, theme_name):
        def _imageColorCallback(data):
            self.prev_time = time.time()
            self.func_data[theme_name].setdefault('avg', [])
            self.func_data[theme_name].setdefault('ok_percent', [])
            self.func_data[theme_name].setdefault('num_channels', [])
            self.func_data[theme_name].setdefault('shape', [])
            self.func_data[theme_name].setdefault('reported_size', [])

            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            except CvBridgeError as e:
                print(e)
                return
            channels = cv_image.shape[2] if len(cv_image.shape) > 2 else 1
            pyimg = np.asarray(cv_image)

            ok_number = (pyimg != 0).sum()

            self.func_data[theme_name]['avg'].append(pyimg.sum() / ok_number)
            self.func_data[theme_name]['ok_percent'].append(float(ok_number) / (pyimg.shape[0] * pyimg.shape[1]) / channels)
            self.func_data[theme_name]['num_channels'].append(channels)
            self.func_data[theme_name]['shape'].append(cv_image.shape)
            self.func_data[theme_name]['reported_size'].append((data.width, data.height, data.step))
        return _imageColorCallback

    def wait_for_messages(self, themes):
        # tests_params = {<name>: {'callback', 'topic', 'msg_type', 'internal_params'}}
        self.func_data = dict([[theme_name, {}] for theme_name in themes])

        print('connect to ROS with name: %s' % self.node_name)
        rospy.init_node(self.node_name, anonymous=True)
        for theme_name in themes:
            theme = self.themes[theme_name]
            rospy.loginfo('Subscribing %s on topic: %s' % (theme_name, theme['topic']))
            self.func_data[theme_name]['sub'] = rospy.Subscriber(theme['topic'], theme['msg_type'],
                                                                 theme['callback'](theme_name))

        self.prev_time = time.time()
        break_timeout = False
        while not any([rospy.core.is_shutdown(), break_timeout]):
            rospy.rostime.wallsleep(0.5)
            if self.timeout > 0 and time.time() - self.prev_time > self.timeout:
                break_timeout = True
                self.unregister_all(self.func_data)
        return self.func_data

    @staticmethod
    def unregister_all(registers):
        for test_name in registers:
            rospy.loginfo('Un-Subscribing test %s' % test_name)
            registers[test_name]['sub'].unregister()

    def callback(self, data):
        msg_time = data.header.stamp.secs + 1e-9 * data.header.stamp.nsecs

        if (self.prev_msg_time > msg_time):
            rospy.loginfo('Out of order: %.9f > %.9f' % (self.prev_msg_time, msg_time))
        if type(data) == msg_Imu:
            col_w = 20
            frame_number = data.header.seq
            accel = data.linear_acceleration
            gyro = data.angular_velocity
            line = ('\n{:<%d}{:<%d.6f}{:<%d.4f}{:<%d.4f}{:<%d.4f}{:<%d.4f}{:<%d.4f}{:<%d.4f}' % (
            col_w, col_w, col_w, col_w, col_w, col_w, col_w, col_w)).format(frame_number, msg_time, accel.x, accel.y,
                                                                            accel.z, gyro.x, gyro.y, gyro.z)
            sys.stdout.write(line)
        self.prev_msg_time = msg_time
        self.prev_msg_data = data
        self.prev_time = time.time()
        if any([self.seq > 0 and data.header.seq >= self.seq]):
            self.result = data
            self.sub.unregister()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wanted_topic", type=str, default="colorStream", help="the topic to listen")
    parser.add_argument("--seq", type=str, default="", help="the sequential number for device")
    parser.add_argument("--timeout", type=float, default=1, help="timeout value")
    opt = parser.parse_args()

    msg_retriever = CWaitForMessage(opt)
    themes = [opt.wanted_topic]
    res = msg_retriever.wait_for_messages(themes)
    print(res)
