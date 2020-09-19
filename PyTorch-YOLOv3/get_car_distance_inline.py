from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import pyrealsense2 as rs
import numpy as np
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

    # init the model
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    # init the camerae
    pipeline = rs.pipeline()

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("深度比例系数为：", depth_scale)

    align = rs.align(rs.stream.color)

    # prepare for out
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    fout = open(os.path.join("data/exp", time_str + ".txt"), "w")

    # start detect
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

            RGBimg = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = Variable(imgTensor.type(Tensor))
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)

            with torch.no_grad():
                detections = model(imgTensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
            if detections is None:
                continue
            detections = rescale_boxes(detections, opt.img_size, RGBimg.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            img = color_image.copy()
            detections = [i for i in detections if i[6] == 39]
            if len(detections) != 0:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    # print(cls_conf)
                    img = cv2.rectangle(img, (x1, y1 + box_h), (x2, y1), color, 2)
                    cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(img, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 2)
                    print("dist: {}".format(depth_image[int(y1+box_h//2), int(x1+box_w//2)]))
                    fout.write("{:.2f}\n".format(depth_image[int(y1+box_h//2), int(x1+box_w//2)]))
            else:
                print("lost")
                fout.write("{}\n".format(-1))
            cv2.imshow("RGB", img)
            cv2.imshow("depth", depth_colormap)
            key = cv2.waitKey(1)
            # Press esc or "q" to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()



