'''
    Author: Ze Ma
    E-mail: maze1234556@sjtu.edu.cn
    Jan. 26, 2020
'''
import argparse

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf

# # import Network
# from network_CPN101 import Network

# # pose estimation utils
# from HPE.dataset import Preprocessing
# from HPE.config import cfg

# import my own utils
import sys, os, time
sys.path.append(os.path.abspath("/project/lighttrack"))
sys.path.append(os.path.abspath("/project/lighttrack/utils"))
from utils_json import *
from visualizer import *
from utils_io_folder import *



def make_video_from_images(img_paths, outvid_path, fps=25, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for ct, img_path in enumerate(img_paths):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        img = imread(img_path)
        if img is None:
            print(img_path)
            continue
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid_path, fourcc, float(fps), size, is_color)

        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    if vid is not None:
        vid.release()
    return vid


if __name__ == '__main__':

    for video_idx in range(74):
        visualize_folder = "video_%02d" %video_idx

        output_video_folder = "./"
        video_name = "video_%02d" %video_idx

        output_video_path = os.path.join(output_video_folder, video_name+".mp4")

        img_paths = get_immediate_childfile_paths(visualize_folder)
        make_video_from_images(img_paths, output_video_path, fps=10, size=None, is_color=True, format="XVID")

        print("Finished video {}".format(output_video_path))
