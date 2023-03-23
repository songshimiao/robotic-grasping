import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from realsenseD435 import RealsenseD535
from camera_date import CameraData
from PIL import Image

import logging
logging.getLogger().setLevel(logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../trained-models/epoch_34_iou_91')
cfgs = parser.parse_args()

class Grasp():
    
    def __init__(self):
        self.model_path = cfgs.model_path
        self.camera = RealsenseD535()
        self.model = None
        self.device = None
        self.cam_data = CameraData()
        
        
        
        
if __name__ == '__main__':
    g = Grasp()
    g.camera.init_cam()
    color_image, depth_image = g.camera.get_data()
    logging.info('color_image.shape:{}'.format(color_image.shape))
    logging.info('depth_image.shape:{}'.format(depth_image.shape))
    # color_image = g.cam_data.get_rgb(color_image)
    # depth_image = g.cam_data.get_depth(depth_image)
    # logging.info('color_image.shape:{}'.format(color_image.shape))
    # logging.info('depth_image.shape:{}'.format(depth_image.shape))
    x, depth_img, color_img = g.cam_data.get_data(color_image, depth_image)
    logging.info('x.shape: {}'.format(x.shape))
    logging.info('depth_img.shape: {}'.format(depth_img.shape))
    logging.info('color_img.shape: {}'.format(color_img.shape))
    
    
    
    