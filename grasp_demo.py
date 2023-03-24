import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from song.realsenseD435 import RealsenseD535
from song.camera_date import CameraData
from PIL import Image

import logging
logging.getLogger().setLevel(logging.INFO)


class Grasp():
    
    def __init__(self, saved_model_path=None):
        if saved_model_path==None:
            self.model_path = './trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_34_iou_0.91'
        self.camera = RealsenseD535()
        self.model = None
        self.device = None
        self.cam_data = CameraData()
        
        # 连接相机，获取相机标定数据 (data from calibrate camera)
        self.camera.init_cam()
        self.camera_pose = np.loadtxt('song/cam_pose/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('song/cam_pose/camera_depth_scale.txt', delimiter=' ')
        logging.info('camera_pose:\n{}'.format(self.camera_pose))
        logging.info('camera_depth_scale:{}'.format(self.cam_depth_scale))
        
        homedir = os.path.join(os.path.abspath('.'), 'grasp_commits')
        self.grasp_request = os.path.join(homedir, 'grasp_request.npy')
        self.grasp_available = os.path.join(homedir, 'grasp_available.npy')
        self.grasp_pose = os.path.join(homedir, 'grasp_pose.npy')
        logging.info('homedir:  {}'.format(homedir))
        
        
    # load trained model
    def load_model(self):
        logging.info('Loading model...')
        self.model = torch.load(self.model_path)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logging.info('Load model from : {}'.format(self.model_path))
        logging.info('select device : {}'.format(self.device))
        
        
    def generate(self):''''''
        
        
        
        
        
        
if __name__ == '__main__':
    g = Grasp()
    g.load_model()
    # color_image, depth_image = g.camera.get_data()
    # logging.info('color_image.shape:{}'.format(color_image.shape))
    # logging.info('depth_image.shape:{}'.format(depth_image.shape))
    # color_image = g.cam_data.get_rgb(color_image)
    # depth_image = g.cam_data.get_depth(depth_image)
    # logging.info('color_image.shape:{}'.format(color_image.shape))
    # # logging.info('depth_image.shape:{}'.format(depth_image.shape))
    # x, depth_img, color_img = g.cam_data.get_data(color_image, depth_image)
    # logging.info('x.shape: {}'.format(x.shape))
    # logging.info('depth_img.shape: {}'.format(depth_img.shape))
    # logging.info('color_img.shape: {}'.format(color_img.shape))
    
    
    
    