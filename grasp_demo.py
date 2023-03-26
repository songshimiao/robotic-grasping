import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from song.realsenseD435 import RealsenseD535
import pyrealsense2 as rs
from song.camera_date import CameraData
import song.grasp
from PIL import Image
from skimage.filters import gaussian

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
        
        
    def generate(self):
        # get RGB-D image from camera (H x W x C)
        color_image, depth_image = self.camera.get_data()
        
        # get x (B x C x H x W) -> net
        x, depth_img, color_img = self.cam_data.get_data(color_image, depth_image)
        logging.info('color_image.shape:{}'.format(color_image.shape))
        logging.info('depth_image.shape:{}'.format(depth_image.shape))
        logging.info('x.shape: {}'.format(x.shape))
        logging.info('depth_img.shape: {}'.format(depth_img.shape))
        logging.info('color_img.shape: {}'.format(color_img.shape))
        
        # predict the grasp pse using the trained model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)
            
        logging.info('pred.key : {}'.format(pred.keys()))
            
        q_img, angle_img, width_img = post_process_output(
            pred['pos'], pred['cos'],
            pred['sin'], pred['width'])
        grasps = song.grasp.detect_grasps(q_img, angle_img, width_img)
        
        # Get grasp position from model output
        pos_z = depth_image[grasps[0].center[0] + self.cam_data.top_left[0],
                            grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)
        
        if pos_z == 0:
            return
        
        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        logging.info('target : {}'.format(target))
        
        
            
        
        
def post_process_output(q_img, cos_img, sin_img, width_img):
    q_img = q_img.cpu().numpy().squeeze()
    angle_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0
    
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    angle_img = gaussian(angle_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    
    return q_img, angle_img, width_img    
        
        
        
if __name__ == '__main__':
    g = Grasp()
    g.load_model()
    g.generate()
    # color_image, depth_image = g.camera.get_data()
    # logging.info('color_image.shape:{}'.format(color_image.shape))
    # logging.info('depth_image.shape:{}'.format(depth_image.shape))
    # color_image = g.cam_data.get_rgb(color_image)
    # depth_image = g.cam_data.get_depth(depth_image)
    # logging.info('color_image.shape:{}'.format(color_image.shape))
    # # logging.info('depth_image.shape:{}'.format(depth_image.shape))
    # x, depth_img, color_img = g.cam_data.get_data(color_image, depth_image)
    
    
    
    
    