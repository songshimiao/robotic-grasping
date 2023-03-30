import argparse
import logging

import cv2
import numpy as np
import torch.utils.data
import torch

from song.realsenseD435 import RealsenseD435
from skimage.filters import gaussian
from song.camera_date import CameraData
from song.grasp import detect_grasps

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_34_iou_0.91',
                        help='Path to saved network to evaluate')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    args = parser.parse_args()
    return args


def post_process_output(q_img, cos_img, sin_img, width_img):
    q_img = q_img.cpu().numpy().squeeze()
    angle_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0
    
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    angle_img = gaussian(angle_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    
    return q_img, angle_img, width_img

def plot_results(rgb, depth, q_img, angle_img, width_img, no_grasps=1):
    gs = detect_grasps(q_img, angle_img, width_img, no_grasps)
    
    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Grasp', cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow('Depth', depth)
    cv2.imshow('RGB', rgb)
    for g in gs:
        cv2.imshow('RGB', g)
    

if __name__ == '__main__':
    
    args = parse_args()
    
    # Connect to Camera
    logging.info('Connecting to camera...')
    camera = RealsenseD435()
    camera.init_cam()
    camera_data = CameraData()
    
    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.info('device : {}'.format(device))
    
    while(True):
        # camera.plot_image()
        color_image, depth_image = camera.get_data()
        x, depth_img, color_img = camera_data.get_data(color_image, depth_image)
        with torch.no_grad():
            xc = x.to(device)
            pred = net.predict(xc)
            q_img, angle_img, width_img = post_process_output(
                pred['pos'],
                pred['cos'],
                pred['sin'],
                pred['width']
            )
            
            plot_results(color_image, depth_image,
                         q_img, angle_img, width_img)
            
            
        key = cv2.waitKey(1)
        if key == 27:
            break