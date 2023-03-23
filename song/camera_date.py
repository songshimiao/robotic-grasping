import numpy as np
import torch
from data_processing import Image

class CameraData:
    def __init__(self, width=640, height=480, output_size=300):
        self.output_size = output_size
        
        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2
        
        self.top_left = (top, left)
        self.bottom_right = (bottom, right)
        
        
    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, axis=0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def get_depth(self, img):
        depth_img = Image(img)
        depth_img.crop(self.bottom_right, self.top_left)
        depth_img.normalise()
        depth_img.img = depth_img.img.transpose((2, 0, 1))
        return depth_img.img
    
    def get_rgb(self, img, norm=True):
        rgb_img = Image(img)
        rgb_img.crop(self.bottom_right, self.top_left)
        if norm:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    def get_data(self, rgb=None, depth=None):
        rgb_img = self.get_rgb(img=rgb)
        depth_img = self.get_depth(img=depth)
        x = self.numpy_to_torch(
            np.concatenate(
                (np.expand_dims(depth_img, axis=0),
                 np.expand_dims(rgb_img, axis=0)),
                1))
        return x, depth_img, rgb_img