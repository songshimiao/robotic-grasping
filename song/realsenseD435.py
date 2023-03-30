import logging
import time
import cv2
import numpy as np
import pyrealsense2 as rs

logging.getLogger().setLevel(logging.INFO)


class RealsenseD435(object):
    
    def __init__(self):
        self.image_height = 480
        self.image_width = 640
        self.intrinsics = None
        self.pipeline = None
        self.config = None
        self.scale = 0.001
        
        
    def init_cam(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth,
                                  self.image_width,
                                  self.image_height,
                                  rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color,
                                  self.image_width,
                                  self.image_height,
                                  rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)
        rgb_profile = profile.get_stream(rs.stream.color)
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx,
                                    0, raw_intrinsics.fy, raw_intrinsics.ppy,
                                    0, 0, 1]).reshape(3, 3)
        # self.intrinsics = np.array([607.879, 0, 325.14,
        #                             0, 607.348, 244.014,
        #                             0, 0, 1]).reshape(3, 3)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        logging.info('Camera depth scale: {}'.format(self.scale))
        logging.info('Camera intrinsics:\n{}'.format(self.intrinsics))
        logging.info('D435 connected ...')
        
        
    def get_intrinsics(self) -> np.array:
        return self.intrinsics
    
    
    def get_depth_scale(self) -> float:
        return self.scale
    
    
    def get_data(self):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # aligned_depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
        
        # depth_image 的维度转换: (480 x 640) -> (480 x 640 x 1)
        depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        
        return color_image, depth_image
    
    
    def plot_image(self):
        color_image, depth_image = self.get_data()
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        cv2.namedWindow('D435_color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('D435_color', color_image)
        cv2.namedWindow('D435_depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('D435_depth', depth_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return True
        else:
            return False
        
        
    def plot_image_stream(self):
        while(True):
            if self.plot_image():
                break
            time.sleep(0.1)
        
        
        
if __name__ == '__main__':
    camera = RealsenseD435()
    camera.init_cam()
    color_img, depth_img = camera.get_data()
    logging.info('color_img.shape:{}'.format(color_img.shape))
    logging.info('depth_img.shape:{}'.format(depth_img.shape))
    logging.info('depth_image.squeeze.shape:{}'.format(depth_img.squeeze(axis=2).shape))
    while(True):
        camera.plot_image()
        time.sleep(0.1)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        
            
