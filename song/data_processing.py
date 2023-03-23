import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import cv2
import numpy as np
from imageio import imread
from skimage.transform import rotate, resize


class Image:
    
    def __init__(self, img):
        self.img = img
            
    def __getattr__(self, attr):
        return getattr(self, attr)
    
    @classmethod
    def from_file(cls, fname):
        return cls(imread(fname))
     
    def copy(self):
        return self.__class__(self.img.copy())
    
    def crop(self, top_left, bottom_right, resize=None):
        '''
        Crop the image to a bounding box 
        - param top_left: tuple, top left pixel.
        - param bottom_right: tuple, bottom right pixel
        - param resize: resize the croped image to this size
        '''
        if resize is not None:
            self.resize(resize)
            
    def cropped(self, *args, **kwargs):
        i = self.copy()
        i.crop(*args, **kwargs)
        return i
    
    def normalise(self):
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()
        
        
    def resize(self, shape):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)
        
    def resized(self, *args, **kwargs):
        i = self.copy()
        i.resize(*args, **kwargs)
        return i
    
    def rotate(self, angle, center=None):
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img,
                          angle / np.pi*180,
                          center=center,
                          mode='symmetric',
                          preserve_range=True).astype(self.img.dtype)
        
    def rotated(self, *args, **kwargs):
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i
    
    