import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from realsenseD435 import RealsenseD535
from PIL import Image

import logging
logging.getLogger().setLevel(logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default='../trained-models/epoch_34_iou_91')
cfgs = parser.parse_args()

# class Grasp():
    
#     def __init__(self):