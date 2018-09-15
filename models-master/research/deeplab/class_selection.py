import cv2
import numpy as np
import random
from PIL import Image
import scipy
from multiprocessing import Pool
from functools import partial
import signal
import time
import glob
import sys
import os
from xml.dom.minidom import parse
import xml.dom.minidom

img_list = tuple(open('/home/lc/VOCdevkit/VOC2012/ImageSets/Main/person_train.txt', 'r'))
for item in img_list:
    item = img_list[0].rstrip()
    filename = item.split(' ')[0]
    filename = filename.rstrip()
    pn = item.split(' ')[-1]
