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
import shutil
from tqdm import tqdm
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs
import random

def form_person_dataset(person_image_list):
    DATA_DIR = '/home/lc/models-master/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012'
    EXP_DATA_DIR = '/home/lc/models-master/research/deeplab/datasets/person_pascal_voc_seg/personVOCdevkit/personVOC2012'
    if not os.path.exists(EXP_DATA_DIR):
        os.makedirs(EXP_DATA_DIR)
    IMG_DIR = '/JPEGImages'
    if not os.path.exists(EXP_DATA_DIR + IMG_DIR):
        os.makedirs(EXP_DATA_DIR + IMG_DIR)
    IMS_DIR = '/ImageSets'
    if not os.path.exists(EXP_DATA_DIR + IMS_DIR):
        os.makedirs(EXP_DATA_DIR + IMS_DIR)
    LIST_DIR = '/Segmentation'
    if not os.path.exists(EXP_DATA_DIR + IMS_DIR + LIST_DIR):
        os.makedirs(EXP_DATA_DIR + IMS_DIR + LIST_DIR)
        trainval_num = len(person_image_list)
        train_num = trainval_num * 2 / 3
        val_num = trainval_num / 3
        train_list = []
        val_list = []
        for i in person_image_list:
            total_list.append(i)
        ftr = codecs.open(EXP_DATA_DIR+IMS_DIR+LIST_DIR+"/train.txt",'w','utf-8')
        fvl = codecs.open(EXP_DATA_DIR+IMS_DIR+LIST_DIR+"/val.txt",'w','utf-8')
        random.shuffle(total_list)
        for i in range(train_num):
            train_list.append(total_list.pop())
            ftr.write(train_list[i]+'\n')
        for i in total_list:
            val_list.append(total_list.pop())
            fvl.write(val_list[i]+'\n')
        ftr.close()
        fvl.close()
    SEG_DIR = '/SegmentationClass'
    if not os.path.exists(EXP_DATA_DIR + SEG_DIR):
        os.makedirs(EXP_DATA_DIR + SEG_DIR)
    SEG_RAW_DIR = '/SegmentationClassRaw'
    if not os.path.exists(EXP_DATA_DIR + SEG_RAW_DIR):
        os.makedirs(EXP_DATA_DIR + SEG_RAW_DIR)
    SEG_OBJ_DIR = '/SegmentationObject'
    if not os.path.exists(EXP_DATA_DIR + SEG_OBJ_DIR):
        os.makedirs(EXP_DATA_DIR + SEG_OBJ_DIR)
    ANNO_DIR = '/Annotations'
    if not os.path.exists(EXP_DATA_DIR + ANNO_DIR):
        os.makedirs(EXP_DATA_DIR + ANNO_DIR)
    for img in person_image_list:
        img_name = img.split('.')[0]
        voc_file = DATA_DIR + IMG_DIR + '/' + img
        file = EXP_DATA_DIR + IMG_DIR + '/' + img
        shutil.copyfile(voc_file, file)
        voc_seg = DATA_DIR + SEG_DIR + '/' + img_name + '.png'
        seg = EXP_DATA_DIR + SEG_DIR + '/' + img_name + '.png'
        shutil.copyfile(voc_seg, seg)
        voc_seg_raw = DATA_DIR + SEG_RAW_DIR + '/' + img_name + '.png'
        seg_raw = EXP_DATA_DIR + SEG_RAW_DIR + '/' + img_name + '.png'
        shutil.copyfile(voc_seg_raw, seg_raw)
        voc_seg_obj = DATA_DIR + SEG_OBJ_DIR + '/' + img_name + '.png'
        seg_obj = EXP_DATA_DIR + SEG_OBJ_DIR + '/' + img_name + '.png'
        shutil.copyfile(voc_seg_obj, seg_obj)
        voc_anno = DATA_DIR + ANNO_DIR + '/' + img_name + '.xml'
        anno = EXP_DATA_DIR + ANNO_DIR + '/' + img_name + '.xml'
        shutil.copyfile(voc_anno, anno)

def get_person_image_list(annotation_dir):
    trainval_list = tuple(open('/home/lc/models-master/research/deeplab/trainval_list.txt', 'r'))
    trainval_list = [_id.rstrip() for _id in trainval_list]
    person_image_list = []
    for item in tqdm(trainval_list):
        item  = item.split('.')[0] + '.xml'
        xml_file = os.path.join(annotation_dir, item)
        read_xml_file(xml_file, person_image_list)
    # print(person_image_list)
    f = codecs.open("trainval_list.txt",'w','utf-8')
    print("Writing trainval list in trainval_list.txt...")
    for img in person_image_list:
        f.write(img+'\n')
    f.close()
    print("There are %d images of person."%len(person_image_list))
    return person_image_list

def read_xml_file(xml_file, person_image_list):
    DOMTree = xml.dom.minidom.parse(xml_file)
    annotation = DOMTree.documentElement
    objects = annotation.getElementsByTagName("object")
    for object in objects:
        name = object.getElementsByTagName("name")[0]
        if name.childNodes[0].data == "person":
            filename = annotation.getElementsByTagName("filename")[0].childNodes[0].data
            person_image_list.append(filename)
            # print(filename)
            break

ANNOTATION_DIR = '/home/lc/models-master/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/Annotations'
form_person_dataset(get_person_image_list(ANNOTATION_DIR))
