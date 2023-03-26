import sys
import os
import distutils.core
import torch
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import json
#!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import random
import cv2

setup_logger()

#load dataset
annotations_path = "/export/home/group05/annotations/instances_val2017.json"
database_path = "/export/home/group05/val2017/"
coco_annotation=COCO(annotations_path)


def load_image_and_annotations(img_filename, database_path):
    img = cv2.imread(database_path+img_filename)
    img_id = int(img_filename.split('.')[0])
    info = coco_annotation.loadImgs([img_id])[0]
    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id])
    anns = coco_annotation.loadAnns(ann_ids)
    masks = [coco_annotation.annToMask(ann) for ann in anns]

    print("Random image annotations" ,anns)
    return img, info, masks


def pad_images(imgs, new_image_height, new_image_width):
    
    color = (0, 0, 0)
    result_imgs = []
    
    for img in imgs:
        if len(img.shape) == 3:
            old_image_height, old_image_width, channels = img.shape

            result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        elif len(img.shape) == 2:
            old_image_height, old_image_width = img.shape
            result = np.full((new_image_height, new_image_width), 0, dtype=np.uint8)

        print("NEW ", new_image_height, "W ", new_image_width)
        print("OLD ", old_image_height, "W ", old_image_width)

        # compute center offset
        x_center = round((new_image_width - old_image_width) / 2)
        y_center = round((new_image_height - old_image_height) / 2)

        if old_image_height > new_image_height:
            img = img[round((old_image_height - new_image_height) / 2):-round((old_image_height - new_image_height) / 2), :]
            old_image_height = new_image_height
        if old_image_width > new_image_width:
            img = img[:, round((old_image_width - new_image_width) / 2):-round((old_image_width - new_image_width) / 2)]
            old_image_width = new_image_width

        print("OLD AFTER CROP ", img.shape[0], "W ", img.shape[1])

        # copy img image into center of result image
        result[y_center:y_center+old_image_height,             x_center:x_center+old_image_width] = img
        
        result_imgs.append(result)
        
    return result_imgs



def transplant_images(image_1, image_2, image_2_mask):
    image_1_copy = image_1.copy() 
    image_1_copy[np.where(image_2_mask == 1)] = image_2[np.where(image_2_mask == 1)]
    rows, cols = np.where(image_2_mask == 1)
    image_1_copy[rows, cols, :] = image_2[rows, cols, :]
    return image_1_copy


def apply_transformation(img_list, tx, ty, angle):
    warped_img_list = []
    for img in img_list:
        print(img.shape)  # add this line to check the shape of img
        M = cv2.getRotationMatrix2D((ty, tx), angle, 1)
        warped_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        warped_img_list.append(warped_img)
    print('warped_img_list', len(warped_img_list))
    return warped_img_list


def predict(predictor, img):
    outputs = predictor(img)
    output_predictions = outputs["instances"].pred_classes
    print(output_predictions)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image= out.get_image()
    return output_image, output_predictions

cfg = get_cfg()
model = 'mask_rcnn'
if model == 'mask_rcnn':
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
elif model == 'fast_rcnn':
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

predictor = DefaultPredictor(cfg)


#load main image, annotations and mask
img1_filename = random.choice(os.listdir(database_path)) #pick randomly or select a filename
img1, info_img1, mask_img1 = load_image_and_annotations(img1_filename, database_path)
img1_h, img1_w, img1_chann = img1.shape

img2 = img1.copy()

plt.figure()
plt.imshow(img2[:,:,::-1])
plt.savefig('./outputs_task_c/img1.jpg',transparent=False)
plt.close()

plt.figure()
plt.imshow(mask_img1[0])
plt.savefig('./outputs_task_c/mask_img1.jpg',transparent=False)
plt.close()

print("Target image: ",img1_filename)


for i in range(4):
    tx = random.randint(-400, 400) # generate a random translation in the x direction
    ty = random.randint(-400, 400) # generate a random translation in the y direction
    rotation_angle_deg = random.randint(-10, 10) # generate a random rotation angle
    new_image_height, new_image_width = img1_h, img1_w
    
    # apply the transformation to the image
    warped_image = apply_transformation([img2], tx, ty, rotation_angle_deg)  
    warped_image = pad_images(warped_image, img1_h, img1_w)
    warped_image = np.array(warped_image)[0]
    
    img2_h, img2_w = warped_image.shape[:2]

    # apply the transformation to the mask
    warped_mask_list = apply_transformation([mask_img1[0]], tx, ty, rotation_angle_deg)  
    warped_mask = pad_images(warped_mask_list, img1_h, img1_w)
    warped_mask = np.array(warped_mask)[0]

    # display the warped image and mask
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(warped_image[:, :, ::-1])
    plt.title("Warped Image")
    plt.subplot(1, 2, 2)
    plt.imshow(warped_mask, cmap='gray')
    plt.title("Warped Mask")
    plt.savefig(f'./outputs_task_c/final_{i+1}.jpg',transparent=False)

    
    x1, y1 = random.randint(0, img1_h - img2_h), random.randint(0, img1_w - img2_w)
    x2, y2 = x1 + img2_h, y1 + img2_w
            
    img_copy = np.copy(warped_image)
    mask_copy = np.copy(warped_mask)

    img2[x1:x2, y1:y2] = img_copy
    mask_img1[0][x1:x2, y1:y2] = mask_copy
            
    img1_copy = np.copy(img2)
    mask_img1_copy = np.copy(mask_img1[0])

    # transplant the object to a random location in the original image
    new_image = transplant_images(img1, img1_copy, mask_img1_copy)
        
    # Save the image and mask after applying 4 random locations
    plt.figure()
    plt.imshow(new_image[:,:,::-1])
    plt.savefig(f'./outputs_task_c/final_transplant{i+1}.jpg',transparent=False)
    plt.close()


    output_image_1, _ = predict(predictor, img2)
    plt.figure()
    plt.imshow(output_image_1)
    plt.savefig(f'./outputs_task_c/output_image_1{i+1}.jpg',transparent=False)
    plt.close()

    output_new_image, _ = predict(predictor,new_image)
    plt.figure()
    plt.imshow(output_new_image)
    plt.savefig(f'./outputs_task_c/output_new_image{i+1}.jpg',transparent=False)
    plt.close()