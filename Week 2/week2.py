import sys
import os
import distutils.core
import torch
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import glob
import argparse

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
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import PIL.Image as Image
import sys
import pycocotools
from skimage import measure
import copy
from torchvision.ops import masks_to_boxes

setup_logger()

DATASET_PATH = '/home/mcv/datasets/KITTI-MOTS/'


def parse_kitti_mots(train=True):
    # In the KITTI-MOTS paper sequences 2, 6, 7, 8, 10, 13, 14, 16 and 18 were chosen for the validation set, the remaining sequences for the training set.
    val_sequences = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    train_sequences = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
    
    dataset_paths = sorted(glob.glob(DATASET_PATH+'instances_txt/*.txt'))
    dataset_paths_parsed = []
    if train:
        for index in sorted(val_sequences, reverse=True):
            del dataset_paths[index]
    else:
        for index in sorted(train_sequences, reverse=True):
            del dataset_paths[index]    
    
    # Parse of KITTI-MOTS 
    # http://carina.cse.lehigh.edu/MaskTrackRCNN-Lihao/dataFormat.html
    dataset_dicts = []
    for file in dataset_paths:         
        index = 0
        prev_frame = 0
        curr_frame = 0
        
        with open(file, 'r') as f:
            for line in f:
                curr_frame, id, class_id, im_height, im_width, rle = line.split()
                
                # If object id is 10000 it is an ignore region, so we do not take it into account
                if id  == '10000':
                    continue
                
                if prev_frame != curr_frame:
                    # Save the previous record and create a new one
                    if 'record' in locals():
                        dataset_dicts.append(record)
                    record = {}
                    record['file_name'] = DATASET_PATH+'training/image_02/%s/%s.png'%(file.split('/')[-1].replace('.txt', ''), str(curr_frame.zfill(6)))
                    record['image_id'] = '%s_%s'%(file.split('/')[-1].replace('.txt', ''), str(curr_frame.zfill(6)))
                    record['height'] = int(im_height)
                    record['width'] = int(im_width)
                    record['annotations'] = []

                class_id = int(class_id)
                category_map = {0:1, 1: 2, 2: 0}
                category = category_map.get(int(class_id), None)
                annotation = {
                    'time_frame': curr_frame,
                    'id': id,
                    'category_id': category,
                    'rle': rle
                }
                 
                mask = {'size': [int(im_height), int(im_width)], 'counts': rle.encode(encoding='UTF-8')} 

                annotation['bbox'] = pycocotools.mask.toBbox(mask)
                annotation['bbox'][2] = annotation['bbox'][0] + annotation['bbox'][2]
                annotation['bbox'][3] = annotation['bbox'][1] + annotation['bbox'][3]
                annotation['bbox_mode'] = BoxMode.XYXY_ABS

                annotation['segmentation'] = mask
                annotation['iscrowd'] = 0                
                
                record['annotations'].append(annotation)
                prev_frame = curr_frame
                index += 1
                
    return dataset_dicts
                
def kitti_mots_train(): return parse_kitti_mots(train=True)
def kitti_mots_test(): return parse_kitti_mots(train=False)

def run(model, weights_file, output_dir, inference=False):
    DatasetCatalog.clear()

    coco_names = [""] * 81
    coco_names[0] = "person"
    coco_names[1] = "nothing"
    coco_names[2] = "car"

    # Define the metadata
    metadata = {"thing_classes": coco_names,}

    DatasetCatalog.register('kitty_mots_train', kitti_mots_train)
    MetadataCatalog.get('kitty_mots_train').set(**metadata)
    DatasetCatalog.register('kitty_mots_valid', kitti_mots_test)
    MetadataCatalog.get('kitty_mots_valid').set(**metadata)
    kitty_mots_metadata = MetadataCatalog.get("kitty_mots_train")

    # Load the model
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    if model == 'mask_rcnn':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif model == 'fast_rcnn':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATASETS.VAL = ('kitty_mots_valid',)
    cfg.DATASETS.TRAIN = ('kitty_mots_train',)
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = output_dir

    if inference == False:  
        #Train NN 
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.0001
        cfg.SOLVER.MAX_ITER = 5    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)

        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        print("TRAINING! ")
        trainer.train()
        print("FINISHED! ")

    #inference finetuned
    if inference:
        cfg.MODEL.WEIGHTS = weights_file  # path to the model we just trained
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 


    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("kitty_mots_valid", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "kitty_mots_valid")
    print(val_loader)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mask_rcnn", "fast_rcnn"], default="fast_rcnn", help="Choose between mask_rcnn or fast_rcnn. Default: fast_rcnn")
    parser.add_argument("--weights", type=str, help="Path to model weights file (inference only)")
    parser.add_argument("--inference", action="store_true", help="Run inference only")
    parser.add_argument("--output_dir", type=str, help="Directory output")

    args = parser.parse_args()
    if args.inference:
        if args.weights:
            print("Running inference with model weights from file: ", args.weights)
            # add code here to run inference using the specified weights file
        else:
            print("You have to define the --weights argument (path to model weights file)")
            sys.exit()

    if args.output_dir:
        run(args.model,args.weights, args.output_dir, args.inference)
    else:
        run(args.model,args.weights, "./output", args.inference)
