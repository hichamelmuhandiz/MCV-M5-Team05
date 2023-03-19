import glob
import numpy as np
import cv2
import pycocotools
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from VisualizerCustom import *
from utils import *
import subprocess
import os
from moviepy.editor import VideoFileClip

DATASET_PATH = '/home/mcv/datasets/KITTI-MOTS/'

def parse_kitti_mots_sequence(train=True, debug=False, save_bboxes_im=False, num_sequence='-1'):
        # In the KITTI-MOTS paper sequences 2, 6, 7, 8, 10, 13, 14, 16 and 18 were chosen for the validation set, the remaining sequences for the training set.
        val_sequences = [2, 6, 7, 8, 10, 13, 14, 16, 18]
        train_sequences = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
        
        dataset_paths = sorted(glob.glob(DATASET_PATH+'instances_txt/*.txt'))
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
            file = file.replace('\\', '/')
            print(file)
            
            # If we only want to run one sequence
            if num_sequence != '-1' and file != DATASET_PATH+'instances_txt/'+num_sequence+'.txt':
                continue
            elif file != DATASET_PATH+'instances_txt/0000.txt':
                pass
            
                
            index = 0
            # record = {}
            prev_frame = 0
            curr_frame = 0
            
            with open(file, 'r') as f:
                for line in f:
                    curr_frame, id, class_id, im_height, im_width, rle = line.split()
                    
                    if debug == True:
                        print('Index:', index)
                        print('Object id:', id)
                        print('Class id:', class_id)
                    
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
                    
                    # It can be 0 or 1, do not know the difference, but by default it was 1 on other implementations
                    annotation['iscrowd'] = 0       
                    
                    if save_bboxes_im:
                        # If we want to save the binary masks with the corresponding bounding boxes for visualization
                        ground_truth_binary_mask = np.stack((pycocotools.mask.decode(mask),)*3, axis=-1)
                        ground_truth_binary_mask = ground_truth_binary_mask.astype(np.uint8).copy() * 255
                        ground_truth_binary_mask = cv2.rectangle(ground_truth_binary_mask,
                                                                (int(annotation['bbox'][0]), int(annotation['bbox'][1])), 
                                                                (int(annotation['bbox'][2]), int(annotation['bbox'][3])),
                                                                (0, 0, 255), 2)
                        cv2.imwrite('./outputs_marcos/output_%d.png' %index, ground_truth_binary_mask)
                        
                        # We can also save the original images with the bounding boxes for visualization
                        orig_im = cv2.imread(record['file_name'])
                        orig_im = cv2.rectangle(orig_im,
                                                (int(annotation['bbox'][0]), int(annotation['bbox'][1])), 
                                                (int(annotation['bbox'][2]), int(annotation['bbox'][3])),
                                                (0, 0, 255), 2)
                        
                        cv2.imwrite('./outputs_marcos/output_%d_orig.png' %index, orig_im)
                    
                    record['annotations'].append(annotation)
                    prev_frame = curr_frame
                    index += 1
                    
        return dataset_dicts
    
def parse_annotations_video(num_sequence='0000', start_frame=0, end_frame=np.Inf):
    def kitti_mots_train(): return parse_kitti_mots_sequence(train=True)
    def kitti_mots_test(): return parse_kitti_mots_sequence(train=False)
        
    DatasetCatalog.clear()

    coco_names = [""] * 81
    coco_names[0] = "person"
    coco_names[1] = "nothing"
    coco_names[2] = "car"

    DatasetCatalog.register('kitty_mots_train', kitti_mots_train)
    MetadataCatalog.get('kitty_mots_train').set(thing_classes=coco_names)
    DatasetCatalog.register('kitty_mots_valid', kitti_mots_test)
    MetadataCatalog.get('kitty_mots_valid').set(thing_classes=coco_names)
    kitty_mots_metadata = MetadataCatalog.get("kitty_mots_train")
    
    
    # We can also visualize a video of the dataset   
    dataset_dicts = parse_kitti_mots_sequence(train=True, num_sequence=num_sequence)
    
    # Load the video file
    coco_colors = [0,0,0] * 81
    coco_colors[0] = (102,255,102)
    coco_colors[1] = (102,255,255)
    coco_colors[2] = (102,102,255)
    MetadataCatalog.get('kitty_mots_train').set(thing_colors=coco_colors)
    # loop through all the frames            
    for index, d in enumerate(dataset_dicts):
        img = cv2.imread(d["file_name"])
        visualizer = VisualizerCustom(img[:, :, ::-1], metadata=kitty_mots_metadata, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        out = visualizer.draw_dataset_dict(d)
        frame = out.get_image()[:, :, ::-1]
        
        if index >= start_frame:
            cv2.imwrite('./output/videos/file%02d.png'%index, frame)
        if index > end_frame:
            break
    
    os.chdir('./output/videos')
    subprocess.call([
        'ffmpeg', '-y', '-framerate', '8', '-i', 'file%02d.png', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-r', '30', '-pix_fmt', 'yuv420p',
        'traffic_video.mp4'
    ])
    
    for file_name in glob.glob('./*.png'):
        os.remove(file_name)
        
    videoClip = VideoFileClip("./traffic_video.mp4")
    videoClip.write_gif("./traffic_video.gif", fps=5)
    
    os.chdir('./../..')