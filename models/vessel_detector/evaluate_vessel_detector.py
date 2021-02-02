import ast
import os
import sys
import time
import torch
import math
import json
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision

from vessel_detector import get_mappings, RandomBlur, calculate_iou, calculate_map, \
    is_valid_box, print_metrics

# Create the service client.
from googleapiclient.discovery import build
from apiclient.http import MediaIoBaseDownload

from google.cloud import storage
from torch import tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.ops.boxes import box_iou
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip,\
    RandomVerticalFlip,  Normalize
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile, ImageFilter

import pathlib
from typing import Callable, Iterator, Union, Optional, List, Tuple, Dict
from torchvision.transforms.functional import resize


def download_img(dl_path, id_num):
    gcs_service = build('storage', 'v1')
    if not os.path.exists(os.path.dirname(dl_path)):
        try:
            os.makedirs(os.path.dirname(dl_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(dl_path, 'wb') as f:
      # Download the file from the Google Cloud Storage bucket.
      request = gcs_service.objects().get_media(bucket=BUCKET_NAME,
                                                object=dl_path)
      media = MediaIoBaseDownload(f, request)
      print('Downloading item ', id_num + 1, '...')
      print('Download Progress: ')
      done = False
      while not done:
          prog, done = media.next_chunk()
          print(prog.progress())

    print('Image ', id_num + 1, ' downloaded.')
    return dl_path


def download_imgs(dirpath, bucket_name, dl=True, max_num_imgs=None, skip_first=0):
    client = storage.Client()
    id = 0
    for blob in client.list_blobs(bucket_name, prefix=dirpath):
        if max_num_imgs is not None:
            if id > (max_num_imgs - 1):
                break
        filepath = blob.name
        print('Filepath: ', filepath)
        #if filepath[-14:] == 'AnalyticMS.tif':
        if id < skip_first:
            id += 1
            continue
        if dl == True:
            if not os.path.isfile(filepath):
                download_img(filepath, id)
        id += 1


def format_bbox(bbox):
    x0 = bbox['x']
    y0 = bbox['y']
    x1 = x0 + bbox['width']
    y1 = y0 + bbox['height']
    return x0, y0, x1, y1


def make_target(in_mask_list, N, shape=(299, 299)):
    if N == 0:
        target = {}
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0), dtype=torch.int64)
        return target
    bbox_array = np.zeros((N, 4), dtype=np.float32)
    labels = torch.ones((N,), dtype=torch.int64)
    i = 0
    for rle in in_mask_list:
        #########MODIFIED FOR TEST##########
        if isinstance(rle, str):
            bbox = ast.literal_eval(rle)
        else:
            bbox = rle
        
        if bbox:
            bbox = format_bbox(bbox)
            bbox_array[i,:] = bbox
            i += 1
    target = {
        'boxes': torch.from_numpy(bbox_array),
        'labels': labels,
    }
    assert not np.any(np.isnan(target['boxes'].numpy()))
    assert not np.any(np.isnan(target['labels'].numpy()))
    return target


class VesselDataset(Dataset):
    def __init__(self, 
                 boxes: dict, 
                 image_ids: list,
                 image_names: dict, 
                 train_image_dir=None, 
                 valid_image_dir=None, 
                 test_image_dir=None, 
                 transform=None, 
                 mode='train', 
                 binary=True):
        self.boxes = boxes
        self.image_ids = image_ids
        self.image_names = image_names
        self.train_image_dir = train_image_dir
        self.valid_image_dir = valid_image_dir
        self.test_image_dir = test_image_dir

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if transform is not None:
            self.train_transform = transform
        else:
            self.train_transform = Compose([
                RandomBlur(p=0.95, radius=2),
                ToTensor(),
                Normalize(mean, std) # Apply to all input images
            ])
        self.valid_transform = Compose([
            ToTensor(),
            Normalize(mean, std) # Apply to all input images
        ])
        self.test_transform = Compose([
            ToTensor(),
            Normalize(mean, std) # Apply to all input images
        ])
        self.mode = mode


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        idx = self.image_ids[idx] # Convert from input to image ID number
        img_file_name = self.image_names[idx]
        if self.mode == 'train':
            img_path = os.path.join(self.train_image_dir, img_file_name)
        elif self.mode == 'valid':
            img_path = os.path.join(self.valid_image_dir, img_file_name)
        else:
            img_path = os.path.join(self.test_image_dir, img_file_name)

        img = Image.open(img_path)
        if self.mode =='train' or self.mode =='valid':
            img_boxes = self.boxes[idx]
            N = sum([1 for i in img_boxes if 'x' in ast.literal_eval(i)])
            target = make_target(img_boxes, N, shape=(299, 299))
            for row in target['boxes']:
                assert is_valid_box(row, shape=(299,299)), print(img_file_name)
        
        if self.mode =='train':
            img = self.train_transform(img)
            assert not np.any(np.isnan(img.numpy()))
            return img, target
        elif self.mode == 'valid':
            img = self.valid_transform(img)
            assert not np.any(np.isnan(img.numpy()))
            return img, target
        else:
            img = self.test_transform(img)
            assert not np.any(np.isnan(img.numpy()))
            return img


def filter_masks(masks: pd.DataFrame, no_null_samples: bool) -> Tuple[dict, dict]:
    grp = list(masks.groupby('filename'))
    image_names =  {idx: filename for idx, (filename, _) in enumerate(grp)} 
    image_masks = {idx: m['region_shape_attributes'].values for idx, (_, m) in enumerate(grp)}
    if no_null_samples:
        to_remove = []
        for idx, in_mask_list in image_masks.items():
            for bbox in in_mask_list:
                bbox = ast.literal_eval(bbox)
                if not bbox:
                    to_remove.append(idx)
                    break              
        for idx in to_remove:
            del image_names[idx]
            del image_masks[idx]
    return image_names, image_masks


def calculate_map_rates(gt_boxes,
                    pr_boxes,
                    scores,
                    thresh,
                    pr_thresh,
                    device,
                    form='pascal_voc'):
    if gt_boxes.shape[0] == 0:
        gt_thresh = len(scores[scores > pr_thresh])
        if gt_thresh == 0:
            return 0.0, 0.0, 0.0, 1.0
        return 0.0, gt_thresh, 0.0, 0.0
    if pr_boxes.shape[0] == 0:
        return 0.0, 0.0, gt_boxes.shape[0], 0.0
    # sorting
    pr_indices = scores > pr_thresh
    pr_scores = scores[pr_indices]
    pr_boxes = pr_boxes[pr_indices]
    pr_boxes = pr_boxes[pr_scores.argsort().flip(-1)]
    if pr_boxes.shape[0] == 0:
        return 0.0, 0.0, gt_boxes.shape[0], 0.0

    iou_mat = calculate_iou(gt_boxes,pr_boxes,form)
    iou_mat = iou_mat.to(device)
    
    # thresholding
    iou_mat = iou_mat.where(iou_mat>thresh,tensor(0.).to(device))
    
    mappings = get_mappings(iou_mat)
    
    # mAP calculation
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()
    tp = tp.cpu().detach().numpy()
    fp = fp.cpu().detach().numpy()
    fn = fn.cpu().detach().numpy()
    return tp, fp, fn, 0.0


@torch.no_grad()
def evaluate(model, data_loader, device, thresh_list, prob_list):
    model.eval()
    tp_dict = {thresh: {prob: [] for prob in prob_list} for thresh in thresh_list}
    fp_dict = {thresh: {prob: [] for prob in prob_list} for thresh in thresh_list}
    tn_dict = {thresh: {prob: [] for prob in prob_list} for thresh in thresh_list}
    fn_dict = {thresh: {prob: [] for prob in prob_list} for thresh in thresh_list}
    for images, targets in data_loader:
        images = list(Variable(img).to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images, targets)
        # Calculate mAP
        for prob in prob_list:
            for thresh in thresh_list:
                rates_list = [calculate_map_rates(target['boxes'], 
                                        output['boxes'], 
                                        output['scores'], 
                                        thresh=thresh,
                                        pr_thresh=prob,
                                        device=device) \
                            for target, output in zip(targets, outputs)]
                #print(rates_list[0])
                tp_dict[thresh][prob] += [tup[0] for tup in rates_list]
                fp_dict[thresh][prob] += [tup[1] for tup in rates_list]
                fn_dict[thresh][prob] += [tup[2] for tup in rates_list]
                tn_dict[thresh][prob] += [tup[3] for tup in rates_list]
    for prob in prob_list:
        for thresh in thresh_list:
            tp_dict[thresh][prob] = np.sum(tp_dict[thresh][prob])
            fp_dict[thresh][prob] = np.sum(fp_dict[thresh][prob])
            fn_dict[thresh][prob] = np.sum(fn_dict[thresh][prob])
            tn_dict[thresh][prob] = np.sum(tn_dict[thresh][prob])
    return tp_dict, fp_dict, tn_dict, fn_dict


# Adapted from https://discuss.pytorch.org/t/faster-rcnn-with-inceptionv3-backbone-very-slow/91455
def make_model(model_path,
               backbone_state_dict,
               num_classes,
               anchor_sizes: tuple,
               box_detections_per_img: int,
               num_trainable_backbone_layers: int):
    '''
    Returns a Faster R-CNN model with pretrained ResNet-50 backbone. Parameters retained from 
    `vessel_detector.py` implementation of `make_model` for compatability with utility methods.
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 progress=True,
                                                                 num_classes=num_classes,
                                                                 pretrained_backbone=True,
    )
    model.load_state_dict(torch.load(model_path))
    return model


def test_models(params: dict, models: dict) -> dict:
    assert torch.cuda.is_available()
    torch.cuda.empty_cache() 
    thresh_list = params['thresh_list']
    prob_list = params['prob_list']
    backbone_state_dict_path = None #r'vessel_classifier_state_dict.pth'
    results_dict = {model_name: {} for model_name in models.keys()}
    for model_name, model_path in models.items():
        print('Testing ', model_name, '...')
        if model_name == 'vd_mask_rcnn_baseline':
            model = make_model(model_path,
                                backbone_state_dict_path,
                                num_classes=2,
                                anchor_sizes=params['anchor_sizes'],
                                box_detections_per_img=params['box_detections_per_img'],
                                num_trainable_backbone_layers=-1)
        else:
            model = make_model_from_dict(model_path,
                                        backbone_state_dict_path,
                                        num_classes=2,
                                        anchor_sizes=params['anchor_sizes'],
                                        box_detections_per_img=params['box_detections_per_img'],
                                        num_trainable_backbone_layers=-1)
        device = torch.device('cuda')
        model = model.to(device)
        tp_dict, fp_dict, tn_dict, fn_dict = evaluate(model, valid_loader, device, thresh_list, prob_list)
        results_dict[model_name]['tp'] = tp_dict
        results_dict[model_name]['fp'] = fp_dict
        results_dict[model_name]['tn'] = tn_dict
        results_dict[model_name]['fn'] = fn_dict
        torch.cuda.empty_cache() 
        print('Done testing ', model_name, '.\n\n')
    return results_dict


def get_AP(thresh, results: dict):
    tp_vals = np.array(list(results['tp'][thresh].values())).reshape(-1,1)
    fp_vals = np.array(list(results['fp'][thresh].values())).reshape(-1,1)
    fn_vals = np.array(list(results['fn'][thresh].values())).reshape(-1,1)
    mAP = tp_vals / (tp_vals + fp_vals + 1e-6) #+ fn_vals)
    return mAP


def get_AR(thresh, results: dict):
    tp_vals = np.array(list(results['tp'][thresh].values())).reshape(-1,1)
    fp_vals = np.array(list(results['fp'][thresh].values())).reshape(-1,1)
    fn_vals = np.array(list(results['fn'][thresh].values())).reshape(-1,1)
    mAP = tp_vals / (tp_vals + fn_vals + 1e-6) #+ fn_vals)
    return mAP


def get_mAP(results: dict, model):
    mAP = np.zeros((11,1))
    mAR = np.zeros((11,1))
    threshes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    num_threshes = len(threshes)
    for thresh in threshes: #params['thresh_list']:
        mAP += get_AP(thresh, results_dict[model])
        mAR += get_AR(thresh, results_dict[model])
    mAP /= num_threshes
    mAR /= num_threshes
    return mAP, mAR


if __name__ == '__main__':
    BUCKET_NAME = r'planet_imagery'
    dirpath = r'test/imgs'
    download_imgs(dirpath, BUCKET_NAME, max_num_imgs=10000)

    bbox_path = r'bboxes.csv'
    bboxes = pd.read_csv(bbox_path)

    image_names, valid_masks = filter_masks(bboxes, no_null_samples=False)
    #valid_ids = np.array(list(valid_masks.keys())).reshape((len(valid_masks),1))
    valid_ids = [i for i in range(len(valid_masks))]

    valid_image_dir = r'test/imgs/'
    # Define all training params in one dict to make assumptions clear
    params = {
        # optimizer params from: https://arxiv.org/pdf/1506.01497.pdf
        'seed': 0,
        'num_classes': 2,
        'num_trainable_backbone_layers': 3, # Set to `-1` to train all layers
        # Lr in paper is .001 but this may lead to NaN losses
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        # All samples have at least one ground truth bbox
        'no_null_samples': True,
        'test_size': 0.01,
        'shuffle': True,       
        'batch_size': 12,
        'num_epochs': 30,
        'print_every': 500,
        # Increase number of detections since there may be many vessels in an image
        'box_detections_per_img': 256,
        # Use small anchor boxes since targets are small
        'anchor_sizes': ((4,), (8,), (16,), (32,), (64,)),
        # IoU thresholds for mAP calculation
        'thresh_list': np.arange(0.05, 0.76, 0.05).round(8),
        'prob_list': np.arange(0.00, 1.01, 0.10).round(8)
    }

    vessel_valid_dataset = VesselDataset(valid_masks,
                                            valid_ids,
                                            image_names,
                                            valid_image_dir=valid_image_dir,
                                            mode='valid')

    print("Valid Size: %d" % len(valid_ids))

    batch_size = params['batch_size']
    shuffle = params['shuffle']
    collate_fn = lambda batch: tuple(zip(*batch))
    valid_loader = DataLoader(
                dataset=vessel_valid_dataset,
                shuffle=shuffle,
                #num_workers = 0,
                batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available()
            )

    models = {
        'vd_all_layers': r'vessel_detector_state_dict.pth',
        'vd_some_layers': r'vessel_detector_state_dict_partial_02.pth',
        'vd_mask_rcnn_baseline': r'vessel_detector_baseline_state_dict.pth'
    }
    
    results_dict = test_models(params, models)
    with open('test_results.json', 'w') as fp:
        json.dump(results_dict, fp, sort_keys=True, indent=4)

    model = r'vd_all_layers'
    mAP, mAR = get_mAP(results_dict, model)
    mAR = mAR[:-1,:]
    mAP = mAP[:-1,:]

    fig, ax = plt.subplots(1,1, figsize=(16,9))
    ax.set_title('Mean Average Recall and Mean Average Precision for Test Model 2')
    ax.set_xlabel('Mean Average Precision')
    ax.set_ylabel('Mean Average Recall')
    plt.xticks(ticks=np.arange(0.0,1.0,0.1))
    ax.grid()
    ax.plot(mAP, mAR, color='r')
    plt.savefig('vd_all_layers_graph.png')

    model = r'vd_some_layers'
    mAP, mAR = get_mAP(results_dict, model)
    mAR = mAR[:-1,:]
    mAP = mAP[:-1,:]

    fig, ax = plt.subplots(1,1, figsize=(16,9))
    ax.set_title('Mean Average Recall and Mean Average Precision for Test Model 1')
    ax.set_xlabel('Mean Average Precision')
    ax.set_ylabel('Mean Average Recall')
    plt.xticks(ticks=np.arange(0.0,1.0,0.1))
    ax.grid()
    plt.savefig('vd_some_layers_graph.png')
