import unittest

from vessel_detector import *

import os
import time
import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision

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


class TestVesselDetector(unittest.TestCase):
    def test_training(self):
        self.main(train = True)
     

    def test_evaluation(self):
        self.main(train = False)


    def main(self, train: bool):
        backbone_state_dict = r'../../../data/vessel_classifier_state_dict.pth'
        # Define all training params in one dict to make assumptions clear
        params = {
            # optimizer params from: https://arxiv.org/pdf/1506.01497.pdf
            'seed': 0,
            'num_classes': 2,
            'num_trainable_backbone_layers': 3,
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            # All samples have at least one ground truth bbox
            'no_null_samples': True,
            'test_size': 1,
            'shuffle': True,       
            'batch_size': 2,
            'num_epochs': 1,
            'print_every': 1,
            # Increase number of detections since there may be many vessels in an image
            'box_detections_per_img': 256,
            # Use small anchor boxes since targets are small
            'anchor_sizes': (8, 16, 32, 64, 128, 256),
            # IoU thresholds for mAP calculation
            'thresh_list': np.arange(0.5, 0.76, 0.05).round(8)
        }

        seed = params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        ImageFile.LOAD_TRUNCATED_IMAGES = True    # Necessary for PIL to work correctly

        # NOTE: InceptionV3 backbone requires input samples of size 299x299x3
        anchor_sizes = params['anchor_sizes']
        num_classes = params['num_classes']
        box_detections_per_img = params['box_detections_per_img']
        num_trainable_backbone_layers = params['num_trainable_backbone_layers']
        model = make_model(backbone_state_dict,
                           num_classes=num_classes,
                           anchor_sizes=anchor_sizes,
                           box_detections_per_img=box_detections_per_img,
                           num_trainable_backbone_layers=num_trainable_backbone_layers
        )

        device = torch.device('cuda')
        model = model.to(device)

        # Params from: https://arxiv.org/pdf/1506.01497.pdf
        lr = params['lr']
        momentum = params['momentum']
        weight_decay = params['weight_decay']
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)

        ship_dir = '../../../data/dev/'
        train_image_dir = os.path.join(ship_dir, 'imgs/')
        valid_image_dir = os.path.join(ship_dir, 'imgs/')
        masks = get_masks(ship_dir, train_image_dir, valid_image_dir)

        no_null_samples = params['no_null_samples']
        image_names, filtered_masks = filter_masks(masks, no_null_samples=no_null_samples)

        test_size = params['test_size']
        train_ids, train_masks, valid_ids, valid_masks = get_train_valid_dfs(
            filtered_masks, seed, test_size=test_size
        )

        vessel_dataset = VesselDataset(train_masks,
                                       train_ids,
                                       image_names,
                                       train_image_dir=train_image_dir,
                                       mode='train')
        vessel_valid_dataset = VesselDataset(valid_masks,
                                             valid_ids,
                                             image_names,
                                             valid_image_dir=valid_image_dir,
                                             mode='valid')

        #print("Train Size: %d" % len(train_ids))
        #print("Valid Size: %d" % len(valid_ids))

        batch_size = params['batch_size']
        shuffle = params['shuffle']
        collate_fn = lambda batch: tuple(zip(*batch))
        loader = DataLoader(
                    dataset=vessel_dataset,
                    shuffle=shuffle,
                    #num_workers = 0,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available()
                )

        valid_loader = DataLoader(
                    dataset=vessel_valid_dataset,
                    shuffle=shuffle,
                    #num_workers = 0,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    pin_memory=torch.cuda.is_available()
                )

        num_epochs = params['num_epochs']
        print_every = params['print_every']
        thresh_list = params['thresh_list']

        #print('Starting Training...\n')
        for epoch in range(num_epochs):
            if train:
                model = train_one_epoch(model,
                                        optimizer,
                                        loader,
                                        device,
                                        epoch,
                                        lr_scheduler = None, 
                                        batch_size=batch_size,
                                        print_every=print_every,
                                        num_epochs = num_epochs
                )
            else:
                mAP = evaluate(model, valid_loader, device, thresh_list)
            #print_metrics(metrics, epoch, thresh_list)
            #print('Saving Model...\n')
            #torch.save(model.state_dict(), savepath)
            #print('Model Saved.\n')
        #print('Finished Training.\n')
        



if __name__ == '__main__':
    unittest.main()
