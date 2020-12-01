import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip,\
    RandomVerticalFlip,  Normalize
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile, ImageFilter

import pathlib
from typing import Callable, Iterator, Union, Optional, List, Tuple, Dict
from torchvision.transforms.functional import resize

import pycocotools
from coco_utils import *
from coco_eval import *



def rle2bbox(rle, shape):
    '''
    rle: run-length encoded image mask, as string
    shape: (height, width) of image on which RLE was produced
    Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask
    
    Note on image vs np.array dimensions:
    
        np.array implies the `[y, x]` indexing order in terms of image dimensions,
        so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
        hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
        for RLE-encoded indices of np.array (which are produced by widely used kernels
        and are used in most kaggle competitions datasets)
    '''
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:,0] -= 1  # `start` is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1


# From: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def is_valid(rle, shape=(768,768)):
    width, height = shape
    xmin, ymin, xmax, ymax = rle2bbox(rle, shape)
    if xmin >= 0 and xmax <= width and xmin < xmax and \
    ymin >= 0 and ymax <= height and ymin < ymax:
        return True
    return False


def make_target(in_mask_list, N, shape=(768, 768)):
    if N == 0:
        target = {}
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0), dtype=torch.int64)
        target["masks"] = torch.from_numpy(np.zeros((N, shape[0], shape[1]), dtype=np.uint8)) 
        target["area"] = torch.zeros((0,), dtype=torch.int64)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return target
    bbox_array = np.zeros((N, 4), dtype=np.float32)
    masks = np.zeros((N, shape[0], shape[1]), dtype=np.uint8)
    labels = torch.ones((N,), dtype=torch.int64)
    i = 0
    for rle in in_mask_list:
        #if isinstance(rle, str):
        # bbox = tuple(x1, y1, x2, y2)
        bbox = rle2bbox(rle, shape)
        bbox_array[i,:] = bbox
        mask = rle_decode(rle)
        masks[i, :, :] = mask
        i += 1
    areas = (bbox_array[:, 3] - bbox_array[:, 1]) * (bbox_array[:, 2] - bbox_array[:, 0])
    # suppose all instances are not crowd
    is_crowd = torch.zeros((N,), dtype=torch.int64)
    target = {
        'boxes': torch.from_numpy(bbox_array),
        'labels': labels,
        'masks': torch.from_numpy(masks),
        'area': torch.from_numpy(areas),
        'iscrowd': is_crowd
    }
    return target


def get_masks(ship_dir: str, 
                train_image_dir: Union[str, pathlib.Path], 
                valid_image_dir: Union[str, pathlib.Path]
               ) -> pd.DataFrame:
    masks = pd.read_csv(os.path.join(ship_dir,
                                     'train_ship_segmentations_v2.csv'
                                    )
                       )
    return masks


def is_valid(rle, shape=(768,768)) -> bool:
    width, height = shape
    xmin, ymin, xmax, ymax = rle2bbox(rle, shape)
    if xmin >= 0 and xmax <= width and xmin < xmax and \
    ymin >= 0 and ymax <= height and ymin < ymax:
        return True
    return False


def filter_masks(masks: pd.DataFrame) -> Tuple[dict, dict]:
    grp = list(masks.groupby('ImageId'))
    image_names =  {idx: filename for idx, (filename, _) in enumerate(grp)} 
    image_masks = {idx: m['EncodedPixels'].values for idx, (_, m) in enumerate(grp)}
    to_remove = []
    for idx, in_mask_list in image_masks.items():
        N = sum([1 for i in in_mask_list if isinstance(i, str)])
        if N > 0:
            for i, rle in enumerate(in_mask_list):
                if not is_valid(rle):
                    to_remove.append(idx)
                    
    for idx in to_remove:
        del image_names[idx]
        del image_masks[idx]
    return image_names, image_masks
        

def get_train_valid_dfs(masks: dict, seed: int = 0) -> Tuple[list, dict, list, dict]:
    ids = np.array(list(masks.keys())).reshape((len(masks),1))
    train_ids, valid_ids = train_test_split(
         ids, 
         test_size = 0.01, 
         random_state=seed
        )
    train_ids, valid_ids = list(train_ids.flatten()), list(valid_ids.flatten())
    train_masks = {idx: masks[idx] for idx in train_ids}
    valid_masks = {idx: masks[idx] for idx in valid_ids}
    return train_ids, train_masks, valid_ids, valid_masks


class Resize:
    def __init__(self, 
                 input_shape = (768, 768), 
                 output_shape = (299, 299), 
                 interpolation=2
                ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.interpolation = interpolation
        
        
    def resize_boxes(self, boxes: torch.tensor) -> torch.tensor:
        x_orig, y_orig = self.input_shape
        x_new, y_new = self.output_shape
        x_scale = x_new / x_orig
        y_scale = y_new / y_orig
        # bbox = tuple(x1, y1, x2, y2)
        row_scaler = torch.tensor([x_scale, y_scale, x_scale, y_scale])
        boxes_scaled = torch.round(boxes * row_scaler).int() # Converts to new coordinates
        return boxes_scaled
        
        
    def __call__(self, image, target) -> Tuple[torch.tensor, dict]:
        image = resize(image, size=self.output_shape, interpolation=self.interpolation)
        target['masks'] = resize(target['masks'], size=self.output_shape,
                                interpolation=self.interpolation)
        target['boxes'] = self.resize_boxes(target['boxes'])
        return image, target
    
    
class RandomBlur:
    def __init__(self, p=0.5, radius=2):
        self.p = p
        self.radius = radius


    def __call__(self, x):
        prob = np.random.rand(1)[0]
        if prob < self.p:
            x = x.filter(ImageFilter.GaussianBlur(self.radius))
        return x
    
    
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
            RandomBlur(p=1.0, radius=2), # Blur all images
            ToTensor(),
            Normalize(mean, std) # Apply to all input images
        ])
        self.test_transform = Compose([
            transforms.Resize(size=(299,299), interpolation=2),
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

        #img = imread(img_path)
        img = Image.open(img_path)
        if self.mode =='train' or self.mode =='valid':
            img_boxes = self.boxes[idx]
            N = sum([1 for i in img_boxes if isinstance(i, str)])
            target = make_target(img_boxes, N, shape=(768, 768))
            img, target = Resize(input_shape = (768, 768), 
                                 output_shape = (299, 299)
                                )(img, target)
            # Make image_id
            image_id = torch.tensor([idx])
            target["image_id"] = image_id
        
        if self.mode =='train':
            img = self.train_transform(img)
            return img, target
        elif self.mode == 'valid':
            img = self.valid_transform(img)
            return img, target
        else:
            img = self.test_transform(img)
            return img
        

# Adapted from https://discuss.pytorch.org/t/faster-rcnn-with-inceptionv3-backbone-very-slow/91455
def make_model(state_dict, num_classes):
        inception = torchvision.models.inception_v3(pretrained=False, progress=False, 
                                                    num_classes=num_classes, aux_logits=False)
        inception.load_state_dict(torch.load(state_dict))
        modules = list(inception.children())[:-1]
        backbone = nn.Sequential(*modules)

        for layer in backbone:
            for p in layer.parameters():
                p.requires_grad = False # Freezes the backbone layers

        backbone.out_channels = 2048

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        model = FasterRCNN(backbone, rpn_anchor_generator=anchor_generator,
                           box_predictor=FastRCNNPredictor(1024, num_classes))

        return model


def train_print(running_loss, 
                print_every, 
                batch_size, 
                epoch, 
                num_minibatches_per_epoch, 
                time_left):
    print('[%d, %5d] Running Loss: %.3f' %
          (epoch + 1, i + 1, (running_loss / print_every)))
    print('           Number of Samples Seen: %d' %
          (batch_size * ((i + 1) + epoch * num_minibatches_per_epoch)))
    print('           Estimated Hours Remaining: %.2f\n' % time_left)


def train_one_epoch(model, 
                    optimizer, 
                    data_loader, 
                    device, 
                    epoch, 
                    lr_scheduler = None, 
                    print_every = 100,
                    num_epochs = 30):
    model.train()
    running_loss = 0.0
    minibatch_time = 0.0

    for i, (inputs, targets) in enumerate(data_loader):
        start = time.time()
        inputs = Variable(inputs).cuda()
        targets = [{k: Variable(v).cuda() for k, v in t.items()} for t in targets]

        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not math.isfinite(losses):
            print("Loss is %-10.5f, stopping training".format(losses))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item()
        end = time.time()
        minibatch_time += float(end - start)
        if (i + 1) % print_every == 0:
            minibatch_time = minibatch_time / (3600.0 * print_every)
            num_minibatches_left = 1.01 * len(data_loader) - (i + 1)
            num_minibatches_per_epoch = 1.01 * len(data_loader) - 1 + \
            ((len(dataloader.dataset) % batch_size) / batch_size)
            num_epochs_left = num_epochs - (epoch + 1)
            time_left = minibatch_time * \
                (num_minibatches_left + num_epochs_left * num_minibatches_per_epoch)
            time_left *= 6.0 # Adjust for timing discrepencies
            train_print(running_loss, print_every, batch_size, epoch, 
                        num_minibatches_per_epoch, time_left)
            running_loss = 0.0
            minibatch_time = 0.0
    return model


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def main(savepath, backbone_state_dict=None):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    model = make_model(backbone_state_dict, num_classes=2)
    
    device = torch.device('cuda')
    model = model.to(device)

    # Params from: https://arxiv.org/pdf/1506.01497.pdf
    lr = 1e-3
    weight_decay = 0.0005 # Default should be 1e-5
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    ship_dir = '../data/airbus-ship-detection/'
    train_image_dir = os.path.join(ship_dir, 'train_v2/')
    valid_image_dir = os.path.join(ship_dir, 'train_v2/')
    masks = get_masks(ship_dir, train_image_dir, valid_image_dir)
    image_names, filtered_masks = filter_masks(masks)
    train_ids, train_masks, valid_ids, valid_masks = get_train_valid_dfs(
        filtered_masks
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

    print("Train Size: %d" % len(train_ids))
    print("Valid Size: %d" % len(valid_ids))
    
    batch_size = 64
    shuffle = True
    loader = DataLoader(
                dataset=vessel_dataset,
                shuffle=shuffle,
                #num_workers = 0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available()
            )

    valid_loader = DataLoader(
                dataset=vessel_valid_dataset,
                shuffle=shuffle,
                #num_workers = 0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available()
            )
    
    num_epochs = 30
    print_freq = 100

    print('Starting Training...\n')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
#####################################################################################        
#        train_one_epoch(model, optimizer, loader, device, epoch, lr_scheduler = None, 
#                        print_every = 100, num_epochs = 30)
#####################################################################################
        print('Epoch %d completed. Running validation...\n' % (epoch + 1))
        metrics = evaluate(model, valid_loader, device)
        print('Saving Model...\n')
        torch.save(model.state_dict(), savepath)
        print('Model Saved.\n')
    print('Finished Training.\n')


if __name__ == '__main__':
    backbone_loadpath = r'../data/vessel_classifier_state_dict.pth'
    savepath = r'vessel_detector_state_dict.pth'
    main(savepath, backbone_loadpath)
