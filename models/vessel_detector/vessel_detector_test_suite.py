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


# # From: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# def rle_decode(mask_rle, shape=(768, 768)):
#     '''
#     mask_rle: run-length as string formated (start length)
#     shape: (height,width) of array to return 
#     Returns numpy array, 1 - mask, 0 - background
#     '''
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape).T  # Needed to align to RLE direction


def make_target(in_mask_list, N, shape=(768, 768)):
    if N == 0:
        target = {}
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0), dtype=torch.int64)
        # target["masks"] = torch.from_numpy(np.zeros((1,
        #                                             null_mask_shape[0],
        #                                             null_mask_shape[1]),
        #                                             dtype=np.uint8)) 
        # target["area"] = torch.zeros((0,), dtype=torch.int64)
        # target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return target
    bbox_array = np.zeros((N, 4), dtype=np.float32)
    #masks = np.zeros((N, shape[0], shape[1]), dtype=np.uint8)
    labels = torch.ones((N,), dtype=torch.int64)
    i = 0
    for rle in in_mask_list:
        if isinstance(rle, str):
            # bbox = tuple(x1, y1, x2, y2)
            bbox = rle2bbox(rle, shape)
            bbox_array[i,:] = bbox
            # mask = rle_decode(rle)
            # masks[i, :, :] = mask
            i += 1
    # areas = (bbox_array[:, 3] - bbox_array[:, 1]) * (bbox_array[:, 2] - bbox_array[:, 0])
    # suppose all instances are not crowd
    # is_crowd = torch.zeros((N,), dtype=torch.int64)
    target = {
        'boxes': torch.from_numpy(bbox_array),
        'labels': labels,
        # #'masks': torch.from_numpy(masks),
        # 'area': torch.from_numpy(areas),
        # 'iscrowd': is_crowd
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


def filter_masks(masks: pd.DataFrame, no_null_samples: bool) -> Tuple[dict, dict]:
    if no_null_samples:
        masks_not_null = masks.drop(
            masks[masks.EncodedPixels.isnull()].index
        )
        masks = masks_not_null
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
        

def get_train_valid_dfs(masks: dict,
                           seed: int,
                           test_size: Union[float, int]
                          ) -> Tuple[list, dict, list, dict]:
    ids = np.array(list(masks.keys())).reshape((len(masks),1))
    train_ids, valid_ids = train_test_split(
         ids, 
         test_size = test_size, 
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
       # if target['masks'].shape[-1] != self.output_shape[-1]:
       #     target['masks'] = T_resize(target['masks'], size=self.output_shape,
       #                             interpolation=self.interpolation)
        target['boxes'] = self.resize_boxes(target['boxes'])
        return image, target
    
    
class RandomBlur:
    def __init__(self, p, radius=2):
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
            # image_id = torch.tensor([idx])
            # target["image_id"] = image_id
        
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
def make_model(backbone_state_dict, num_classes, anchor_sizes: tuple):
        inception = torchvision.models.inception_v3(pretrained=True, progress=False, 
                                                    num_classes=num_classes, aux_logits=False)
        #inception.load_state_dict(torch.load(state_dict))
        modules = list(inception.children())[:-1]
        backbone = nn.Sequential(*modules)

        for layer in backbone:
            for p in layer.parameters():
                p.requires_grad = False # Freezes the backbone layers

        backbone.out_channels = 2048

        anchor_generator = AnchorGenerator(sizes=(anchor_sizes,),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        model = FasterRCNN(backbone,
                           min_size=299,
                           max_size=299,
                           rpn_anchor_generator=anchor_generator,
                           box_predictor=FastRCNNPredictor(1024, num_classes)
        )

        return model


def train_print(i, running_loss, 
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
                    lr_scheduler,
                    batch_size,
                    print_every,
                    num_epochs):
    model.train()
    running_loss = 0.0
    minibatch_time = 0.0

    for i, (inputs, targets) in enumerate(data_loader):
        start = time.time()
        inputs = [Variable(input).to(device) for input in inputs]
        targets = [{k: Variable(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(inputs, targets)
        print('LOSS DICT: ', loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        if not math.isfinite(losses):
            print("Loss is %-10.5f, stopping training".format(losses))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += losses
        end = time.time()
        minibatch_time += float(end - start)
        if (i + 1) % print_every == 0:
            minibatch_time = minibatch_time / (3600.0 * print_every)
            num_minibatches_left = 1.01 * len(data_loader) - (i + 1)
            num_minibatches_per_epoch = 1.01 * len(data_loader) - 1 + \
            ((len(data_loader.dataset) % batch_size) / batch_size)
            num_epochs_left = num_epochs - (epoch + 1)
            time_left = minibatch_time * \
                (num_minibatches_left + num_epochs_left * num_minibatches_per_epoch)
            time_left *= 6.0 # Adjust for timing discrepencies
            train_print(i, running_loss, print_every, batch_size, epoch, 
                        num_minibatches_per_epoch, time_left)
            running_loss = 0.0
            minibatch_time = 0.0
    return model


def align_coordinates(boxes):
    """Align coordinates (x1,y1) < (x2,y2) to work with torchvision `box_iou` op
    Arguments:
        boxes (Tensor[N,4])
    
    Returns:
        boxes (Tensor[N,4]): aligned box coordinates
    """
    x1y1 = torch.min(boxes[:,:2,],boxes[:, 2:])
    x2y2 = torch.max(boxes[:,:2,],boxes[:, 2:])
    boxes = torch.cat([x1y1,x2y2],dim=1)
    return boxes


def calculate_iou(gt, pr, form='pascal_voc'):
    """Calculates the Intersection over Union.

    Arguments:
        gt: (torch.Tensor[N,4]) coordinates of the ground-truth boxes
        pr: (torch.Tensor[M,4]) coordinates of the prdicted boxes
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
        IoU values for every element in boxes1 and boxes2
    """
    if form == 'coco':
        gt = gt.clone()
        pr = pr.clone()

        gt[:,2] = gt[:,0] + gt[:,2]
        gt[:,3] = gt[:,1] + gt[:,3]
        pr[:,2] = pr[:,0] + pr[:,2]
        pr[:,3] = pr[:,1] + pr[:,3]

    gt = align_coordinates(gt)
    pr = align_coordinates(pr)
    
    return box_iou(gt,pr)


def get_mappings(iou_mat):
    mappings = torch.zeros_like(iou_mat)
    gt_count, pr_count = iou_mat.shape
    
    #first mapping (max iou for first pred_box)
    if not iou_mat[:,0].eq(0.).all():
        # if not a zero column
        mappings[iou_mat[:,0].argsort()[-1],0] = 1

    for pr_idx in range(1,pr_count):
        # Sum of all the previous mapping columns will let 
        # us know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[:,pr_idx]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the pivot element for mapping
        pivot = targets.argsort()[-1]
        mappings[pivot,pr_idx] = 1
    return mappings


def calculate_map(gt_boxes,pr_boxes,scores,thresh, form='pascal_voc'):
    # print("GT SHAPE: ", gt_boxes.shape)
    # print("PR SHAPE: ", pr_boxes.shape)
    if gt_boxes.shape[0] == 0:
        if pr_boxes.shape[0] == 0:
            return 1.0
        return 0.0
    if pr_boxes.shape[0] == 0:
        return 0.0
    # sorting
    pr_boxes = pr_boxes[scores.argsort().flip(-1)]
    iou_mat = calculate_iou(gt_boxes,pr_boxes,form) 
    
    # thresholding
    iou_mat = iou_mat.where(iou_mat>thresh,tensor(0.))
    # print('IOU SHAPE: ', iou_mat.shape)
    
    mappings = get_mappings(iou_mat)
    
    # mAP calculation
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()
    mAP = tp / (tp+fp+fn)
    
    return mAP


@torch.no_grad()
def evaluate(model, data_loader, device, thresh_list):
    # n_threads = torch.get_num_threads()
    # torch.set_num_threads(n_threads)
    cpu_device = torch.device("cpu")
    model.eval()

    #coco = get_coco_api_from_dataset(data_loader.dataset)
    #iou_types = _get_iou_types(model)
    #coco_evaluator = CocoEvaluator(coco, iou_types)
    start = time.time()
    mAP_dict = {thresh: [] for thresh in thresh_list}
    for images, targets in data_loader:
        images = list(Variable(img).to(device) for img in images)
        # print('Targets: ', targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #        model_time = time.time()
        outputs = model(images, targets)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # model_time = time.time() - model_time
        # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # Calculate mAP
        for thresh in thresh_list:
            mAP_list = [calculate_map(target['boxes'], 
                                      output['boxes'], 
                                      output['scores'], 
                                      thresh=thresh) \
                        for target, output in zip(targets, outputs)]
            #mAP = np.mean(mAP_list)
            mAP_dict[thresh] += mAP_list # Creates a list of mAP's for each sample
    #n_samples = len(data_loader.dataset)
    end = time.time()
    for thresh in thresh_list:
        mAP_dict[thresh] = np.mean(mAP_dict[thresh])
    # Create metrics dict
    metrics = mAP_dict
    metrics['eval_time'] = end - start
        

    # gather the stats from all processes
   # metric_logger.synchronize_between_processes()
   # print("Averaged stats:", metric_logger)
   # coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
   # coco_evaluator.accumulate()
   # coco_evaluator.summarize()
   # torch.set_num_threads(n_threads)
    return metrics


def print_metrics(metrics: dict, epoch: int, thresh_list: list) -> None:
    print('[Epoch %-2.d] Evaluation results:' % (epoch + 1))
    for thresh in thresh_list:
        mAP = metrics[thresh]
        print('    IoU (>) Threshold: %-3.3f | mAP: %-3.3f' % (thresh, mAP))
    print('\n')


def main(savepath, backbone_state_dict=None):
    # Define all numeric params in one dict to make assumptions clear
    params = {
        'seed': 0,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'test_size': 1,
        'batch_size': 2,
        'num_epochs': 3,
        'print_every': 1,
        'anchor_sizes': (8, 16, 32, 64, 128),
        'thresh_list': [0.5, 0.75, 1.0]
    }
    
    seed = params['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    anchor_sizes = params['anchor_sizes']
    model = make_model(backbone_state_dict=None, num_classes=1000, anchor_sizes=anchor_sizes)
    
    device = torch.device('cpu')
    model = model.to(device)

    # Params from: https://arxiv.org/pdf/1506.01497.pdf
    lr = params['lr']
    weight_decay = params['weight_decay']
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    ship_dir = '../../../dev/imgs/'
    train_image_dir = ship_dir
    valid_image_dir = ship_dir
    masks = get_masks(ship_dir, train_image_dir, valid_image_dir)
    
    no_null_samples = True
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

    print("Train Size: %d" % len(train_ids))
    print("Valid Size: %d" % len(valid_ids))
    
    batch_size = params['batch_size']
    shuffle = True
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

    print('Starting Training...\n')
    for epoch in range(num_epochs):  # loop over the dataset multiple times       
        train_one_epoch(model, optimizer, loader, device, epoch, lr_scheduler = None, 
                        batch_size=batch_size, print_every=print_every, num_epochs = num_epochs)
        print('Epoch %d completed. Running validation...\n' % (epoch + 1))
        metrics = evaluate(model, valid_loader, device, thresh_list)
        print_metrics(metrics, epoch, thresh_list)
        print('Saving Model...\n')
        #torch.save(model.state_dict(), savepath)
        print('Model Saved.\n')
    print('Finished Training.\n')


if __name__ == '__main__':
    #backbone_loadpath = r'../data/vessel_classifier_state_dict.pth'
    #savepath = r'vessel_detector_state_dict.pth'
    main(savepath=None, backbone_state_dict=None)
