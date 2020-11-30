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
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip,\
    RandomVerticalFlip, Resize, Normalize
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile, ImageFilter


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
def rle_decode(mask_rle, shape=(299, 768)):
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
        target["area"] = torch.zeros((0,), dtype=torch.int64)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return target
    bbox_array = np.empty((N, 4), dtype=np.float32)
    masks = np.empty((N, shape[0], shape[1]), dtype=np.uint8)
    labels = torch.ones((N,), dtype=torch.int64)
    i = 0
    for rle in in_mask_list:
        if isinstance(rle, str):
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
    def __init__(self, img_df, train_image_dir=None, valid_image_dir=None, 
                 test_image_dir=None, transform=None, mode='train', binary=True):
        self.image_ids = list(img_df.ImageId.unique())
        if binary:
            self.image_labels = list(map(lambda x: 1 if x > 1 else 0, img_df.counts))
        else:
            self.image_labels = list(map(lambda x: x.values, img_df.EncodedPixels))
        self.train_image_dir = train_image_dir
        self.valid_image_dir = valid_image_dir
        self.test_image_dir = test_image_dir

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if transform is not None:
            self.train_transform = transform
        else:
            self.train_transform = Compose([
                Resize(size=(299,299), interpolation=2),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomBlur(p=0.95, radius=2),
                ToTensor(),
                Normalize(mean, std) # Apply to all input images
            ])
        self.valid_transform = Compose([
            Resize(size=(299,299), interpolation=2),
            RandomBlur(p=1.0, radius=2), # Blur all images
            ToTensor(),
            Normalize(mean, std) # Apply to all input images
        ])
        self.test_transform = Compose([
            Resize(size=(299,299), interpolation=2),
            ToTensor(),
            Normalize(mean, std) # Apply to all input images
        ])
        self.mode = mode


    def __len__(self):
        return len(self.image_ids)


    def _transform(self, target, boxes, area):
        # Calculate areas of ta


    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if self.mode == 'train':
            img_path = os.path.join(self.train_image_dir, img_file_name)
        elif self.mode == 'valid':
            img_path = os.path.join(self.valid_image_dir, img_file_name)
        else:
            img_path = os.path.join(self.test_image_dir, img_file_name)

        #img = imread(img_path)
        img = Image.open(img_path)
        if self.mode =='train':
            img = self.train_transform(img)
        elif self.mode == 'valid':
            img = self.valid_transform(img)
        else:
            img = self.test_transform(img)
            
        N = sum([1 for i in in_mask_list if isinstance(i, str)])
        if N > 0:
            for i, rle in enumerate(in_mask_list):
                if not is_valid(rle):
                    if N > 1:
                        N -= 1
                        in_mask_list = np.delete(in_mask_list, i)
                    else:
                        return self.__getitem__(np.random.randint(0, high=len(self)))
            
        label = self.image_labels[idx]
        target = make_target(label, N, shape=(299,299))
        img, target['boxes'], target['area'] = self._transform(img, target['boxes'], target['area'])
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        
        return img, target


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


def main(savepath, backbone_state_dict=None):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    backbone.load_state_dict(torch.load(backbone_state_dict))
    model = make_model(backbone_state_dict, num_classes=2)
    
    device = torch.device('cuda')
    model = model.to(device)
    
    lr = 1e-4
    weight_decay = 1e-7 # Default should be 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    ship_dir = '../data/airbus-ship-detection/'
    train_image_dir = os.path.join(ship_dir, 'train_v2/')
    valid_image_dir = os.path.join(ship_dir, 'train_v2/')
    masks = pd.read_csv(os.path.join(ship_dir,
                                     'train_ship_segmentations_v2.csv'))
    unique_img_ids = masks.groupby('ImageId').reset_index(name='counts')
    train_ids, valid_ids = train_test_split(unique_img_ids, 
                     test_size = 0.01, 
                     stratify = unique_img_ids['counts'],
                     random_state=seed
                    )
    print("Train Size: %d" % len(train_ids))
    print("Valid Size: %d" % len(valid_ids))
    train_df = pd.merge(unique_img_ids, train_ids)
    valid_df = pd.merge(unique_img_ids, valid_ids)

    binary = True
    vessel_dataset = VesselDataset(train_df, train_image_dir=train_image_dir, 
                                   mode='train', binary=binary)

    vessel_valid_dataset = VesselDataset(valid_df, valid_image_dir=valid_image_dir, 
                                   mode='valid', binary=binary)
    
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
        train_one_epoch(model, optimizer, loader, device, epoch, print_freq, loss_plt=False)
        print('\nEpoch %d completed. Running validation...\n' % (epoch + 1))
        evaluate(model, valid_loader, device)
        print('\nSaving Model...\n')
        torch.save(model.state_dict(), savepath)
        print('Done.\n')

    
if __name__ == '__main__':
    # Changes:
    #    - Increase percentage of training images blurred
    #    - Decrease weight decay from 1e-5 to 1e-7
    #    - Use state_dict from previous run
    backbone_loadpath = r'../data/vessel_classifier_state_dict.pth'
    savepath = r'vessel_detector_state_dict.pth'
    main(savepath, backbone_loadpath)
