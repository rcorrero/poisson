import os 
import torch
import torchvision

import numpy as np
import pandas as pd

from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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


class ShipDataset(Dataset):
    def __init__(self, in_df, transforms=None, mode='train'):
        grp = list(in_df.groupby('ImageId'))
        self.image_ids =  [_id for _id, _ in grp] 
        self.image_masks = [m['EncodedPixels'].values for _,m in grp]
        self.transforms = transforms
        self.mode = mode


    def __len__(self):
        return len(self.image_ids)
               
        
    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        in_mask_list = self.image_masks[idx]
        N = sum([1 for i in in_mask_list if isinstance(i, str)])
        if N > 0:
            for i, rle in enumerate(in_mask_list):
                if not is_valid(rle):
                    if N > 1:
                        N -= 1
                        in_mask_list = np.delete(in_mask_list, i)
                    else:
                        return self.__getitem__(np.random.randint(0, high=len(self)))

        if self.mode == 'train':
            rgb_path = os.path.join(train_image_dir, img_file_name)
        else:
            rgb_path = os.path.join(test_image_dir, img_file_name)

        image = imread(rgb_path)

        # Make target dict
        target = make_target(in_mask_list, N, shape=(768,768))
        
        # Make image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
               
        if self.transforms is not None:
            image, mask = self.transforms(image, target)

        return image, target

    
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_box_size(rle):
    if type(rle) is not str:
        return float('inf')
    s = rle.split()
    length = sum([int(x) for x in s[1:][::2]])
    return length


def filter_ids(masks, max_thresh=300, n_neg_samples = 50000):
    # Remove duplicates using set
    id_set_1 = set(masks.loc[masks['EncodedPixels'].apply(
        lambda x: get_box_size(x) <= max_thresh
        )]['ImageId'].tolist())
    id_set_2 = set(masks.drop(
            masks[masks.EncodedPixels.notnull()].index
            ).sample(n_neg_samples).ImageId.tolist())
    
    return list(id_set_1.union(id_set_2))

    
if __name__ == '__main__':
    try:
        import utils
        import transforms as T
        from engine import train_one_epoch, evaluate, make_loss_plt
    except:
        # Download TorchVision repo to use some files from
        # references/detection
        os.system(
            'git clone https://github.com/pytorch/vision.git & \
            cd vision & \
            git checkout v0.3.0'
        )
        os.system(
        'cp references/detection/utils.py ./ & \
         cp references/detection/transforms.py ./ & \
         cp references/detection/coco_eval.py ./ & \
         cp references/detection/engine.py ./ & \
         cp references/detection/coco_utils.py ./'
        )
        import utils
        import transforms as T
        from engine import train_one_epoch, evaluate, make_loss_plt
        
    assert torch.cuda.is_available(), "cuda is not available."
    torch.cuda.empty_cache()

    num_epochs = 8
    max_num_samples = np.float('inf')
    valid_size = 3000
    max_thresh = 300
    batch_size = 16
    ship_dir = r'../../../airbus-dataset/'
    state_dict_path = r'maskrcnn_resnet50_state_dict.pth'
    
    print('-------------')
    
    # Make filepaths – Relative to `hacedor` directory
    if not isinstance(ship_dir, str) or len(ship_dir) == 0:
        ship_dir = r'../../../airbus-dataset/'
    train_image_dir = os.path.join(ship_dir, 'train_v2')
    test_image_dir = os.path.join(ship_dir, 'test_v2')

    print('Loading masks...')
    masks = pd.read_csv(ship_dir + r'train_ship_segmentations_v2.csv')
    print('Masks loaded.')
    print('-------------')
    print('Creating datasets...')

    labeled_masks = masks.groupby('ImageId').apply(lambda grp:
        grp.EncodedPixels.apply(
        lambda rle: get_box_size(rle) <= max_thresh).any()
        ).reset_index(name='counts')

    filtered_masks = masks[
        masks.ImageId.isin(labeled_masks[labeled_masks.counts == True].ImageId)]
    if isinstance(max_num_samples, int) and max_num_samples > 0:
        filtered_masks = filtered_masks.sample(max_num_samples)

    # use our dataset and defined transformations
    dataset = ShipDataset(filtered_masks, get_transform(train=True))
    dataset_valid = ShipDataset(filtered_masks, get_transform(train=False))
    print('Datasets created.')
    print('-------------')

    print('Size of training dataset: ', dataset.__len__())
    print('Size of validation set: ', valid_size)
    print('-------------')
    
    
    # split the dataset in train and validation set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-valid_size])
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-valid_size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    
    if isinstance(state_dict_path, str):
        model.load_state_dict(torch.load(state_dict_path))
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    print('-------------')

    # Training loop
    print('Training Model...')
    print('-------------')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, 
                        print_freq=1e1, loss_plt=True)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the validation dataset
        evaluate(model, data_loader_valid, device=device)
    
    print('Done Training.')
    print('Saving Model...')
    
    savepath = r'maskrcnn_resnet50_state_dict.pth'
    torch.save(model.state_dict(), savepath)
    
    print('Done.')
