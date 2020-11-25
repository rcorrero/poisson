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
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from PIL import Image, ImageFile, ImageFilter


def binary_acc(outputs, labels):
    preds = torch.argmax(outputs, axis=1)
    num_correct = (preds == labels).sum().float()
    acc = num_correct / labels.shape[0]
    return acc


def calculate_precision(outputs, labels):
    labels = labels.cpu().numpy()
    preds = torch.argmax(outputs, axis=1)
    preds = preds.cpu().numpy()
    precision = precision_score(labels, preds)
    return precision


def calculate_recall(outputs, labels):
    labels = labels.cpu().numpy()
    preds = torch.argmax(outputs, axis=1)
    preds = preds.cpu().numpy()
    recall = recall_score(labels, preds)
    return recall


def make_confusion_matrix(outputs, labels):
    labels = labels.cpu().numpy()
    preds = torch.argmax(outputs, axis=1)
    preds = preds.cpu().numpy()
    confusion = confusion_matrix(labels, preds)
    print(confusion)
    return confusion


class VesselDataset(Dataset):
    def __init__(self, img_df, train_image_dir=None, valid_image_dir=None, 
                 test_image_dir=None, transform=None, mode='train', binary=True):
        self.image_ids = img_df['sample_id'].tolist()
        if binary:
            self.image_labels = img_df['label'].tolist()
        else:
            self.image_labels = list(img_df.counts - 1) # Image with no mask has 'count' == 1 in df
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
                RandomBlur(p=0.85, radius=2),
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


    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if self.mode == 'train':
            img_path = os.path.join(self.train_image_dir, img_file_name)
        elif self.mode == 'valid':
            img_path = os.path.join(self.valid_image_dir, img_file_name)
        else:
            img_path = os.path.join(self.test_image_dir, img_file_name + '.jpg')

        #img = imread(img_path)
        img = Image.open(img_path)
        label = self.image_labels[idx]
        if self.mode =='train':
            img = self.train_transform(img)
        elif self.mode == 'valid':
            img = self.valid_transform(img)
        else:
            img = self.test_transform(img)
        return img, label


def test(model, criterion, test_loader):
    model.eval()
    losses = []
    accs, precisions, recalls = [], [], []
    confusion_matrix = np.zeros((2,2))
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            acc = binary_acc(outputs, labels)
            accs.append(acc.item())
            precision = calculate_precision(outputs, labels)
            precisions.append(precision)
            recall = calculate_recall(outputs, labels)
            recalls.append(recall)
            confusion_matrix += make_confusion_matrix(outputs, labels)
        
    test_loss = np.mean(losses)  # type: float
    test_acc = np.mean(accs)
    test_precision = np.mean(precisions)
    test_recall = np.mean(recalls)
    
    metrics = {'test_loss': test_loss, 'test_acc': test_acc, 
               'test_precision': test_precision, 'test_recall': test_recall,
               'confusion_matrix': confusion_matrix}
    return metrics


def make_test_loader():
    seed = 0
    ship_dir = '../data/test/'
    test_image_dir = os.path.join(ship_dir, 'imgs/')
    labels = pd.read_csv(os.path.join(ship_dir, 'labels.csv'))
    print("Test Size: %d" % len(labels['sample_id'].tolist()))
    test_df = labels

    binary = True
    vessel_test_dataset = VesselDataset(test_df, test_image_dir=test_image_dir, 
                                   mode='test', binary=binary)
    
    batch_size = 64
    shuffle = False
    test_loader = DataLoader(
                dataset=vessel_test_dataset,
                shuffle=shuffle,
                #num_workers = 0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available()
            )
    return test_loader


if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    state_dict =  r'../data/vessel_classifier_state_dict.pth'
    model = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=2, 
                                            aux_logits=False)
    model.load_state_dict(torch.load(state_dict))
    device = torch.device('cuda')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loader = make_test_loader()
    print("Running test...")
    metrics = test(model, criterion, test_loader)
    print("Done.\n")
    for metric_name, metric_val in metrics.items():
        print(metric_name, '\n', metric_val, '\n')
