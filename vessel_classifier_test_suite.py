from vessel_classifier import *

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
    preds = torch.argmax(outputs, axis=1)
    precision = precision_score(outputs, preds)
    return precision


def calculate_recall(outputs, labels):
    preds = torch.argmax(outputs, axis=1)
    recall = recall_score(outputs, preds)
    return recall


def make_confusion_matrix(outputs, labels):
    preds = torch.argmax(outputs, axis=1)
    confusion = confusion_matrix(labels, outputs)
    return confusion


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
    ship_dir = '../data/airbus-ship-detection/'
    test_image_dir = os.path.join(ship_dir, 'train_v2/')
    masks = pd.read_csv(os.path.join(ship_dir,
                                     'train_ship_segmentations_v2.csv'))
    unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
    _, test_ids = train_test_split(unique_img_ids, 
                     test_size = 0.01, 
                     stratify = unique_img_ids['counts'],
                     random_state=seed
                    )
    print("Test Size: %d" % len(test_ids))
    test_df = pd.merge(unique_img_ids, test_ids)

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
    metrics = test(model, criterion, test_loader)
    for metric_name, metric_val in metrics.items():
        print(metric_name, '\n', metric_val, '\n')
