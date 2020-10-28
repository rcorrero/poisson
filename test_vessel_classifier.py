import unittest

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
from torchvision.transforms import ToTensor, Compose, RandomCrop, Lambda
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile


class TestVesselClassifier(unittest.TestCase):
    def test_load_state_dict(self):
        state_dict = r'../data/vessel_classifier_state_dict-01.pth'
        model = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=2, 
                                                aux_logits=False)
        model.load_state_dict(torch.load(state_dict))
        device = torch.device('cuda')
        model = model.to(device)

        
    def test_vessel_dataset(self):
        ship_dir = '../data/airbus-ship-detection/'
        train_image_dir = os.path.join(ship_dir, 'train_v2/')
        masks = pd.read_csv(os.path.join(ship_dir,
                                         'train_ship_segmentations_v2.csv'))
        unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
        train_ids, valid_ids = train_test_split(unique_img_ids, 
                         test_size = 0.05, 
                         stratify = unique_img_ids['counts'],
                         random_state=42
                        )
        train_df = pd.merge(unique_img_ids, train_ids)
        valid_df = pd.merge(unique_img_ids, valid_ids)
        binary = True
        vessel_dataset = VesselDataset(img_df=train_df, train_image_dir=train_image_dir, 
                                       mode='train', binary=binary)
        if binary == True:
            self.assertEqual(vessel_dataset.__getitem__(3)[1], 1)
        else:
            self.assertEqual(vessel_dataset.__getitem__(3)[1], 4)
    

    def test_validation(self):
        criterion = nn.CrossEntropyLoss()

        ship_dir = '../data/airbus-ship-detection/'
        valid_image_dir = os.path.join(ship_dir, 'train_v2/')
        masks = pd.read_csv(os.path.join(ship_dir,
                                         'train_ship_segmentations_v2.csv'))
        unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
        train_ids, valid_ids = train_test_split(unique_img_ids, 
                         test_size = 0.01, 
                         stratify = unique_img_ids['counts'],
                         random_state=0
                        )
        #self.assertEqual(len(train_ids), 192456)
        #self.assertEqual(len(valid_ids), 100)
        train_df = pd.merge(unique_img_ids, train_ids)
        valid_df = pd.merge(unique_img_ids, valid_ids)

        binary = True
        vessel_valid_dataset = VesselDataset(valid_df, train_image_dir=valid_image_dir, 
                                       mode='train', binary=binary)
        batch_size = 64
        valid_loader = DataLoader(
                dataset=vessel_valid_dataset,
                shuffle=False,
                #num_workers = 0,
                batch_size=batch_size,
                #pin_memory=torch.cuda.is_available()
            )
        model = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=2, 
                                                aux_logits=False)
        model = model.to('cuda')
        metrics = validation(model, criterion, valid_loader)
        self.assertTrue(metrics['valid_acc'] >= 0.0 and metrics['valid_acc'] <= 1.0)
        
        
    def test_io(self):
        model = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=2, 
                                                 aux_logits=False)
        model = model.to('cuda')
        savepath = '../data/test_vessel_classifier_state_dict.pth'
        torch.save(model.state_dict(), savepath)
        
        
    def test_full_training_loop(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        state_dict = r'../data/vessel_classifier_state_dict-01.pth'
        model = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=2, 
                                                aux_logits=False)
        model.load_state_dict(torch.load(state_dict))
        device = torch.device('cuda')
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        
        lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)

        ship_dir = '../data/airbus-ship-detection/'
        train_image_dir = os.path.join(ship_dir, 'train_v2/')
        valid_image_dir = os.path.join(ship_dir, 'train_v2/')
        masks = pd.read_csv(os.path.join(ship_dir,
                                         'train_ship_segmentations_v2.csv'))
        unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
        train_ids, valid_ids = train_test_split(unique_img_ids, 
                         test_size = 0.01, 
                         stratify = unique_img_ids['counts'],
                        )
        train_df = pd.merge(unique_img_ids, train_ids)
        valid_df = pd.merge(unique_img_ids, valid_ids)

        binary = True
        vessel_dataset = VesselDataset(train_df, train_image_dir=train_image_dir, 
                                       mode='train', binary=binary)

        vessel_valid_dataset = VesselDataset(valid_df, train_image_dir=valid_image_dir, 
                                       mode='train', binary=binary)

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

        num_epochs = 1
        print_every = 1

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(loader):
                start = time.time()
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if (i + 1) % print_every == 0: 
                    metrics = validation(model, criterion, valid_loader)

                    end = time.time()
                    minibatch_time = float(end - start) / 3600.0
                    num_minibatches_left = len(loader) - (i + 1)
                    num_minibatches_per_epoch = len(loader) - 1 + ((len(vessel_dataset) % batch_size) / batch_size)
                    num_epochs_left = num_epochs - (epoch + 1)
                    time_left = minibatch_time * \
                        (num_minibatches_left + num_epochs_left * num_minibatches_per_epoch)
                    print('[%d, %5d] Running Loss: %.3f' %
                          (epoch + 1, i + 1, running_loss))
                    print('           Number of samples seen: %d' %
                          (batch_size * ((i + 1) + epoch * num_minibatches_per_epoch)))
                    print('           Estimated hours remaining: %.2f\n' % time_left)
                    running_loss = 0.0
                    break
            print('Epoch %d completed. Running validation...\n' % (epoch + 1))
            metrics = validation(model, criterion, valid_loader)
            print('[Epoch %d] Validation Accuracy: %.3f | Validation Loss: %.3f\n' %
                 ((epoch + 1), metrics['valid_acc'], metrics['valid_loss']))
            print('Saving Model...')
            savepath = 'vessel_classifier_state_dict.pth'
            torch.save(model.state_dict(), savepath)
        print('Finished Training.')
        print('Saving Model...')
        savepath = 'vessel_classifier_state_dict.pth'
        torch.save(model.state_dict(), savepath)
        print('Done.')
    
    
if __name__ == '__main__':
    unittest.main()
