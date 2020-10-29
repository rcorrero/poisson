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
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, Normalize
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile, ImageFilter


class TestVesselClassifier(unittest.TestCase):
    def test_validation_loader(self):
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)

        
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
                    self.image_labels = list(img_df.counts - 1) # Since an image with no mask has 'count' == 1 in df
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
                        RandomBlur(p=0.5, radius=2),
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
                    img_path = os.path.join(self.test_image_dir, img_file_name)

                #img = imread(img_path)
                img = Image.open(img_path)
                label = self.image_labels[idx]
                if self.mode =='train':
                    img = self.train_transform(img)
                elif self.mode == 'valid':
                    img = self.valid_transform(img)
                else:
                    img = self.test_transform(img)

                if self.mode == 'train' or self.mode == 'valid':
                    return img, label
                else:
                    return img, img_file_name

        ship_dir = '/dev/'
        train_image_dir = os.path.join(ship_dir, 'imgs/')
        valid_image_dir = os.path.join(ship_dir, 'imgs/')
        masks = pd.read_csv(os.path.join(ship_dir,'train_ship_segmentations_v2.csv'))
        unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
        train_ids, valid_ids = train_test_split(
            unique_img_ids, 
            test_size = 0.01, 
            stratify = unique_img_ids['counts'],
        )
        train_df = pd.merge(unique_img_ids, train_ids)
        valid_df = pd.merge(unique_img_ids, valid_ids)

        binary = True
        vessel_dataset = VesselDataset(train_df, train_image_dir=train_image_dir, 
                                       mode='train', binary=binary)

        vessel_valid_dataset = VesselDataset(valid_df, valid_image_dir=valid_image_dir, 
                                       mode='valid', binary=binary)
        img_id = 47 # Image contained in valid_image_dir
        vessel_valid_dataset.__getitem__(img_id)

    
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
        vessel_valid_dataset = VesselDataset(valid_df, valid_image_dir=valid_image_dir, 
                                       mode='valid', binary=binary)
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

        num_epochs = 1
        print_every = 1

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            model.train()
            running_loss = 0.0
            for i, data in enumerate(list(loader[0])):
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                for j in range(100): # Overfit the sample
                    start = time.time()

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Print statistics
                running_loss += loss.item()
                end = time.time()
                minibatch_time += float(end - start)
                if (i + 1) % print_every == 0: 
                    minibatch_time = minibatch_time / (3600.0 * print_every)
                    num_minibatches_left = 1.01 * len(loader) - (i + 1)
                    num_minibatches_per_epoch = 1.01 * len(loader) - 1 + ((len(vessel_dataset) % batch_size) / batch_size)
                    num_epochs_left = num_epochs - (epoch + 1)
                    time_left = minibatch_time * \
                        (num_minibatches_left + num_epochs_left * num_minibatches_per_epoch)
                    print('[%d, %5d] Running Loss: %.3f' %
                          (epoch + 1, i + 1, (running_loss / print_every)))
                    print('           Number of Samples Seen: %d' %
                          (batch_size * ((i + 1) + epoch * num_minibatches_per_epoch)))
                    print('           Estimated Hours Remaining: %.2f\n' % time_left)
                    running_loss = 0.0
                    minibatch_time = 0.0
            print('Epoch %d completed. Running validation...\n' % (epoch + 1))
            metrics = validation(model, criterion, valid_loader)
            print('[Epoch %d] Validation Accuracy: %.3f | Validation Loss: %.3f\n' %
                 ((epoch + 1), metrics['valid_acc'], metrics['valid_loss']))
            print('Saving Model...\n')
            savepath = '../data/test_vessel_classifier_state_dict.pth'
            torch.save(model.state_dict(), savepath)
        print('Finished Training.')
        print('Saving Model...\n')
        savepath = '../data/test_vessel_classifier_state_dict.pth'
        torch.save(model.state_dict(), savepath)
        print('Done.')
    
    
if __name__ == '__main__':
    unittest.main()
