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

        
def binary_acc(outputs, labels):
    preds = torch.argmax(outputs, axis=1)
    num_correct = (preds == labels).sum().float()
    acc = num_correct / labels.shape[0]
    return acc


def validation(model, criterion, valid_loader):
    #print("Calculating validation on hold-out....")
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            acc = binary_acc(outputs, labels)
            accs.append(acc.item())
        
    valid_loss = np.mean(losses)  # type: float
    valid_acc = np.mean(accs)
    #print('Average loss: %f' % valid_loss)
    #print('Average accuracy: %f' % valid_acc)
    
    metrics = {'valid_loss': valid_loss, 'valid_acc': valid_acc}
    return metrics


def main(savepath, load_state_dict=False, state_dict=None):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=2, 
                                            aux_logits=False)
    if load_state_dict:
        model.load_state_dict(torch.load(state_dict))
    #model = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=2)
    device = torch.device('cuda')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    lr = 1e-4
    weight_decay=1e-6
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
    print_every = 100

    print('Starting Training...\n')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        minibatch_time = 0.0
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
        torch.save(model.state_dict(), savepath)
    print('Finished Training.\n')
    print('Saving Model...\n')
    torch.save(model.state_dict(), savepath)
    print('Done.')

    
if __name__ == '__main__':
    load_state_dict = False
    loadpath = r'../data/vessel_classifier_state_dict.pth'
    savepath = r'vessel_classifier_state_dict.pth'
    main(savepath, load_state_dict, loadpath)
