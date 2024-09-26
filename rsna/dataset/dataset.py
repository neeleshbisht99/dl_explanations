from torch.utils import data
from pydicom import dcmread
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_f = './rsna-dataset/train_images'
test_f = './rsna-dataset/test_images'
columns = ['patientId', 'Target']


class Dataset(data.Dataset):
    
    def __init__(self, paths, labels, transform=None, all_data = None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.all_data = all_data
    
    def __getitem__(self, index):
        image = dcmread(f'{self.paths[index]}.dcm')
        image = image.pixel_array
        image = image / 255.0

        image = (255*image).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')

        label = self.labels[index][1]
        
        if self.transform is not None:
            image = self.transform(image)
        
        
        name = self.paths[index].split("/")[-1]
        GH = self.all_data['patientId']==name
        FIL = self.all_data[GH]
        #print("From the datset loader, name", name)
        box = [FIL['x'].values[0], FIL['y'].values[0], FIL['width'].values[0], FIL['height'].values[0]]
            
        return image, label, box
    
    def __len__(self):
        
        return len(self.paths)
    

class TrainAndValidateDataset:
    def __init__(self, batch_size=64, test_size=0.1, val_size=0.1, use_presaved=False):
        self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(299),
                        transforms.ToTensor()])
        if use_presaved:
            self.train_dataset = torch.load('./rsna-dataset/presaved-dataset/train_dataset.pth')
            self.val_dataset = torch.load('./rsna-dataset/presaved-dataset/val_dataset.pth')
            self.test_dataset = torch.load('./rsna-dataset/presaved-dataset/test_dataset.pth')
        else:
            label_data = pd.read_csv('./rsna-dataset/stage_2_train_labels.csv')
            self.all_data = label_data
            self.label_data = label_data.filter(columns)

            # self.train_labels, self.val_labels = train_test_split(self.label_data.values, test_size=test_size)
            # Split data into train and remaining
            self.train_labels, remaining_data = train_test_split(self.label_data.values, test_size=test_size + val_size, random_state=42)
            
            # Split remaining data into validation and test
            self.val_labels, self.test_labels = train_test_split(remaining_data, test_size=test_size / (test_size + val_size), random_state=42)
            
            self.train_paths = [os.path.join(train_f, image[0]) for image in self.train_labels]
            self.val_paths = [os.path.join(train_f, image[0]) for image in self.val_labels]
            self.test_paths = [os.path.join(train_f, image[0]) for image in self.test_labels]

            self.train_dataset = Dataset(self.train_paths, self.train_labels, transform=self.transform, all_data=self.all_data)
            self.val_dataset = Dataset(self.val_paths, self.val_labels, transform=self.transform, all_data=self.all_data)
            self.test_dataset = Dataset(self.test_paths, self.test_labels, transform=self.transform, all_data=self.all_data)

            torch.save(self.train_dataset, './rsna-dataset/presaved-dataset/train_dataset.pth')
            torch.save(self.val_dataset, './rsna-dataset/presaved-dataset/val_dataset.pth')
            torch.save(self.test_dataset, './rsna-dataset/final-dataset/test_dataset.pth')

        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = data.DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)


    def show_raw_dataset(self, num_to_show=9):
        plt.figure(figsize=(10,10))
    
        for i in range(num_to_show):
            plt.subplot(3, 3, i+1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            
            img_dcm = dcmread(f'{self.train_paths[i+20]}.dcm')
            img_np = img_dcm.pixel_array
            plt.imshow(img_np, cmap=plt.cm.binary)
            plt.xlabel(self.train_labels[i+20][1])

    
    def show_train_dataset(self):
        image = iter(self.train_dataset)
        #print(train_dataset.paths)
        img, label, box = next(image)
        print(label, box)
        #print(f'Tensor:{img}, Label:{label}')
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        print("image size: ",img.shape)

    def show_dataloader(self):
        batch = iter(self.train_loader)
        images, labels, _ = next(batch)
        print("batch shape",images.shape)
        image_grid = torchvision.utils.make_grid(images[:4])
        image_np = image_grid.numpy()
        img = np.transpose(image_np, (1, 2, 0))
        plt.imshow(img)