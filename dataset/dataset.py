import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset.image_dataset import ImageDataset
from dataset.utils import Utils
from config import CnnConfig

class CTScanDataset(Dataset):
    def __init__(self, scans, labels, transform=None):
        self.scans = scans
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = self.scans[idx]
        label = self.labels[idx]

        if self.transform:
            scan = self.transform(scan)

        return scan, label
    

class TrainTestDataset:

    def __init__(self):
        config = CnnConfig()
        normal_dataset, abnormal_dataset = self.get_data()
        abnormal_labels = np.array([1 for _ in range(len(abnormal_dataset))])
        normal_labels = np.array([0 for _ in range(len(normal_dataset))])

        train_dataset, validation_dataset, test_dataset = self.get_train_validate_test_dataset(normal_dataset, abnormal_dataset, normal_labels, abnormal_labels)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True) # channel, depth, width, height
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=config.batch_size, shuffle=False) # channel, depth, width, height
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False) # channel, depth, width, height


    def get_data(self):
        # Create directories
        output_dir = os.path.join(os.getcwd(), "MosMedData")
        os.makedirs(output_dir, exist_ok=True)

        normal_scan_paths = [os.path.join(output_dir, "CT-0", x) for x in os.listdir(os.path.join(output_dir, "CT-0"))]
        abnormal_scan_paths = [os.path.join(output_dir, "CT-23", x) for x in os.listdir(os.path.join(output_dir, "CT-23"))] + [os.path.join(output_dir, "CT-1", x) for x in os.listdir(os.path.join(output_dir, "CT-1"))]

        print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
        print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

        abnormal_dataset = np.array([ImageDataset.process_image(path) for path in abnormal_scan_paths])
        normal_dataset = np.array([ImageDataset.process_image(path) for path in normal_scan_paths])

        return normal_dataset, abnormal_dataset

        
    def get_train_validate_test_dataset(self, normal_dataset, abnormal_dataset, normal_labels, abnormal_labels):
        # Split data in the ratio 80-10-10 for training, validation and test.

        # Concatenate normal and abnormal scans and labels
        scans = np.concatenate((normal_dataset, abnormal_dataset), axis=0)
        labels = np.concatenate((normal_labels, abnormal_labels), axis=0)

        # First split: train and temp (which will be split into validation and test)
        x_train, x_temp, y_train, y_temp = train_test_split(scans, labels, test_size=0.2, stratify=labels, random_state=42)

        # Second split: validation and test
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        print(
            'Number of samples in train, validation and test are %d , %d and %d.'
            % (x_train.shape[0], x_val.shape[0], x_test.shape[0])
        )

        x_train_tensor = torch.tensor(x_train)
        y_train_tensor = torch.tensor(y_train)
        x_val_tensor = torch.tensor(x_val)
        y_val_tensor = torch.tensor(y_val)

        # Convert to tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)


        # Create datasets
        train_dataset = CTScanDataset(x_train_tensor, y_train_tensor, transform=Utils.train_preprocessing)
        validation_dataset = CTScanDataset(x_val_tensor, y_val_tensor, transform=Utils.validation_preprocessing)
        test_dataset = CTScanDataset(x_test_tensor, y_test_tensor, transform=Utils.validation_preprocessing)

        return train_dataset, validation_dataset, test_dataset