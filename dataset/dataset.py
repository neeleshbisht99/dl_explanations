import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset

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
        normal_scan_paths, abnormal_scan_paths = self.get_paths()
        abnormal_labels = [1 for _ in range(len(abnormal_scan_paths))]
        normal_labels = [0 for _ in range(len(normal_scan_paths))]

        train_dataset, validation_dataset, test_dataset = self.get_train_validate_test_dataset(normal_scan_paths, abnormal_scan_paths, normal_labels, abnormal_labels)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True) # channel, depth, height, width
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=config.batch_size, shuffle=False) # channel, depth, height, width
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False) # channel, depth, height, width


    def get_paths(self):
        # Create directories
        output_dir = os.path.join(os.getcwd(), "MosMedData")
        os.makedirs(output_dir, exist_ok=True)

        normal_scan_paths = [os.path.join(output_dir, "CT-0", x) for x in os.listdir(os.path.join(output_dir, "CT-0"))]
        abnormal_scan_paths = [os.path.join(output_dir, "CT-23", x) for x in os.listdir(os.path.join(output_dir, "CT-23"))] + [os.path.join(output_dir, "CT-1", x) for x in os.listdir(os.path.join(output_dir, "CT-1"))]

        print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
        print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

        return normal_scan_paths, abnormal_scan_paths

    def get_train_validate_test_dataset(self, normal_scan_paths, abnormal_scan_paths, normal_labels, abnormal_labels):
        # Split data in the ratio 80-10-10 for training, validation and test.
        train_end_index = int(len(normal_scan_paths) * 0.8)
        valid_end_index = train_end_index + int(len(normal_scan_paths) * 0.15)
        
        train_scan_paths = normal_scan_paths[:train_end_index] + abnormal_scan_paths[:train_end_index]
        train_labels = normal_labels[:train_end_index] + abnormal_labels[:train_end_index]
        train_scan_paths, train_labels = Utils.shuffle(train_scan_paths, train_labels)

        valid_scan_paths = normal_scan_paths[train_end_index:valid_end_index] + abnormal_scan_paths[train_end_index:valid_end_index]
        valid_labels = normal_labels[train_end_index:valid_end_index] + abnormal_labels[train_end_index:valid_end_index]
        valid_scan_paths, valid_labels = Utils.shuffle(valid_scan_paths, valid_labels)

        test_scan_paths = normal_scan_paths[valid_end_index:] + abnormal_scan_paths[valid_end_index:]
        test_labels = normal_labels[valid_end_index:] + abnormal_labels[valid_end_index:]
        test_scan_paths, test_labels = Utils.shuffle(test_scan_paths, test_labels)

        x_train = np.array([ImageDataset.process_image(path) for path in train_scan_paths])
        x_val = np.array([ImageDataset.process_image(path, train=False) for path in valid_scan_paths])
        x_test = np.array([ImageDataset.process_image(path, train=False) for path in test_scan_paths])

        y_train = np.array([label for label in train_labels])
        y_val = np.array([label for label in valid_labels])
        y_test = np.array([label for label in test_labels])

        print(
            'Number of samples in train, validation and test are %d , %d and %d.'
            % (x_train.shape[0], x_val.shape[0], x_test.shape[0])
        )

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