import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from matplotlib import rcParams
import matplotlib.patches as patches
from math import ceil
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from dataset.dataset import TrainAndValidateDataset
from resnet import Resnet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_and_validate_dataset = TrainAndValidateDataset(use_presaved=True)
test_loader = train_and_validate_dataset.test_loader

model = Resnet(device=device, path='./rsna-dataset/model_inception_v3_24092024_dict.pth')

model.model.eval()
correct = 0
total = 0
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels, _ in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()

        # Collect labels and probabilities
        probs = nn.functional.softmax(predictions, dim=1)[:, 1] 
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


print(f'Test_Acc: {100 * correct / total}')

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate AUC
auc_score = roc_auc_score(all_labels, all_probs)
print(f"AUC: {auc_score:.4f}")

# Calculate Precision-Recall curve and AUPRC
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
auprc_score = auc(recall, precision)
print(f"AUPRC: {auprc_score:.4f}")