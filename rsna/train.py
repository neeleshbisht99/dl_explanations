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

config = {
    'learning_rate': 0.0001,
    'num_epochs' : 20
}

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_and_validate_dataset = TrainAndValidateDataset(use_presaved=True)

train_loader = train_and_validate_dataset.train_loader
val_loader = train_and_validate_dataset.val_loader
test_loader = train_and_validate_dataset.test_loader

model = Resnet(device=device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=0.00001)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


num_epochs = config['num_epochs']
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    # Training step
    model.model.train()
    for i, (images, labels, _) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    
    exp_lr_scheduler.step()

    # Validation step
    correct = 0
    total = 0
    model.model.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()
    print(f'Epoch: {epoch + 1}/{num_epochs}, Val_Acc: {100 * correct / total}')


current_time = datetime.now()
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

torch.save(model.model.state_dict(), f'./rsna-dataset/model_inception_v3_{current_time_str}_dict.pth')
print("Model and weights saved.")


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
