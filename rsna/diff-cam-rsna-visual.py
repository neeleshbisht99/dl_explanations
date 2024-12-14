#!/usr/bin/env python
# coding: utf-8

# In[1]:
# [1]
"""
    Mostly Imports...
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models import ResNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from math import ceil
import matplotlib.patches as patches
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score
from datetime import datetime
from torch.utils import data
from torchcam.methods import SmoothGradCAMpp, GradCAM, ScoreCAM, LayerCAM, XGradCAM

from model import Inceptionv3
from utils import CommonUtils, IOU, AUPRC

matplotlib.style.use('ggplot')

print(torch.__version__) #2.0.1+cu117

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()
shape = [299, 299]
# In[2]:
# [2]

batch_size = 64

print("START: dataset prep")
train_dataset = torch.load('./rsna-dataset/presaved-dataset/train_dataset.pth')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

val_dataset = torch.load('./rsna-dataset/presaved-dataset/val_dataset.pth')
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

test_dataset = torch.load('./rsna-dataset/presaved-dataset/test_dataset.pth')
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print("DONE: dataset prep")

# Load the model weights
cls_model = Inceptionv3(device=device, path='./rsna-dataset/model_inception_v3_24092024_dict.pth')
cls_model.model.eval()
print("DONE: model prep")

def write_to_json_file(filename, data):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write data to the JSON file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data written to {filename}")
    

# In[3]:
# [3]
"""
Create a collection of disciminative features from our trained classification model.
Register a forward hook on the final fully connected layer on our model to get the input discriminative features/embeddings.
With the validation data loader, for every batch of images in validation set, get the predictions.
For every image in the batch, get the ground truth label and append it to the cls_label_arr (label collection).
For every image in the batch, get the embeddings / features, detach, and copy to cpu and convert from tensor to numpy array.
Fianlly create a collection of features(cls_feature_arr), by appending the above calculated features to it.
Delete the not needed tensors from the GPU to avoid memory overlfow.
Finally save to a features and label file and load it.
"""
# generating discriminative features for the classificatiom model

cls_feature_arr = [] # value for all_features
cls_label_arr = []

embedding_global = None
def _embedding_hook_fn(module, input, output):
    global embedding_global
    embedding_global = input[0]  # Storing the input to the fc layer

embedding_hook = cls_model.model.fc.register_forward_hook(_embedding_hook_fn)

fea_file = './val_features_cls.npy'
label_file = './train_labels.npy'
if not os.path.exists(fea_file):
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader):
            # images = images.to(device) ## Will already go to device in the call function
            predictions = cls_model(images)

            for i in range(images.shape[0]):
                class_id = labels[i].item()
                cls_label_arr.append(class_id)

                embedding_global[i] /= embedding_global[i].norm()
                feature = embedding_global[i].detach().cpu().numpy()
                if len(cls_feature_arr) == 0:
                    cls_feature_arr = np.expand_dims(feature, axis=0)
                else:
                    cls_feature_arr = np.concatenate((cls_feature_arr, np.expand_dims(feature, axis=0)), axis = 0)

            del images, labels, predictions, feature
            torch.cuda.empty_cache()

    np.save(fea_file, cls_feature_arr)
    np.save(label_file, cls_label_arr)
else:
    cls_feature_arr = np.load(fea_file)
    cls_label_arr = np.load(label_file).tolist()


embedding_hook.remove()

print(cls_feature_arr.shape, len(cls_label_arr))

# In[4]:
# [4]
labels = np.array(cls_label_arr)
ref_indices = np.argwhere(labels == 0).squeeze(1)

labels = np.array(cls_label_arr)
target_indices = np.argwhere(labels == 1).squeeze(1)

# In[5]:
# [5]
"""
Create a demo set, that is an array of tuple (image, label and bbox).
In the demo set only those images are present that have atleast one bbox.
"""

demo_set = []

for images, labels, bboxs in tqdm(test_loader):
     # Iterate over each image in the batch
    for i in range(images.size(0)):
        # Select a single image and label
        image = images[i:i+1]  # Keep batch dimension (1, C, H, W)
        label = labels[i:i+1].cpu().item()

        bbox_coords = []
        for idx, bbox in enumerate(bboxs):
            # Convert tensor to a list or numpy array if needed
            coords_arr = bbox.tolist() 
            bbox_coords.append(coords_arr[i])

        # filter the images with atleast one bounding box for calculating IOU.
        if any(np.isnan(coord) for coord in bbox_coords):
            continue
        
        bbox_coords = [int(x) for x in bbox_coords]
        demo_set.append((image, label, bbox_coords))


# In[6]:
# [6]
"""
Get GradCAM, ScoreCAM and SmoothGradCAMpp for all the images in the above created demo set using the torchcam library.
And save the resulting cams to there corresponding .pth files.
Memoization is there, load them when run next time.
"""

from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

def init_model():
    cls_model = Inceptionv3(device=device, path='./rsna-dataset/model_inception_v3_24092024_dict.pth')
    cls_model.model.eval()
    return cls_model.model

file_name = 'results'

x_methods = [
    {
        'title': 'Grad-CAM',
        'class': GradCAM,
        'results': None,
        'heatmap_mask_arr': None,
        'file_name': f'{file_name}_grad.pth',
        'heatmap_mask_file_name': f'heatmap_mask_grad.npy',
        'auprc': [],
        'iou': []
    },
    {
        'title': 'Smooth Grad-CAM++',
        'class': SmoothGradCAMpp,
        'results': None,
        'heatmap_mask_arr': None,
        'file_name': f'{file_name}_sg.pth',
        'heatmap_mask_file_name': f'heatmap_mask_sg.npy',
        'auprc': [],
        'iou': []
    },
    {
        'title': 'Score-CAM',
        'class': ScoreCAM,
        'results': None,
        'heatmap_mask_arr': None,
        'file_name': f'{file_name}_score.pth',
        'heatmap_mask_file_name': f'heatmap_mask_score.npy',
        'auprc': [],
        'iou': []
    },
    {
        'title': 'Layer-CAM',
        'class': LayerCAM,
        'results': None,
        'heatmap_mask_arr': None,
        'file_name': f'{file_name}_layer.pth',
        'heatmap_mask_file_name': f'heatmap_mask_layer.npy',
        'auprc': [],
        'iou': []
    },
    {
        'title': 'XGrad-CAM',
        'class': XGradCAM,
        'results': None,
        'heatmap_mask_arr': None,
        'file_name': f'{file_name}_xgrad.pth',
        'heatmap_mask_file_name': f'heatmap_mask_xgrad.npy',
        'auprc': [],
        'iou': []
    }
]

if not os.path.exists(x_methods[-1]['file_name']):
    for method in x_methods:
        results = []
        heatmap_mask_arr = None
        model = init_model()
        x_class = method['class']
        x_file_name = method['file_name']
        x_heatmap_mask_file_name = method['heatmap_mask_file_name']
        cam_extractor = x_class(model)
        for image, target_id, bbox_coords in demo_set:
            # forward pass through model
            out = model(image.to(device))
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(1, out)
            result = activation_map[0].squeeze(0)
            results.append(result)

            # save to npy file
            shape = [299, 299]
            #pepare heatmap
            result = result.to('cpu').numpy().squeeze()
            resized_result = CommonUtils.get_resized_heatmap(result, shape)
            normalized_resized_result = resized_result / 255.0
            #prepare mask
            resized_mask = CommonUtils.bbox_to_mask(bbox_coords, shape)
            #join
            joined_heatmap_mask = np.stack((normalized_resized_result, resized_mask), axis=0) 
            if heatmap_mask_arr is None:
                heatmap_mask_arr = np.expand_dims(joined_heatmap_mask, axis=0)
            else:
                heatmap_mask_arr = np.concatenate([heatmap_mask_arr, np.expand_dims(joined_heatmap_mask, axis=0)], axis=0)
        method['results'] = results
        method['heatmap_mask_arr'] = heatmap_mask_arr
        torch.save(results, x_file_name)
        np.save(x_heatmap_mask_file_name, heatmap_mask_arr)
else:
    for method in x_methods:
        x_file_name = method['file_name']
        x_heatmap_mask_file_name = method['heatmap_mask_file_name']
        method['results'] = torch.load(x_file_name, map_location=device)
        method['heatmap_mask_arr'] = np.load(x_heatmap_mask_file_name)

print(len(x_methods[0]['results']))
    


# In[7]:
# [7]
"""
Following are the algos for calculating IOU, AUPRC and AUC.
Besides a class FeatureExtractor is defined that registers hook on a particular layer in the model etc
"""

Orig_img_size = 1000
img_size = 299

import cv2
import torch
import glob as glob
from torchvision import transforms
from torch.nn import functional as F
from torch import topk
import matplotlib.patches as patches

def cal_iou(bbox_coords, cam_result):
    heatmap = CommonUtils.get_resized_heatmap(cam_result.cpu().numpy(), (img_size, img_size))
    iou = IOU.compute_iou(bbox_coords, heatmap)
    return iou

def cal_auprc(bbox_coords, cam_result):
    heatmap = CommonUtils.get_resized_heatmap(cam_result.cpu().numpy(), (img_size, img_size))
    auprc = AUPRC.compute_auprc(bbox_coords, heatmap)
    return auprc

def cal_auc(bbox_coords, cam_result):
    heatmap = CommonUtils.get_resized_heatmap(cam_result.cpu().numpy(), (img_size, img_size))
    auc = AUPRC.compute_auc(bbox_coords, heatmap)
    return auc
    
last_conv_layer_name = 'Mixed_7c'

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features_blobs = []
        self.hooks = []  # Store hooks to allow removal
        self.embedding = None

    def hook_fn(self, module, input, output):
        self.features_blobs.append(output)

    def _embedding_hook_fn(self, module, input, output):
        # This method will be called during the forward pass
        self.embedding = input[0]  # Storing the input to the fc layer

    def register_hooks(self, layer_names):
        for name, layer in self.model.named_modules():
            if name in layer_names:
                hook = layer.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)

        embedding_hook = self.model.fc.register_forward_hook(self._embedding_hook_fn)
        self.hooks.append(embedding_hook)

    def get_features(self):
        return self.features_blobs

    def get_embedding(self):
        return self.embedding

    def remove_hooks(self):
        """Removes all hooks after they have been used."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# In[8]:
# [8]

cls_feature_arr_0 = cls_feature_arr[ref_indices,:]  # used in diff-cam calculation
cls_feature_arr_1 = cls_feature_arr[target_indices,:]  # used in diff-cam calculation
feature_1_mean = cls_feature_arr_1.mean(axis = 0)
feature_0_mean = cls_feature_arr_0.mean(axis = 0)

def calc_cam(feature_conv, ws):
    bz, nc, h, w = feature_conv.shape
    cam = ws.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

def DiffCAM_v3(feature_conv, version = 'w-diff'):
    output_cam = []
    if version == 'w-diff':
        cam1 = calc_cam(feature_conv, feature_1_mean - feature_0_mean)
        cam2 = calc_cam(feature_conv, feature_1_mean)
        cam = cam1 * cam2
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
    output_cam.append(torch.from_numpy(cam_img).float())    
    return output_cam

def decorate_saliency_method(result):
    result = result.to('cpu').numpy().squeeze()
    result = CommonUtils.get_resized_heatmap(result, shape)
    return result

# In[9]:
# [9]

scores = []
results = []
model = init_model()
model.eval()

iou_gradcam = []
iou_sgcam = []
iou_scorecam = []
iou_diffcam = []

auprc_gradcam = []
auprc_sgcam = []
auprc_scorecam = []
auprc_diffcam = []

for method in x_methods:
    method['iou'] = []
    method['auprc']= []

batch_size = 10
fig, ax = plt.subplots(batch_size, 6, figsize=(30, 5 * (batch_size + 1)))
shining_yellow = '#FFD700'

# Adding titles for the first row
# ax[0, 0].set_title('Sample')
# ax[0, 1].set_title('Mask')
# ax[0, 2].set_title('DiffCAM')
# ax[0, 3].set_title('GradCAM')
# ax[0, 4].set_title('ScoreCAM')
# ax[0, 5].set_title('LayerCAM')

i = -1
### BEST EXAMPLE
example_ids = [95, 885, 498, 824, 511, 165, 96, 129, 474, 313, 154, 487, 815, 873, 240, 176, 866, 183, 10, 16, 282, 845, 15, 297, 0, 187, 195, 532, 675, 481, 904, 527, 265, 41, 951, 234, 876, 270, 193, 371, 698, 816, 329, 770, 644, 266, 304, 387, 53, 394, 396, 672, 855, 811, 696, 463, 404, 142, 618, 575, 488, 355, 228, 365, 196, 661, 668, 44, 874, 241, 892, 837, 605, 13, 695, 673, 108, 197, 938, 550, 835, 294, 79, 262, 178, 716, 251, 648, 198, 949, 397, 596, 48, 915, 105, 793, 724, 942, 930, 590, 912, 875, 285, 218, 247, 805, 450, 416, 733, 700, 701, 449, 441, 124, 717, 230, 459, 622, 745, 134, 859, 680, 120, 615, 558, 860, 326, 665, 528, 288, 115, 685, 186, 501, 842, 90, 63, 862, 132, 629, 514, 797, 828, 929, 620, 28, 853, 854, 925, 500, 841, 62, 924, 635, 410, 286, 76, 99, 8, 146, 534, 789, 579, 678, 110, 123, 725, 570, 291, 497, 654, 702, 318, 641, 372, 382, 406, 839, 3, 630, 87, 790, 566, 122, 54, 795, 465, 188, 755, 572, 260, 813, 389, 298, 932, 358, 250, 694, 446, 391, 462, 891, 375, 883, 900, 37, 71, 713, 684, 553, 27, 281, 119, 822, 103, 59, 803, 469, 954, 939, 760, 94, 6, 911, 257, 826, 392, 200, 34, 321, 93, 799, 705, 393, 276, 852, 89, 603, 552, 533, 636, 599, 144, 902, 690, 332, 944, 97, 170, 32, 177, 336, 542, 325, 585, 150, 621, 627, 779, 308, 567, 1, 584, 840, 66, 236, 205, 221, 164, 315, 357, 530, 405, 613, 765, 512, 52, 338, 919, 801, 237, 594, 712, 61, 401, 340, 507, 149, 244, 658, 748, 152, 708, 444, 116, 738, 926, 284, 158, 255, 814, 478, 323, 84, 484, 19, 764, 571, 121, 67, 25, 546, 453, 897, 180, 827, 57, 637, 762, 541, 73, 227, 113, 137, 616, 466, 829, 646, 750, 226, 125, 870, 399, 719, 922, 356, 280, 2, 72, 638, 494, 299, 104, 610, 777, 426, 794, 419, 950, 834, 682, 166, 561, 767, 480, 448, 189, 565, 693, 612, 40, 414, 626, 370, 704, 182, 85, 778, 49, 135, 863, 293, 928, 407, 519, 424, 368, 720, 895, 548, 261, 817, 896, 505, 786, 390, 666, 568, 921, 451, 55, 832, 709, 544, 518, 493, 168, 460, 420, 699, 70, 256, 910, 418, 212, 471, 12, 937, 333, 520, 953, 302, 808, 223, 623, 917, 645, 303, 82, 422, 545, 362, 277, 766, 222, 269, 806, 769, 948, 305, 131, 677, 884, 639, 42, 455, 747, 36, 752, 607, 33, 697, 723, 349, 309, 429, 373, 109, 711, 369, 279, 219, 235, 823, 774, 890, 201, 727, 361, 274, 148, 194, 376, 398, 907, 802, 190, 159, 458, 931, 631, 952, 319, 133, 316, 101, 242, 238, 913, 364, 58, 126, 537, 601, 258, 593, 821, 267]

### RANDOM EXAMPLE 
example_ids = [913, 613, 639, 616, 623, 103, 630, 406, 285, 769, 638, 755, 119, 120, 542, 133, 596, 188, 907, 176, 49, 158, 355, 724, 892, 603, 618, 180, 568, 201, 567, 530, 698, 828, 242, 149, 599, 870, 148, 855, 109, 358, 57, 572, 910, 546, 802, 234, 53, 262, 96, 794, 82, 37, 712, 725, 52, 72, 394, 101, 912, 528, 291, 455, 66, 371, 95, 821, 702, 493, 612, 258, 716, 890, 853, 917, 63, 637, 93, 196, 552, 532, 286, 8, 778, 164, 424, 418, 827, 142, 805, 824, 711, 799, 326, 942, 900, 839, 541, 183, 392, 938, 866, 13, 885, 6, 480, 2, 449, 696, 404, 137, 463, 87, 123, 336, 924, 708, 397, 390, 356, 97, 132, 398, 863, 876, 62, 814, 338, 333, 817, 750, 230, 459, 621, 250, 108, 928, 571, 747, 733, 305, 194, 448, 256, 590, 954, 401, 478, 629, 620, 694, 902, 186, 484, 55, 279, 146, 678, 110, 565, 469, 765, 67, 533, 321, 854, 332, 166, 680, 309, 373, 10, 705, 44, 841, 277, 241, 797, 545, 512, 835, 673, 627, 767, 786, 507, 365, 695, 115, 690, 247, 33, 808, 615, 816, 168, 195, 113, 260, 713, 71, 921, 319, 829, 952, 951, 34, 466, 779, 915, 481, 59, 299, 419, 527, 720, 840, 329, 646, 265, 41, 444, 316, 645, 684, 450, 298, 752, 875, 789, 760, 178, 501, 349, 54, 308, 626, 407, 270, 420, 727, 159, 42, 84, 607, 925, 488, 429, 244, 561, 593, 842, 261, 823, 237, 94, 426, 422, 197, 131, 36, 500, 226, 154, 672, 622, 28, 405, 135, 221, 648, 644, 558, 219, 770, 70, 222, 953, 315, 318, 451, 929, 895, 361, 550, 497, 274, 276, 212, 73, 544, 150, 498, 303, 19, 937, 518, 884, 738, 187, 636, 926, 813, 257, 236, 288, 930, 280, 85, 697, 281, 370, 790, 834, 948, 200, 304, 860, 709, 922, 677, 340, 12, 537, 357, 514, 548, 931, 205, 79, 766, 462, 362, 949, 822, 826, 635, 762, 293, 104, 177, 223, 61, 391, 700, 811, 852, 665, 27, 832, 364, 122, 897, 815, 719, 126, 896, 594, 904, 369, 235, 748, 658, 701, 465, 269, 255, 939, 323, 90, 116, 375, 266, 584, 238, 410, 745, 764, 570, 717, 152, 654, 641, 944, 566, 387, 58, 845, 911, 494, 806, 190, 198, 793, 251, 795, 441, 182, 227, 883, 579, 297, 575, 393, 144, 862, 723, 121, 704, 487, 382, 302, 699, 240, 16, 458, 105, 129, 414, 218, 585, 932, 368, 837, 284, 376, 325, 520, 666, 453, 803, 124, 668, 396, 891, 474, 553, 801, 3, 0, 685, 389, 661, 519, 313, 460, 125, 777, 372, 631, 25, 99, 471, 267, 505, 282, 15, 399, 874, 193, 170, 859, 294, 873, 165, 189, 675, 1, 693, 134, 32, 601, 774, 605, 511, 76, 228, 534, 416, 40, 682, 446, 950, 89, 610, 919, 48]

### RANDOM EXAMPLE - TOP 121 ONLY RANDOMIZED
example_ids = [396, 717, 716, 262, 596, 745, 197, 371, 527, 487, 644, 811, 355, 165, 511, 198, 724, 550, 304, 622, 404, 859, 835, 949, 816, 270, 698, 904, 183, 668, 855, 134, 294, 282, 481, 176, 648, 129, 394, 459, 297, 53, 874, 824, 15, 672, 575, 387, 196, 498, 48, 805, 618, 885, 912, 675, 95, 251, 938, 266, 696, 845, 876, 793, 700, 450, 701, 241, 474, 365, 329, 79, 449, 605, 240, 661, 416, 105, 915, 13, 230, 815, 0, 124, 142, 532, 285, 10, 951, 154, 108, 195, 218, 441, 673, 875, 96, 313, 228, 930, 770, 265, 187, 892, 178, 397, 837, 41, 193, 590, 234, 942, 44, 463, 866, 16, 488, 247, 695, 873, 733]

count = 0
for i in example_ids[90:100]:
    image, target_id, bbox_coords = demo_set[i]
    if target_id != 1:
        assert "wrong example"
    
    feature_extractor = FeatureExtractor(model)
    feature_extractor.register_hooks([last_conv_layer_name])  # Adjust based on the last conv layer name
        # Forward pass to get features and predictions
    output = model(image.to(device))
    probs = F.softmax(output, dim = 1).data.squeeze()

    # Obtain feature maps from the last convolutional layer
    feature_conv = feature_extractor.get_features()[-1].cpu().detach().numpy()

    score_obj = {"index": i}
    for method in x_methods:
        iou = cal_iou(bbox_coords, method['results'][i])
        method['iou'].append(iou)

        auprc = cal_auprc(bbox_coords, method['results'][i])
        method['auprc'].append(auprc)
        score_obj[method['title']] = {'iou': iou, 'auprc': auprc}

    scores.append(score_obj)

    N = 5
    if True:
        # visualization
        image = image.squeeze(0).cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        results.append((image, f'Image_{i} p={probs[1].item():.3f}'))
        
        for method in x_methods:
            ret = overlay_mask(to_pil_image(image), to_pil_image(method['results'][i], mode='F'), alpha=0.5)
            iou = method['iou'][-1]
            auprc = method['auprc'][-1]
            results.append((ret, f'{method["title"]} {iou:.2f}'))
            
        if True:
            version = 'w-diff'
            CAMs = DiffCAM_v3(feature_conv, version)
            ret = overlay_mask(to_pil_image(image), to_pil_image(CAMs[0], mode='F'), alpha=0.5)            
            iou = cal_iou(bbox_coords, CAMs[0])
            auprc = cal_auprc(bbox_coords, CAMs[0])
            results.append((ret, f'DiffCAM {iou:.2f}'))        

            iou_diffcam.append(iou)
            auprc_diffcam.append(auprc)

    feature_extractor.remove_hooks()

    rx = ceil(bbox_coords[0]*img_size/Orig_img_size) if not np.isnan(bbox_coords[0]) else 0
    ry = ceil(bbox_coords[1]*img_size/Orig_img_size) if not np.isnan(bbox_coords[1]) else 0
    rw = ceil(bbox_coords[2]*img_size/Orig_img_size) if not np.isnan(bbox_coords[2]) else 0
    rh = ceil(bbox_coords[3]*img_size/Orig_img_size) if not np.isnan(bbox_coords[3]) else 0

    ax[count, 0].imshow(image, origin='upper', cmap='bone')
    ax[count, 0].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=4, edgecolor=shining_yellow, facecolor='none'))
    ax[count, 0].set_title(f'Pneumonia (Prob={probs[1].item():.3f})', fontsize=14)
    ax[count, 0].set_xticks([])  
    ax[count, 0].set_yticks([])

    img9 = ax[count, 1].imshow(image, cmap='bone')
    img10 = ax[count, 1].imshow(decorate_saliency_method(CAMs[0]), cmap='jet', alpha=0.5, extent=img9.get_extent())
    img11 = ax[count, 1].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=4, edgecolor=shining_yellow, facecolor='none'))
    ax[count, 1].set_title(f'''DiffCAM (AUPRC={auprc_diffcam[count]:.3f})''', fontsize=14)
    ax[count, 1].set_xticks([])  
    ax[count, 1].set_yticks([])

    img12 = ax[count, 2].imshow(image, cmap='bone')
    img13 = ax[count, 2].imshow(decorate_saliency_method(x_methods[0]['results'][i]), cmap='jet', alpha=0.5, extent=img12.get_extent())
    img14 = ax[count, 2].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=4, edgecolor=shining_yellow, facecolor='none'))
    ax[count, 2].set_title(f'''GradCAM (AUPRC={x_methods[0]['auprc'][count]:.3f})''', fontsize=14)
    ax[count, 2].set_xticks([])  
    ax[count, 2].set_yticks([])

    img15 = ax[count, 3].imshow(image, cmap='bone')
    img16 = ax[count, 3].imshow(decorate_saliency_method(x_methods[2]['results'][i]), cmap='jet', alpha=0.5, extent=img15.get_extent())
    img17 = ax[count, 3].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=4, edgecolor=shining_yellow, facecolor='none'))
    ax[count, 3].set_title(f'''ScoreCAM (AUPRC={x_methods[2]['auprc'][count]:.3f})''', fontsize=14)
    ax[count, 3].set_xticks([])  
    ax[count, 3].set_yticks([])

    img18 = ax[count, 4].imshow(image, cmap='bone')
    img19 = ax[count, 4].imshow(decorate_saliency_method(x_methods[3]['results'][i]), cmap='jet', alpha=0.5, extent=img18.get_extent())
    img20 = ax[count, 4].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=4, edgecolor=shining_yellow, facecolor='none'))
    ax[count, 4].set_title(f'''LayerCAM (AUPRC={x_methods[3]['auprc'][count]:.3f})''', fontsize=14)
    ax[count, 4].set_xticks([])  
    ax[count, 4].set_yticks([])

    img3 = ax[count, 5].imshow(image, cmap='bone')
    img4 = ax[count, 5].imshow(decorate_saliency_method(x_methods[4]['results'][i]), cmap='jet', alpha=0.5, extent=img3.get_extent())
    img21 = ax[count, 5].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=4, edgecolor=shining_yellow, facecolor='none'))
    ax[count, 5].set_title(f'''XGradCAM (AUPRC={x_methods[4]['auprc'][count]:.3f})''', fontsize=14)
    ax[count, 5].set_xticks([])  
    ax[count, 5].set_yticks([])

    count+=1


# %%
