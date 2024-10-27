#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
    Mostly Imports...
"""

import os
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

from torchcam.methods import SmoothGradCAMpp, GradCAM, ScoreCAM

#from model import SmallNet, ResNet18, ResNet50
#from train_test import train

matplotlib.style.use('ggplot')

print(torch.__version__) #2.0.1+cu117
# torch.cuda.set_device(1)
# define the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

# In[9]:

"""
    Mostly Imports, data loaders and inception model class initialization
"""

from math import ceil
import matplotlib.patches as patches
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score
from datetime import datetime
from torch.utils import data

from model import Inceptionv3
from utils import IOU, AUPRC

batch_size = 64

print("START: dataset prep")
train_dataset = torch.load('./rsna-dataset/presaved-dataset/train_dataset.pth')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

val_dataset = torch.load('./rsna-dataset/presaved-dataset/val_dataset.pth')
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

test_dataset = torch.load('./rsna-dataset/presaved-dataset/test_dataset.pth')
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print("DONE: dataset prep")

target_class_idx = 1  # Target class
ref_class_idx = 0  # Reference class


# Load the model weights
cls_model = Inceptionv3(device=device, path='./rsna-dataset/model_inception_v3_24092024_dict.pth')
cls_model.model.eval()
print("DONE: model prep")


# In[4]:

"""
    Gets the pretrained model from timm library and sets the num classes to 0 in order to remove the classification head to output the raw feature embeddings.
    Set the model to evaluation.
    Using the valiation loader, for the images in the valiation dataset, it get the embeddings (general_embs) for a batch of images.
    For each iamges, it extracts the image embeddings/features, normalizes it with euclid norm, detaches from the computation graph (to avoid gradient computation) transfers to cpu and converts to numpy array.
    Create a feature_arr by concatenating embeddings.
    Finally deleted the elements liek images, embedding and tensors are deleted to avoid memory overflow. The GPU cache is explicity cleared.
    Once the feature array is created it is saved to a .npy file.
    These stored general features are used as a reference group.

    To avoid redundant computation, if the .npy file is not empty, then the file is loaded and feature array is extracted from it.
"""

# generating general features for reference group discovery

fea_file = './val_features_general.npy'
#fea_file = './train_features_general.npy'
general_feature_arr = []

import timm
model_feature = timm.create_model('inception_v3.tv_in1k', pretrained=True, num_classes=0).to(device)
model_feature.eval()

if not os.path.exists(fea_file):
    with torch.no_grad():
        for images, _, _ in tqdm(val_loader):
            # images = images.to(device) ## Will already go to device in the call function
            general_embs = model_feature(images.to(device))

            for i in range(images.shape[0]):
                general_embs[i] /= general_embs[i].norm()
                feature = general_embs[i].detach().cpu().numpy()
                if len(general_feature_arr) == 0:
                    general_feature_arr = np.expand_dims(feature, axis=0)
                else:
                    general_feature_arr = np.concatenate((general_feature_arr, np.expand_dims(feature, axis=0)), axis = 0)
            
            # Delete tensors explicitly after use
            # print(torch.cuda.memory_allocated())
            del images, general_embs, feature
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated())

    np.save(fea_file, general_feature_arr)
else:
    general_feature_arr = np.load(fea_file)
    
print(general_feature_arr.shape)


# In[5]:

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

            # Delete tensors explicitly after use
            # print(torch.cuda.memory_allocated())
            del images, labels, predictions, feature
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated())

    np.save(fea_file, cls_feature_arr)
    np.save(label_file, cls_label_arr)
else:
    cls_feature_arr = np.load(fea_file)
    cls_label_arr = np.load(label_file).tolist()


embedding_hook.remove()

print(cls_feature_arr.shape, len(cls_label_arr))


# In[ ]:
"""
Create faiss index of features from the pretrained-inceptionNet-model that correspond to label 0. 
"""

import faiss                     # make faiss available
index_file = "index_features_general_0.faiss"
labels = np.array(cls_label_arr)
ref_indices = np.argwhere(labels == 0).squeeze(1)
target_indices = np.argwhere(labels == 1).squeeze(1)

general_feature_arr_0 = general_feature_arr[ref_indices,:]

print(general_feature_arr_0.shape)

feature_dim = general_feature_arr.shape[1]
if not os.path.exists(index_file):
    # res = faiss.StandardGpuResources()  # use a single GPU
    index_flat_general = faiss.IndexFlatL2(feature_dim)  # build a flat (CPU) index    
    # make it a flat GPU index
    # gpu_index_flat_general = faiss.index_cpu_to_gpu(res, 0, index_flat_general)
    index_flat_general.add(general_feature_arr_0)         # add vectors to the index
    faiss.write_index(index_flat_general, index_file)
else:
    index_flat_general = faiss.read_index(index_file)


# In[6]:
"""
Create faiss index of features from the our trained model that correspond to label 0. 
"""

import faiss                     # make faiss available
index_file = "index_features_cls_0.faiss"
labels = np.array(cls_label_arr)
ref_indices = np.argwhere(labels == 0).squeeze(1)

cls_feature_arr_0 = cls_feature_arr[ref_indices,:]

print(cls_feature_arr_0.shape)

feature_dim = cls_feature_arr_0.shape[1]
if not os.path.exists(index_file):
    # res = faiss.StandardGpuResources()  # use a single GPU
    index_flat_cls_0 = faiss.IndexFlatL2(feature_dim)  # build a flat (CPU) index    
    # make it a flat GPU index
    # gpu_index_flat_general = faiss.index_cpu_to_gpu(res, 0, index_flat_general)
    index_flat_cls_0.add(cls_feature_arr_0)         # add vectors to the index
    faiss.write_index(index_flat_cls_0, index_file)
else:
    index_flat_cls_0 = faiss.read_index(index_file)


# In[7]:

"""
Create faiss index of features from the our trained model that correspond to label 1. 
"""

import faiss                     # make faiss available
index_file = "index_features_cls_1.faiss"
labels = np.array(cls_label_arr)
target_indices = np.argwhere(labels == 1).squeeze(1)

cls_feature_arr_1 = cls_feature_arr[target_indices,:]

print(cls_feature_arr_1.shape)

feature_dim = cls_feature_arr_1.shape[1]
if not os.path.exists(index_file):
    # res = faiss.StandardGpuResources()  # use a single GPU
    index_flat_cls_1 = faiss.IndexFlatL2(feature_dim)  # build a flat (CPU) index    
    # make it a flat GPU index
    # gpu_index_flat_general = faiss.index_cpu_to_gpu(res, 0, index_flat_general)
    index_flat_cls_1.add(cls_feature_arr_1)         # add vectors to the index
    faiss.write_index(index_flat_cls_1, index_file)
else:
    index_flat_cls_1 = faiss.read_index(index_file)


# In[10]:
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


# In[13]:
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
if not os.path.exists('%s_grad.pth' % file_name):
    results_grad = []
    model = init_model()
    cam_extractor = GradCAM(model)
    for image, target_id, _ in demo_set:
        # forward pass through model
        out = model(image.cuda())
          # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(1, out)
        result = activation_map[0].squeeze(0)
        results_grad.append(result)
    
    results_score = []
    model = init_model()
    cam_extractor = ScoreCAM(model)
    for image, target_id, _ in demo_set:
        # forward pass through model
        out = model(image.cuda())
          # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(1, out)
        result = activation_map[0].squeeze(0)
        results_score.append(result)
    
    results_sg = []
    model = init_model()
    cam_extractor = SmoothGradCAMpp(model)
    for image, target_id, _ in demo_set:
        # forward pass through model
        out = model(image.cuda())
          # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(1, out)
        result = activation_map[0].squeeze(0)
        results_sg.append(result)
    
    torch.save(results_grad, '%s_grad.pth' % file_name)
    torch.save(results_sg, '%s_sg.pth' % file_name)
    torch.save(results_score, '%s_score.pth' % file_name)
else:
    results_grad = torch.load('%s_grad.pth' % file_name, map_location=device)
    results_sg = torch.load('%s_sg.pth' % file_name, map_location=device)
    results_score = torch.load('%s_score.pth' % file_name, map_location=device)

print(len(results_score))
    


# In[24]:

"""
Following are the algos for calculating IOU, AUPRC and AUC.
Besides a class FeatureExtractor is defined that registers hook on a particular layer in the model etc
"""

import cv2
import torch
import glob as glob
from torchvision import transforms
from torch.nn import functional as F
from torch import topk
import matplotlib.patches as patches
from utils import CommonUtils, IOU, AUPRC

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


# In[37]:
"""
Create an array of intra-variance, for each of the features for both the positive and negative classes.
"""

num_classes = 2
all_features_gpu = torch.from_numpy(cls_feature_arr).to(device)
all_labels_gpu = torch.from_numpy(np.array(cls_label_arr)).to(device)
class_indices = []
for class_id in range(num_classes):
    indices = (all_labels_gpu == class_id).nonzero()
    class_indices.append(indices)

intra_var = torch.zeros(feature_dim).to(device)
for i in tqdm(range(feature_dim)):
    for class_id in range(num_classes):
        indices = class_indices[class_id]
        feature_individual = all_features_gpu[indices,i]
        feature_individual -= feature_individual.mean()
        intra_var[i] += torch.var(feature_individual)

print(intra_var[:5])

del all_features_gpu, all_labels_gpu
torch.cuda.empty_cache()


# In[41]:
"""
Explain this !!!
"""

feature_centers = np.zeros((num_classes, feature_dim))
feature_centers[0] = cls_feature_arr_0.mean()
feature_centers[1] = cls_feature_arr_1.mean()

feature_centers_gpu = torch.from_numpy(feature_centers).to(device)
inter_var = torch.zeros(feature_dim).to(device)
for i in tqdm(range(feature_dim)):
    feature_centers_gpu_i = feature_centers_gpu[:,i]
    feature_centers_gpu_i -= feature_centers_gpu_i.mean()
    inter_var[i] = torch.var(feature_centers_gpu_i)
print(inter_var[:5])

fea_weight = torch.pow(inter_var / intra_var, 1)
fea_weight /= fea_weight.norm()
fea_weight = fea_weight.cpu().numpy()
print(fea_weight.mean(), fea_weight.max(), fea_weight.min())

del feature_centers_gpu
torch.cuda.empty_cache()


# In[50]:


K = 200

def DiffCAM(feature_conv, embedding, ref_class_idx):
    bz, nc, h, w = feature_conv.shape
    output_cam = []

    #for idx in class_idx:
    if True:
        vdiff = embedding - feature_centers[ref_class_idx]
        svs = np.identity(vdiff.shape[0])
        #ws = np.matmul(svs, vdiff.T)
        ws = np.matmul(Svs[ref_class_idx], vdiff.T)
        #print(vdiff[:20])
        #print(ws[:20])

        cam = ws.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        #cam[cam < 0] = 0
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(torch.from_numpy(cam_img).float())
#        cam_img = np.uint8(255 * cam_img)
#        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_reference_group(image, fea_emb):
    '''
    fea_emb = model_feature(image.to(device))    
    fea_emb = fea_emb[0] / fea_emb[0].norm()
    fea_emb = fea_emb.detach().cpu().numpy()
    '''
    
    D, I = index_flat_cls_0.search(np.expand_dims(fea_emb, axis=0), K) # sanity check

    knn_indexs = [I[0, i] for i in range(K)]
    return knn_indexs

def DiffCAM_v2(feature_conv, embedding, ref_group):
    bz, nc, h, w = feature_conv.shape
    output_cam = []

    counter_center = np.zeros(feature_dim)
    #print(ref_group)
    for idx in ref_group:
        counter_center += cls_feature_arr_0[idx]
    counter_center /= K

    sv = None
    for idx in ref_group:
        feature = cls_feature_arr_0[idx]
        feadiff = feature - counter_center
        #print(feature[:5], counter_center[:5], feadiff[:5])
        if sv is None:
            sv = np.zeros((feature_dim, feature_dim))
        sv += np.matmul(feadiff.T, feadiff)

    sv = np.linalg.inv(sv + 0.0001 * np.identity(feature_dim))

    #for idx in class_idx:
    if True:
        vdiff = embedding - counter_center
        ws = np.matmul(sv, vdiff.T)

        cam = ws.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        #cam[cam < 0] = 0
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(torch.from_numpy(cam_img).float())
    return output_cam

Orig_img_size = 1000
img_size = 299

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

def DiffCAM_v3(feature_conv, embedding, ref_group, version = 'w-diff'):
    output_cam = []

    counter_center = np.zeros(feature_dim)
    #print(ref_group)
    N = 50
    for idx in ref_group[:N]:
        counter_center += cls_feature_arr_0[idx]
    counter_center /= N
#    print(feature_1_mean[:10])
#    print(counter_center[:10])

    D, I = index_flat_cls_1.search(np.expand_dims(embedding, axis=0), K) # sanity check
    knn_indexs = [I[0, i] for i in range(K)]
    positive_center = np.zeros(feature_dim)
    for idx in knn_indexs[:N]:
        positive_center += cls_feature_arr_1[idx]
    positive_center /= N

    if True:
        if version == 'emb-diff':
            ws = positive_center - counter_center
            cam_img = calc_cam(feature_conv, ws)
        elif version == 'w':
            ws = feature_1_mean
            cam_img = calc_cam(feature_conv, ws)
        elif version == 'w-diff-2':
            cam1 = calc_cam(feature_conv, feature_1_mean - feature_0_mean)
            cam2 = calc_cam(feature_conv, fea_weight * feature_1_mean)
            cam = cam1 * cam2
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
        elif version == 'w-diff':
            cam1 = calc_cam(feature_conv, feature_1_mean - feature_0_mean)
            cam2 = calc_cam(feature_conv, feature_1_mean)
            cam = cam1 * cam2
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
        elif version == 'joint': #w-diff & emb-diff
            cam1 = calc_cam(feature_conv, (feature_1_mean - feature_0_mean))
            cam2 = calc_cam(feature_conv, feature_1_mean)
            cam = cam1 * cam2
            cam = cam - np.min(cam)
            cam_img_1 = cam / np.max(cam)
            
            ws = positive_center - counter_center
            cam_img_2 = calc_cam(feature_conv, ws)
            
            cam = cam_img_1 * cam_img_2
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.sqrt(cam_img)
        #ws = np.abs(ws)
 
        output_cam.append(torch.from_numpy(cam_img).float())    

    return output_cam


# In[ ]:


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

i = -1
#example_ids = [4, 34, 41, 16,23,  48, 2, 3, 6, 15,24, 100,101,102,103,104,105,106,107,108,109,110]
example_ids = [x for x in range(len(demo_set))]
for i in example_ids:
    image, target_id, box = demo_set[i]
    if target_id != 1:
        assert "wrong example"

    rx = ceil(box[0]*img_size/Orig_img_size) if not np.isnan(box[0]) else 0
    ry = ceil(box[1]*img_size/Orig_img_size) if not np.isnan(box[1]) else 0
    rw = ceil(box[2]*img_size/Orig_img_size) if not np.isnan(box[2]) else 0
    rh = ceil(box[3]*img_size/Orig_img_size) if not np.isnan(box[3]) else 0
    box_pos = rx,ry,rw,rh

    bbox_coords = [int(x) for x in box]
    
    feature_extractor = FeatureExtractor(model)
    feature_extractor.register_hooks([last_conv_layer_name])  # Adjust based on the last conv layer name
        # Forward pass to get features and predictions
    output = model(image.cuda())
    probs = F.softmax(output, dim = 1).data.squeeze()

    # Obtain feature maps from the last convolutional layer
    feature_conv = feature_extractor.get_features()[-1].cpu().detach().numpy()
    embedding = feature_extractor.get_embedding()[0].detach()
    embedding = embedding / embedding.norm()
    embedding = embedding.cpu().numpy()
    #CAMs = DiffCAM(feature_conv, embedding, 0)

    iou = cal_iou(bbox_coords, results_grad[i])
    iou_gradcam.append(iou)
    iou = cal_iou(bbox_coords, results_sg[i])
    iou_sgcam.append(iou)
    iou = cal_iou(bbox_coords, results_score[i])
    iou_scorecam.append(iou)

    auprc = cal_auprc(bbox_coords, results_grad[i])
    auprc_gradcam.append(auprc)
    auprc = cal_auprc(bbox_coords, results_sg[i])
    auprc_sgcam.append(auprc)
    auprc = cal_auprc(bbox_coords, results_score[i])
    auprc_scorecam.append(auprc)


    N = 5
    if True:
        # visualization
        image = image.squeeze(0).cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        results.append((image, f'Image_{i} p={probs[1].item():.3f}', box_pos))
        
        ret = overlay_mask(to_pil_image(image), to_pil_image(results_grad[i], mode='F'), alpha=0.5)
        iou = iou_gradcam[-1]
        auprc = auprc_gradcam[-1]
        results.append((ret, f'GradCAM {iou:.2f}', box_pos))
            
        ret = overlay_mask(to_pil_image(image), to_pil_image(results_score[i], mode='F'), alpha=0.5)
        iou = iou_scorecam[-1]
        auprc = auprc_scorecam[-1]
        results.append((ret, f'ScoreCAM {iou:.2f}', box_pos))

        ref_group = get_reference_group(image, embedding)

        if True:
            version = 'w-diff'
            CAMs = DiffCAM_v3(feature_conv, embedding, ref_group, version)
            ret = overlay_mask(to_pil_image(image), to_pil_image(CAMs[0], mode='F'), alpha=0.5)            
            iou = cal_iou(bbox_coords, CAMs[0])
            auprc = cal_auprc(bbox_coords, CAMs[0])
            results.append((ret, f'DiffCAM {iou:.2f}', box_pos))        

            iou_diffcam.append(iou)
            auprc_diffcam.append(auprc)

    feature_extractor.remove_hooks()

    if i%50 == 0:
        print('auprc', i, np.mean(auprc_gradcam), np.mean(auprc_sgcam), np.mean(auprc_scorecam), np.mean(auprc_diffcam))
        print('iou', i, np.mean(iou_gradcam), np.mean(iou_sgcam), np.mean(iou_scorecam), np.mean(iou_diffcam))


