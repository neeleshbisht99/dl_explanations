#!/usr/bin/env python
# coding: utf-8

# In[1]:
# [1]
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


# In[9]:
# [9]

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

    for method in x_methods:
        iou = cal_iou(bbox_coords, method['results'][i])
        method['iou'].append(iou)

        auprc = cal_auprc(bbox_coords, method['results'][i])
        method['auprc'].append(auprc)

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

    if i%50 == 0:
        avg_auprc_str = ''
        avg_iou_str = ''
        for method in x_methods:
            avg_auprc_str += f'{method["title"]}: {np.mean(method["auprc"])}, '
            avg_iou_str += f'{method["title"]}: {np.mean(method["iou"])}, '

        print('auprc', i, avg_auprc_str, 'Diff-CAM: ', np.mean(auprc_diffcam))
        print('iou', i, avg_iou_str, 'Diff-CAM: ', np.mean(iou_diffcam))

# auprc 0 Grad-CAM: 0.8072348517914129, Smooth Grad-CAM++: 0.16709032554442166, Score-CAM: 0.8285882905283114, Layer-CAM: 0.8143598058950382, XGrad-CAM: 0.8072348517914129,  Diff-CAM:  0.8543154337046005
# iou 0 Grad-CAM: 0.5742155893194282, Smooth Grad-CAM++: 0.0010576565760512249, Score-CAM: 0.5905168498501226, Layer-CAM: 0.5779039380012616, XGrad-CAM: 0.5742155893194282,  Diff-CAM:  0.6184490248636658
# auprc 50 Grad-CAM: 0.26372263126189954, Smooth Grad-CAM++: 0.10106938652065262, Score-CAM: 0.24348546717957295, Layer-CAM: 0.26196656542184193, XGrad-CAM: 0.263722677908879,  Diff-CAM:  0.31652160722609546
# iou 50 Grad-CAM: 0.1551745247521576, Smooth Grad-CAM++: 0.03786622970466809, Score-CAM: 0.1345478524926644, Layer-CAM: 0.14660564008226654, XGrad-CAM: 0.1551745247521576,  Diff-CAM:  0.18699625616684715
# auprc 100 Grad-CAM: 0.2810173569773482, Smooth Grad-CAM++: 0.09502221932492572, Score-CAM: 0.26728555042414637, Layer-CAM: 0.2913119955465014, XGrad-CAM: 0.28101739366592016,  Diff-CAM:  0.3183847006729431
# iou 100 Grad-CAM: 0.1689094135818027, Smooth Grad-CAM++: 0.031951302011036006, Score-CAM: 0.1508217947342634, Layer-CAM: 0.166694337001311, XGrad-CAM: 0.1689094135818027,  Diff-CAM:  0.19083911847050256
# auprc 150 Grad-CAM: 0.29449671242765146, Smooth Grad-CAM++: 0.09210305202887568, Score-CAM: 0.27668786065433343, Layer-CAM: 0.2995729362674984, XGrad-CAM: 0.29449680104474607,  Diff-CAM:  0.32592325470660805
# iou 150 Grad-CAM: 0.17700373279344903, Smooth Grad-CAM++: 0.027511091075454183, Score-CAM: 0.15773472501672736, Layer-CAM: 0.1752407360589368, XGrad-CAM: 0.17700399611054593,  Diff-CAM:  0.19599808294238308
# auprc 200 Grad-CAM: 0.30166710130026414, Smooth Grad-CAM++: 0.09120653633202871, Score-CAM: 0.26702964582620936, Layer-CAM: 0.2891639940963609, XGrad-CAM: 0.3016671775681278,  Diff-CAM:  0.32740528999971436
# iou 200 Grad-CAM: 0.18156280065934646, Smooth Grad-CAM++: 0.02560335060662474, Score-CAM: 0.15121768923755674, Layer-CAM: 0.16793379337669523, XGrad-CAM: 0.181562998474678,  Diff-CAM:  0.19943176258006085
# auprc 250 Grad-CAM: 0.2892489103355009, Smooth Grad-CAM++: 0.0914362214678514, Score-CAM: 0.25542098461306, Layer-CAM: 0.2741019352374753, XGrad-CAM: 0.28924896867668526,  Diff-CAM:  0.31111796506602346
# iou 250 Grad-CAM: 0.17217247701544822, Smooth Grad-CAM++: 0.02621156431644834, Score-CAM: 0.14260768417201516, Layer-CAM: 0.15618088289151352, XGrad-CAM: 0.1721726354253352,  Diff-CAM:  0.18444380618032313
# auprc 300 Grad-CAM: 0.2935632807139464, Smooth Grad-CAM++: 0.09624631827917014, Score-CAM: 0.2575127271215545, Layer-CAM: 0.2747733946662712, XGrad-CAM: 0.29356334136704026,  Diff-CAM:  0.31471814139793436
# iou 300 Grad-CAM: 0.17537250955742228, Smooth Grad-CAM++: 0.029057178586162115, Score-CAM: 0.1416177037252285, Layer-CAM: 0.15512701516912902, XGrad-CAM: 0.1753722983807373,  Diff-CAM:  0.18689815259027237
# auprc 350 Grad-CAM: 0.2829866780963207, Smooth Grad-CAM++: 0.09306519406026938, Score-CAM: 0.2527516551357237, Layer-CAM: 0.268784443496384, XGrad-CAM: 0.2829867171670279,  Diff-CAM:  0.3032797104314249
# iou 350 Grad-CAM: 0.16746671646078948, Smooth Grad-CAM++: 0.026497725567508817, Score-CAM: 0.1378264489185828, Layer-CAM: 0.1508462959743026, XGrad-CAM: 0.16746653536625333,  Diff-CAM:  0.17878815453830713
# auprc 400 Grad-CAM: 0.2793878136341015, Smooth Grad-CAM++: 0.09234718670567389, Score-CAM: 0.25348293759063617, Layer-CAM: 0.2685669666495066, XGrad-CAM: 0.27938784575521164,  Diff-CAM:  0.30560720065618835
# iou 400 Grad-CAM: 0.16349770539733924, Smooth Grad-CAM++: 0.025652158810305716, Score-CAM: 0.13810683199821056, Layer-CAM: 0.15001750396400187, XGrad-CAM: 0.16349897289732068,  Diff-CAM:  0.1793159835992595
# auprc 450 Grad-CAM: 0.27569939234020197, Smooth Grad-CAM++: 0.09173907893574271, Score-CAM: 0.25496008217018845, Layer-CAM: 0.26910267003177185, XGrad-CAM: 0.275699421793055,  Diff-CAM:  0.30312752831460194
# iou 450 Grad-CAM: 0.16062725472379916, Smooth Grad-CAM++: 0.025239585700745873, Score-CAM: 0.1394005608272255, Layer-CAM: 0.1505851033491198, XGrad-CAM: 0.1606283817027184,  Diff-CAM:  0.17732044422419474
# auprc 500 Grad-CAM: 0.27483159194333007, Smooth Grad-CAM++: 0.09450378108204635, Score-CAM: 0.25357113246228474, Layer-CAM: 0.2665133278127979, XGrad-CAM: 0.2748316189189283,  Diff-CAM:  0.3031336330000878
# iou 500 Grad-CAM: 0.15937559962489306, Smooth Grad-CAM++: 0.027393025965550728, Score-CAM: 0.13831867967924572, Layer-CAM: 0.14877880112150532, XGrad-CAM: 0.15937667112546763,  Diff-CAM:  0.17747462324118052
# auprc 550 Grad-CAM: 0.2707562550257422, Smooth Grad-CAM++: 0.09348105073316788, Score-CAM: 0.24904574185730796, Layer-CAM: 0.2628469622636898, XGrad-CAM: 0.27075627398923857,  Diff-CAM:  0.2992701351851674
# iou 550 Grad-CAM: 0.1564844506914083, Smooth Grad-CAM++: 0.027053974144060494, Score-CAM: 0.1341663718367715, Layer-CAM: 0.1454506912862781, XGrad-CAM: 0.15648493859265278,  Diff-CAM:  0.17488020184578254
# auprc 600 Grad-CAM: 0.2672708915121523, Smooth Grad-CAM++: 0.0939955933496972, Score-CAM: 0.2487566820029851, Layer-CAM: 0.26139569165844045, XGrad-CAM: 0.26727091535906394,  Diff-CAM:  0.296635504713437
# iou 600 Grad-CAM: 0.15330736654480004, Smooth Grad-CAM++: 0.027344359834183752, Score-CAM: 0.13399584330377903, Layer-CAM: 0.14444724341591927, XGrad-CAM: 0.15330781385525882,  Diff-CAM:  0.17243686319304727
# auprc 650 Grad-CAM: 0.27153545943194524, Smooth Grad-CAM++: 0.09357123231123148, Score-CAM: 0.24805015411484493, Layer-CAM: 0.26035963397284445, XGrad-CAM: 0.2715354813494833,  Diff-CAM:  0.29541340566899027
# iou 650 Grad-CAM: 0.1566322688416383, Smooth Grad-CAM++: 0.0266244592324207, Score-CAM: 0.1328614032606808, Layer-CAM: 0.14306898250531702, XGrad-CAM: 0.15663268179645506,  Diff-CAM:  0.17155867617006335
# auprc 700 Grad-CAM: 0.2710733778304988, Smooth Grad-CAM++: 0.09362578873505906, Score-CAM: 0.25428744590470387, Layer-CAM: 0.26780713761771197, XGrad-CAM: 0.2710733994472416,  Diff-CAM:  0.29987877600597224
# iou 700 Grad-CAM: 0.15661390911247086, Smooth Grad-CAM++: 0.027128038966888916, Score-CAM: 0.13775898911075268, Layer-CAM: 0.14865289396025616, XGrad-CAM: 0.15661429261259313,  Diff-CAM:  0.1748758341802891
# auprc 750 Grad-CAM: 0.2681863478041066, Smooth Grad-CAM++: 0.09378262911931329, Score-CAM: 0.2528754897586205, Layer-CAM: 0.2649368938229757, XGrad-CAM: 0.268186371738912,  Diff-CAM:  0.2953777908901789
# iou 750 Grad-CAM: 0.1543728006445626, Smooth Grad-CAM++: 0.02725486774063353, Score-CAM: 0.13712700711881973, Layer-CAM: 0.14668641962947082, XGrad-CAM: 0.15437315861205356,  Diff-CAM:  0.1713093765978311
# auprc 800 Grad-CAM: 0.2661320268803293, Smooth Grad-CAM++: 0.09446460801355445, Score-CAM: 0.2513865748815532, Layer-CAM: 0.2636516932582632, XGrad-CAM: 0.26613204918204614,  Diff-CAM:  0.2931498745340939
# iou 800 Grad-CAM: 0.15279562256147633, Smooth Grad-CAM++: 0.02790234590140136, Score-CAM: 0.1363147882397119, Layer-CAM: 0.14579694621891542, XGrad-CAM: 0.1527959581839304,  Diff-CAM:  0.1696477867844756
# auprc 850 Grad-CAM: 0.26836967177331383, Smooth Grad-CAM++: 0.09505497791647508, Score-CAM: 0.2524349387680279, Layer-CAM: 0.2661025257605312, XGrad-CAM: 0.2683696893655065,  Diff-CAM:  0.2949443035354901
# iou 850 Grad-CAM: 0.15495778950939473, Smooth Grad-CAM++: 0.0286947204773544, Score-CAM: 0.1372757190706984, Layer-CAM: 0.14761427974375352, XGrad-CAM: 0.1549581054125507,  Diff-CAM:  0.17130891032102152
# auprc 900 Grad-CAM: 0.2691238359130433, Smooth Grad-CAM++: 0.09473896971966818, Score-CAM: 0.25093393878604237, Layer-CAM: 0.2656222644271644, XGrad-CAM: 0.2691238542066994,  Diff-CAM:  0.29652519428707763
# iou 900 Grad-CAM: 0.15594312117353817, Smooth Grad-CAM++: 0.028571691952022037, Score-CAM: 0.1359947079406572, Layer-CAM: 0.1468880037950509, XGrad-CAM: 0.15594341954599736,  Diff-CAM:  0.17296831586232525
# auprc 950 Grad-CAM: 0.2684850891538493, Smooth Grad-CAM++: 0.09519733113300016, Score-CAM: 0.25180177129820797, Layer-CAM: 0.2666745321061579, XGrad-CAM: 0.26848510603577047,  Diff-CAM:  0.29789950888034233
# iou 950 Grad-CAM: 0.15544801695887334, Smooth Grad-CAM++: 0.029173560017907874, Score-CAM: 0.13641212144460615, Layer-CAM: 0.14727539350061497, XGrad-CAM: 0.15544829964403178,  Diff-CAM:  0.17338003655541714

# %%


# cp /shared/home/v_neelesh_bisht/local_scratch/dl_explanations/rsna/output/npy_v1_299/*.npy /tmp/rsna-dataset/