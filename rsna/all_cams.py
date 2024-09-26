import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('/tmp/local_scratch/v_neelesh_bisht/3d-cnn/rsna')

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from math import ceil
import matplotlib.patches as patches
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score
from datetime import datetime
from torch.utils import data

from resnet_cams import ResnetCAMS
from resnet import Resnet
from utils import IOU, AUPRC

print("intial allocation",torch.cuda.memory_allocated())
Orig_img_size = 1024
img_size = 299
# last conv_block : model.Mixed_7c


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


batch_size = 64

print("START: dataset prep")
val_dataset = torch.load('./rsna-dataset/presaved-dataset/val_dataset.pth')
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torch.load('./rsna-dataset/presaved-dataset/test_dataset.pth')
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print("DONE: dataset prep")

target_class_idx = 1  # Target class
ref_class_idx = 0  # Reference class


# Load the model weights
resnet_model = Resnet(device=device, path='./rsna-dataset/model_inception_v3_24092024_dict.pth')
print("DONE: model prep")

print("START: validation")
final_label_arr = [] # value for all_labels
final_feature_arr = [] # value for all_features

embedding = None

def _embedding_hook_fn(module, input, output):
    global embedding 
    embedding = input[0]  # Storing the input to the fc layer

embedding_hook = resnet_model.model.fc.register_forward_hook(_embedding_hook_fn)


resnet_model.model.eval()
correct = 0
total = 0  
with torch.no_grad():
    for images, labels, _ in tqdm(val_loader):
        # images = images.to(device) ## Will already go to device in the call function
        labels = labels.to(device)
        predictions = resnet_model(images)
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()

        for i in range(images.shape[0]):
            class_id = labels[i].cpu().item()
            final_label_arr.append(class_id)

            feature = embedding[i].detach().cpu().numpy()
            if len(final_feature_arr) == 0:
                final_feature_arr = np.expand_dims(feature, axis=0)
            else:
                final_feature_arr = np.concatenate((final_feature_arr, np.expand_dims(feature, axis=0)), axis = 0)
        
        # Delete tensors explicitly after use
        # print(torch.cuda.memory_allocated())
        del images, labels, predictions, predicted, feature
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())

print(f'Val_Acc: {100*correct/total}')

embedding_hook.remove()

print("DONE: validation")



print("START: cams, iou and auprc evaluation")
last_conv_layer_name = 'Mixed_7c'
methods = ["grad_cam_heatmap", "diff_grad_cam_heatmap", "diff_cam_heatmap", "counter_factual_heatmap", "torch_cam_grad_cam_heatmap", "torch_cam_grad_campp_heatmap", "torch_cam_score_cam_heatmap"]

final_image_arr = []
final_ground_truth_label_arr = []
final_bbox_arr = []
final_heatmap_arr = {method: [] for method in methods}
final_cam_wise_probability_arr = {method: [] for method in methods}
heatmap_iou_obj = {method: [] for method in methods}
heatmap_auprc_obj = {method: [] for method in methods}
heatmap_auc_obj = {method: [] for method in methods}

np.set_printoptions(precision=4, suppress=True)

for images, labels, bboxs in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    
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

        final_image_arr.append(image.cpu())
        final_ground_truth_label_arr.append(label)
        final_bbox_arr.append(bbox_coords)

        cams = ResnetCAMS.get_cams(image, resnet_model.model, last_conv_layer_name, target_class_idx, ref_class_idx, final_feature_arr=final_feature_arr, final_label_arr=final_label_arr)

        for type in methods:
            heatmap = cams[type]["heatmap"]
            iou = IOU.compute_iou(bbox_coords, heatmap)
            auprc = AUPRC.compute_auprc(bbox_coords, heatmap)
            auc = AUPRC.compute_auc(bbox_coords, heatmap)
            
            output_probs = cams[type]["output"].cpu().detach().numpy()[0]
            output_probs = np.round(output_probs, 4)

            final_heatmap_arr[type].append(heatmap)
            heatmap_iou_obj[type].append(iou)
            heatmap_auprc_obj[type].append(auprc)
            heatmap_auc_obj[type].append(auc)
            final_cam_wise_probability_arr[type].append(output_probs)
        
        # Explicitly delete tensors to free GPU memory
        del image, heatmap, output_probs
        torch.cuda.empty_cache() 
        
    # Delete images and labels after processing the batch
    del images, labels
    torch.cuda.empty_cache()

print("DONE: cams, iou and auprc evaluation")


print("START: cams output creation")
numrows = 90 # len(final_image_arr)

# Create a single figure with subplots
fig, ax = plt.subplots(numrows, 8, figsize=(30, 5 * (numrows + 1)))

# Loop through each slice and plot the images in the appropriate subplot
for slice_idx in range(0, numrows):

    # Adding titles for the first row
    if slice_idx == 0:
        ax[slice_idx, 0].set_title('Sample')
        ax[slice_idx, 1].set_title('Mask')
        ax[slice_idx, 2].set_title('Grad-CAM')
        ax[slice_idx, 3].set_title('Diff Grad-CAM')
        ax[slice_idx, 4].set_title('Diff-CAM')
        ax[slice_idx, 5].set_title('Counter Factual')
        ax[slice_idx, 6].set_title('Torch-CAM Grad-CAM')
        ax[slice_idx, 7].set_title('Torch-CAM Grad-CAM++')

    final_image = final_image_arr[slice_idx]
    final_image = final_image.cpu().numpy()
    final_image = final_image[0]
    img = np.transpose(final_image, (1, 2, 0))

    box = final_bbox_arr[slice_idx]
    # 'r' means relative. 'c' means center.
    rx = ceil(box[0]*img_size/Orig_img_size) if not np.isnan(box[0]) else 0
    ry = ceil(box[1]*img_size/Orig_img_size) if not np.isnan(box[1]) else 0
    rw = ceil(box[2]*img_size/Orig_img_size) if not np.isnan(box[2]) else 0
    rh = ceil(box[3]*img_size/Orig_img_size) if not np.isnan(box[3]) else 0

    grad_cam_heatmap = final_heatmap_arr["grad_cam_heatmap"][slice_idx]
    diff_grad_cam_heatmap = final_heatmap_arr["diff_grad_cam_heatmap"][slice_idx]
    diff_cam_heatmap = final_heatmap_arr["diff_cam_heatmap"][slice_idx]
    counter_factual_heatmap = final_heatmap_arr["counter_factual_heatmap"][slice_idx]
    torch_cam_grad_cam_heatmap = final_heatmap_arr["torch_cam_grad_cam_heatmap"][slice_idx]
    torch_cam_grad_campp_heatmap = final_heatmap_arr["torch_cam_grad_campp_heatmap"][slice_idx]
    # grad_cam_heatmap = final_heatmap_arr["grad_cam_heatmap"][slice_idx]

    ax[slice_idx, 0].imshow(img, origin='upper', cmap='bone')
    ax[slice_idx, 0].set_xlabel(f'Image {slice_idx + 1}, gt: {final_ground_truth_label_arr[slice_idx]}')

    # TODO: FIT SCORE CAM HERE
    # ax[slice_idx, 1].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))

    img3 = ax[slice_idx, 2].imshow(img, cmap='bone')
    img4 = ax[slice_idx, 2].imshow(grad_cam_heatmap, cmap='jet', alpha=0.5, extent=img3.get_extent())
    img5 = ax[slice_idx, 2].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))
    ax[slice_idx, 2].set_xlabel(f'''Grad-CAM, iou: {heatmap_iou_obj["grad_cam_heatmap"][slice_idx]},
    probs: {final_cam_wise_probability_arr["grad_cam_heatmap"][slice_idx]}, auprc: {heatmap_auprc_obj["grad_cam_heatmap"][slice_idx]} ''')

    img6 = ax[slice_idx, 3].imshow(img, cmap='bone')
    img7 = ax[slice_idx, 3].imshow(diff_grad_cam_heatmap, cmap='jet', alpha=0.5, extent=img6.get_extent())
    img8 = ax[slice_idx, 3].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))
    ax[slice_idx, 3].set_xlabel(f'''Diff Grad-CAM, iou: {heatmap_iou_obj["diff_grad_cam_heatmap"][slice_idx]},
    probs: {final_cam_wise_probability_arr["diff_grad_cam_heatmap"][slice_idx]}, auprc: {heatmap_auprc_obj["grad_cam_heatmap"][slice_idx]} ''')

    img9 = ax[slice_idx, 4].imshow(img, cmap='bone')
    img10 = ax[slice_idx, 4].imshow(diff_cam_heatmap, cmap='jet', alpha=0.5, extent=img9.get_extent())
    img11 = ax[slice_idx, 4].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))
    ax[slice_idx, 4].set_xlabel(f'''Diff-CAM, iou: {heatmap_iou_obj["diff_cam_heatmap"][slice_idx]},
    probs: {final_cam_wise_probability_arr["diff_cam_heatmap"][slice_idx]}, auprc: {heatmap_auprc_obj["grad_cam_heatmap"][slice_idx]} ''')

    img12 = ax[slice_idx, 5].imshow(img, cmap='bone')
    img13 = ax[slice_idx, 5].imshow(counter_factual_heatmap, cmap='jet', alpha=0.5, extent=img12.get_extent())
    img14 = ax[slice_idx, 5].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))
    ax[slice_idx, 5].set_xlabel(f'''Counter Factual, iou: {heatmap_iou_obj["counter_factual_heatmap"][slice_idx]},
    probs: {final_cam_wise_probability_arr["counter_factual_heatmap"][slice_idx]}, auprc: {heatmap_auprc_obj["grad_cam_heatmap"][slice_idx]} ''')

    img15 = ax[slice_idx, 6].imshow(img, cmap='bone')
    img16 = ax[slice_idx, 6].imshow(torch_cam_grad_cam_heatmap, cmap='jet', alpha=0.5, extent=img15.get_extent())
    img17 = ax[slice_idx, 6].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))
    ax[slice_idx, 6].set_xlabel(f'''Torch-CAM Grad-CAM, iou: {heatmap_iou_obj["torch_cam_grad_cam_heatmap"][slice_idx]},
    probs: {final_cam_wise_probability_arr["torch_cam_grad_cam_heatmap"][slice_idx]}, auprc: {heatmap_auprc_obj["grad_cam_heatmap"][slice_idx]} ''')

    img18 = ax[slice_idx, 7].imshow(img, cmap='bone')
    img19 = ax[slice_idx, 7].imshow(torch_cam_grad_campp_heatmap, cmap='jet', alpha=0.5, extent=img18.get_extent())
    img20 = ax[slice_idx, 7].add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor='none'))
    ax[slice_idx, 7].set_xlabel(f'''Torch-CAM Grad-CAM++, iou: {heatmap_iou_obj["torch_cam_grad_campp_heatmap"][slice_idx]},
    probs: {final_cam_wise_probability_arr["torch_cam_grad_campp_heatmap"][slice_idx]}, auprc: {heatmap_auprc_obj["grad_cam_heatmap"][slice_idx]} ''')

# Adjust the layout
plt.tight_layout()

# plt.show()

current_time = datetime.now()
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
# plt.savefig(f'output_image_{current_time_str}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'output_image_{current_time_str}.pdf', bbox_inches='tight')  # Use PDF for large images

print("DONE: cams output creation")


print("START: aggregated metrics calculation")
def get_correlation_score(arr1=[], arr2=[]):
    score = stats.pearsonr(arr1, arr2)
    return score.statistic

def compute_mean_iou_for_heatmaps(heatmap_iou_obj, heatmap_auprc_obj, heatmap_auc_obj):
    """Calculate the mean IoU for each type of heatmap."""
    mean_iou = {}

    # Iterate over each type of heatmap
    for tup1, tup2, tup3 in list(zip(heatmap_iou_obj.items(), heatmap_auprc_obj.items(), heatmap_auc_obj.items())):
        heatmap_type, iou_list = tup1
        _, auprc_list = tup2
        _, auc_list = tup3
        # Check if the IoU list is not empty
        probs = [item[1] for item in final_cam_wise_probability_arr[heatmap_type]]
        if iou_list:
            mean_iou[heatmap_type] = {'iou': np.mean(iou_list), 'auprc': np.mean(auprc_list), 'auc': np.mean(auc_list), 'correlation_score': get_correlation_score(iou_list, probs)}
        else:
            mean_iou[heatmap_type] = {'iou': None}  # Use None to indicate no IoU values available
    
    return mean_iou

mean_iou = compute_mean_iou_for_heatmaps(heatmap_iou_obj, heatmap_auprc_obj, heatmap_auc_obj)
print("DONE: aggregated metrics calculation")


for heatmap_type, obj in mean_iou.items():
    iou = obj["iou"]
    auprc = obj["auprc"]
    auc = obj["auc"]
    correlation_score = obj["correlation_score"]
    print(f"IOU for {heatmap_type}: ", iou*100,'%')
    print(f"AUPRC for {heatmap_type}: ", auprc)
    print(f"AUC for {heatmap_type}: ", auc)
    print(f"Correlation Score for {heatmap_type}: ", correlation_score)
    print('\n')

