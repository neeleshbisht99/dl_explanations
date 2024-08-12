import matplotlib.pyplot as plt
import numpy as np

from utils import CommonUtils
from cams.resnet_cams import GradCam, DiffCAM, CounterFactual, TorchCAM

#TODO: move to Config 
target_class_idx = 1  # Target class
ref_class_idx = 0  # Reference class
last_conv_layer_name = 'layer2'
last_conv_layer_name_with_module = 'module.layer2'

def print_cam(heatmap):
    plt.matshow(np.squeeze(heatmap[:, :, 3]))
    plt.show()

def get_cams(model, test_volume, heatmap_shape):
    
    #FIXME the score cam code dosen't works as of now.
    # torch_cam_score_cam_heatmap = TorchCAM.compute_score_cam(test_volume, model, last_conv_layer_name_with_module, target_class_idx)
    # plt.matshow(np.squeeze(torch_cam_score_cam_heatmap[:, :, 3]))
    # plt.show()

    torch_cam_grad_cam_heatmap = TorchCAM.compute_grad_cam(test_volume, model, last_conv_layer_name_with_module, target_class_idx)
    # print_cam(torch_cam_grad_cam_heatmap)

    torch_cam_grad_campp_heatmap = TorchCAM.compute_grad_campp(test_volume, model, last_conv_layer_name_with_module, target_class_idx)
    # print_cam(torch_cam_grad_campp_heatmap)

    # Generate grad and diff-grad class activation heatmap
    grad_cam_heatmap, _,  diff_grad_cam_heatmap= GradCam.compute(test_volume, model, last_conv_layer_name, target_class_idx, ref_class_idx)
    # print_cam(grad_cam_heatmap)
    # print_cam(diff_grad_cam_heatmap)

    # Generate diff class activation heatmap
    diff_cam_heatmap = DiffCAM.make_diffcam_heatmap(test_volume, model, target_class_idx, ref_class_idx)
    # print_cam(diff_cam_heatmap)

    #Generate counter factual heatmap
    counter_factual_heatmap = CounterFactual.make_counter_factual_heatmap(test_volume, model, target_class_idx, ref_class_idx)
    # print_cam(counter_factual_heatmap)

    # Resize heatmap
    grad_cam_heatmap = CommonUtils.get_resized_heatmap(grad_cam_heatmap, heatmap_shape)
    diff_grad_cam_heatmap = CommonUtils.get_resized_heatmap(diff_grad_cam_heatmap, heatmap_shape)
    diff_cam_heatmap = CommonUtils.get_resized_heatmap(diff_cam_heatmap, heatmap_shape)
    counter_factual_heatmap = CommonUtils.get_resized_heatmap(counter_factual_heatmap, heatmap_shape)
    torch_cam_grad_cam_heatmap = CommonUtils.get_resized_heatmap(torch_cam_grad_cam_heatmap, heatmap_shape)
    torch_cam_grad_campp_heatmap = CommonUtils.get_resized_heatmap(torch_cam_grad_campp_heatmap, heatmap_shape)
    # torch_cam_score_cam_heatmap = CommonUtils.get_resized_heatmap(torch_cam_score_cam_heatmap, heatmap_shape)

    return grad_cam_heatmap, diff_grad_cam_heatmap, diff_cam_heatmap, counter_factual_heatmap, torch_cam_grad_cam_heatmap, torch_cam_grad_campp_heatmap


def print_cams_series(test_item, test_item_gt_mask, grad_cam_heatmap, diff_grad_cam_heatmap, diff_cam_heatmap, counter_factual_heatmap, torch_cam_grad_cam_heatmap, torch_cam_grad_campp_heatmap):
    # Set non-1 values to NaN for transparency
    mask_test = np.where(test_item_gt_mask == 1, 1, np.nan) 

    indices = np.where(mask_test == 1)
    indices_list = list(zip(*indices))
    arr = [i[2] for i in indices_list]

    # Find the range of the slices
    min_slice = np.min(np.unique(arr))
    max_slice = np.max(np.unique(arr))

    # Create a single figure with subplots
    fig, ax = plt.subplots(max_slice - min_slice + 1, 8, figsize=(30, 5 * (max_slice - min_slice + 1)))

    # Loop through each slice and plot the images in the appropriate subplot
    for i in range(min_slice, max_slice + 1):
        slice_idx = i - min_slice

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

        ax[slice_idx, 0].imshow(np.squeeze(test_item[:, :, i]), origin='upper', cmap='bone')
        ax[slice_idx, 1].imshow(np.squeeze(mask_test[:, :, i]), origin='upper', cmap='spring')

        img3 = ax[slice_idx, 2].imshow(np.squeeze(test_item[:, :, i]), cmap='bone')
        img4 = ax[slice_idx, 2].imshow(np.squeeze(grad_cam_heatmap[:, :, i]), cmap='jet', alpha=0.5, extent=img3.get_extent())
        img5 = ax[slice_idx, 2].imshow(np.squeeze(mask_test[:, :, i]), cmap='spring', alpha=0.7, extent=img3.get_extent())

        img6 = ax[slice_idx, 3].imshow(np.squeeze(test_item[:, :, i]), cmap='bone')
        img7 = ax[slice_idx, 3].imshow(np.squeeze(diff_grad_cam_heatmap[:, :, i]), cmap='jet', alpha=0.5, extent=img6.get_extent())
        img8 = ax[slice_idx, 3].imshow(np.squeeze(mask_test[:, :, i]), cmap='spring', alpha=0.7, extent=img6.get_extent())

        img9 = ax[slice_idx, 4].imshow(np.squeeze(test_item[:, :, i]), cmap='bone')
        img10 = ax[slice_idx, 4].imshow(np.squeeze(diff_cam_heatmap[:, :, i]), cmap='jet', alpha=0.5, extent=img9.get_extent())
        img11 = ax[slice_idx, 4].imshow(np.squeeze(mask_test[:, :, i]), cmap='spring', alpha=0.7, extent=img9.get_extent())

        img12 = ax[slice_idx, 5].imshow(np.squeeze(test_item[:, :, i]), cmap='bone')
        img13 = ax[slice_idx, 5].imshow(np.squeeze(counter_factual_heatmap[:, :, i]), cmap='jet', alpha=0.5, extent=img12.get_extent())
        img14 = ax[slice_idx, 5].imshow(np.squeeze(mask_test[:, :, i]), cmap='spring', alpha=0.7, extent=img12.get_extent())

        img15 = ax[slice_idx, 6].imshow(np.squeeze(test_item[:, :, i]), cmap='bone')
        img16 = ax[slice_idx, 6].imshow(np.squeeze(torch_cam_grad_cam_heatmap[:, :, i]), cmap='jet', alpha=0.5, extent=img15.get_extent())
        img17 = ax[slice_idx, 6].imshow(np.squeeze(mask_test[:, :, i]), cmap='spring', alpha=0.7, extent=img15.get_extent())

        img18 = ax[slice_idx, 7].imshow(np.squeeze(test_item[:, :, i]), cmap='bone')
        img19 = ax[slice_idx, 7].imshow(np.squeeze(torch_cam_grad_campp_heatmap[:, :, i]), cmap='jet', alpha=0.5, extent=img18.get_extent())
        img20 = ax[slice_idx, 7].imshow(np.squeeze(mask_test[:, :, i]), cmap='spring', alpha=0.7, extent=img18.get_extent())

    # Adjust the layout
    plt.tight_layout()
    plt.show()