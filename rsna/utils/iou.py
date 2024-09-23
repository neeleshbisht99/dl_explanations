####################################### Calculate IOU #######################################
import matplotlib.pyplot as plt
import numpy as np
from .common_utils import CommonUtils


Orig_img_size = 1024
img_size = 299

class IOU:

    @staticmethod
    def plot_heatmap_and_mask(binary_heatmap, bbox_mask):
        """Plot binary heatmap and bounding box mask side by side."""
        
        # Plot heatmap and mask
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot binary heatmap
        ax[0].imshow(binary_heatmap, cmap='jet')
        ax[0].set_title('Binary Heatmap')
        ax[0].axis('off')
        
        # Plot bbox mask
        ax[1].imshow(bbox_mask, cmap='jet')
        ax[1].set_title('Bounding Box Mask')
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def compute_iou(bbox, heatmap, threshold=63):
        """Calculate the 2D IoU between bounding box and heatmap."""
        # Convert heatmap to binary mask using the threshold
        # print("heatmap", np.max(heatmap), np.min(heatmap))
        # print("bbox", bbox)
        binary_heatmap = (heatmap > threshold).astype(np.uint8)

        # Get the bounding box as a binary mask
        bbox_mask = CommonUtils.bbox_to_mask(bbox, heatmap.shape)

        # Calculate the number of pixels in the bbox_mask
        bbox_pixel_count = bbox_mask.sum()

        # Flatten heatmap and sort pixels by intensity (descending order)
        flat_heatmap = heatmap.flatten()
        sorted_indices = np.argsort(flat_heatmap)[::-1]  # Indices of pixels sorted by intensity (highest first)
        
        # Select top bbox_pixel_count pixels from the sorted heatmap
        selected_indices = sorted_indices[:bbox_pixel_count]

        # Create a mask from the selected indices
        selected_heatmap_mask = np.zeros_like(flat_heatmap)
        selected_heatmap_mask[selected_indices] = 1
        selected_heatmap_mask = selected_heatmap_mask.reshape(heatmap.shape)

        binary_heatmap = selected_heatmap_mask

        # print("shape",binary_heatmap.shape, bbox_mask.shape )
        # plot_heatmap_and_mask(binary_heatmap, bbox_mask)
        # Compute intersection and union
        intersection = np.logical_and(binary_heatmap, bbox_mask).sum()
        union = np.logical_or(binary_heatmap, bbox_mask).sum()
        
        # Calculate IoU
        iou = intersection / union if union != 0 else 0
        # print("iou",iou)
        return iou