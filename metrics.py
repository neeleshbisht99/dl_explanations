import numpy as np

####################################### Calculate IOU #######################################
def calculate_iou_3d(pred_mask, gt_mask):
    """
    Calculate the Intersection over Union (IoU) of two 3D masks.
    
    Parameters:
    pred_mask: 3D numpy array of predicted mask (binary)
    gt_mask: 3D numpy array of ground truth mask (binary)
    
    Returns:
    float: IoU value
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    intersection_volume = np.sum(intersection)
    union_volume = np.sum(union)
    
    if union_volume == 0:
        return 0
    
    iou = intersection_volume / union_volume
    
    return iou



####################################### Calculate Localization Metric, Energy Proportion #######################################
def calculate_energy_proportion(saliency_map_3d, gt_mask):
    """
    Calculate the Localization metric, energy proportion.
    
    Parameters:
    saliency_map_3d: 3D numpy array of the saliency map.
    gt_mask: 3D numpy array of ground truth mask (binary)
    
    Returns:
    float: proportion value
    """
    # Point-wise multiply the saliency map with the mask
    masked_saliency_map = saliency_map_3d * gt_mask

    # Calculate the energy in the bounding box
    energy_in_bbox = np.sum(masked_saliency_map)

    # Calculate the total energy in the saliency map
    total_energy = np.sum(saliency_map_3d)

    # Calculate the proportion of energy in the bounding box
    proportion = energy_in_bbox / total_energy

    return proportion