from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import numpy as np
from .common_utils import CommonUtils

class AUPRC:
    @staticmethod
    def compute_auprc(bbox, heatmap):
        ground_truth_mask = CommonUtils.bbox_to_mask(bbox, heatmap.shape)
        saliency_map = heatmap / 255.0
        
        y_true = ground_truth_mask.flatten()  # Flatten to 1D   
        y_scores = saliency_map.flatten()  # Flatten to 1D

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Calculate AUPRC (Area Under Precision-Recall Curve)
        auprc = average_precision_score(y_true, y_scores)

        # print(f"AUPRC: {auprc:.4f}")
        return auprc
    
    @staticmethod
    def compute_auc(bbox, heatmap):
        ground_truth_mask = CommonUtils.bbox_to_mask(bbox, heatmap.shape)
        saliency_map = heatmap / 255.0

        y_true = ground_truth_mask.flatten()  # Flatten to 1D   
        y_scores = saliency_map.flatten()  # Flatten to 1D

        # Compute precision-recall curve
        auc_score = roc_auc_score(y_true, y_scores)

        # print(f"AUC Score: {auc_score:.4f}")
        return auc_score