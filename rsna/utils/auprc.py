from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
from .common_utils import CommonUtils

class AUPRC:
    @staticmethod
    def compute_auprc(bbox, heatmap):
        ground_truth_mask = CommonUtils.bbox_to_mask(bbox, heatmap.shape)
        saliency_map = np.uint8(heatmap / 255)
        # print("ground_truth_mask",ground_truth_mask.shape)
        # print("saliency_map",saliency_map.shape)
        y_true = ground_truth_mask.flatten()  # Flatten to 1D   
        y_scores = saliency_map.flatten()  # Flatten to 1D

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # Calculate AUPRC (Area Under Precision-Recall Curve)
        auprc = average_precision_score(y_true, y_scores)

        # print(f"AUPRC: {auprc:.4f}")
        return auprc