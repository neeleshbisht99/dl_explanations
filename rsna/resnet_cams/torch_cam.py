### This file contains the CAM methods based on "torch-cam" library.
from torchcam.methods import GradCAM, GradCAMpp, ScoreCAM
import numpy as np
import torch

class TorchCAM():

    @staticmethod
    def compute_grad_cam(img_tensor, model, last_conv_layer_name='module.layer4', target_class_idx=1):
        model.eval()
        grad_cam = GradCAM(model, last_conv_layer_name)
        output, _ = model(img_tensor)
        grad_cams = grad_cam(class_idx=target_class_idx, scores=output)
        heatmap = grad_cams[0].cpu().numpy().squeeze()
        heatmap = np.transpose(heatmap, (2, 1, 0))
        return heatmap
    
    @staticmethod
    def compute_grad_campp(img_tensor, model, last_conv_layer_name='module.layer4', target_class_idx=1):
        model.eval()
        grad_campp = GradCAMpp(model, last_conv_layer_name)
        output, _ = model(img_tensor)
        grad_campps = grad_campp(class_idx=target_class_idx, scores=output)
        heatmap = grad_campps[0].cpu().numpy().squeeze()
        heatmap = np.transpose(heatmap, (2, 1, 0))
        return heatmap
    
    @staticmethod
    def compute_score_cam(img_tensor, model, last_conv_layer_name='module.layer4', target_class_idx=1):
        model.eval()
        score_cam = ScoreCAM(model, last_conv_layer_name)
        with torch.no_grad():
            output, _ = model(img_tensor)
        score_cams = score_cam(class_idx=target_class_idx)
        heatmap = score_cams[0].cpu().numpy().squeeze()
        heatmap = np.transpose(heatmap, (2, 1, 0))
        return heatmap