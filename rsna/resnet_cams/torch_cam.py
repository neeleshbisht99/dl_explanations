### This file contains the CAM methods based on "torch-cam" library.
from torchcam.methods import GradCAM, GradCAMpp, ScoreCAM
import numpy as np
import torch

class TorchCAM():

    @staticmethod
    def compute_grad_cam(img_tensor, model, last_conv_layer_name='layer4', target_class_idx=1):
        model.eval()
        grad_cam = GradCAM(model, last_conv_layer_name)
        output = model(img_tensor)
        output_probabilities = torch.softmax(output, dim=1)
        heatmaps = []
        for i in range(img_tensor.size(0)):
            grad_cams = grad_cam(class_idx=target_class_idx, scores=output[i:i+1])
            heatmap = grad_cams[0].cpu().numpy().squeeze()
            heatmaps.append(heatmap)
        return heatmaps, output_probabilities
    
    @staticmethod
    def compute_grad_campp(img_tensor, model, last_conv_layer_name='layer4', target_class_idx=1):
        model.eval()
        grad_campp = GradCAMpp(model, last_conv_layer_name)
        output = model(img_tensor)
        output_probabilities = torch.softmax(output, dim=1)
        heatmaps = []
        for i in range(img_tensor.size(0)):
            grad_campps = grad_campp(class_idx=target_class_idx, scores=output[i:i+1])
            heatmap = grad_campps[0].cpu().numpy().squeeze()
            heatmaps.append(heatmap)
        return heatmaps, output_probabilities
    
    @staticmethod
    def compute_score_cam(img_tensor, model, last_conv_layer_name='layer4', target_class_idx=1):
        model.eval()
        score_cam = ScoreCAM(model, last_conv_layer_name)
        with torch.no_grad(): # TODO: think on this
            output = model(img_tensor)
        heatmaps = []
        for i in range(img_tensor.size(0)):
            score_cams = score_cam(class_idx=target_class_idx)
            heatmap = score_cams[0].cpu().numpy().squeeze()
            heatmaps.append(heatmap)
        return heatmaps