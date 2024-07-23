import numpy as np
import torch
import torch.nn.functional as F

class CounterFactual:
    @staticmethod
    def returnCF(feature_conv, weight_softmax, confidence_score, target_class_idx, ref_class_idx):
        _, nc, d, h, w = feature_conv.shape
        
        feature_conv = feature_conv.reshape((nc, d * h * w))
        
        #attribution map for target class
        ws =  weight_softmax[target_class_idx]
        heatmap = ws.dot(feature_conv)
        heatmap = heatmap.reshape(d, h, w)
        
        #complement attribution map for ref class
        ws_ref = weight_softmax[ref_class_idx]
        heatmap_ref = ws_ref.dot(feature_conv)
        heatmap_ref = heatmap_ref.reshape(d, h, w)
        heatmap_ref = np.max(heatmap_ref) - heatmap_ref
        
        # Attribution map for the confidence score
        heatmap_conf = confidence_score * heatmap
        heatmap_conf = heatmap_conf.reshape(d, h, w)
            
        # Compute the discriminant map 
        heatmap = heatmap * heatmap_ref * heatmap_conf
        
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.transpose(heatmap, (2, 1, 0))
        return heatmap

    @staticmethod
    def make_counter_factual_heatmap(img_tensor, model, target_class_idx, ref_class_idx): 
        """Generate differential class activation heatmap for 3D CNN model"""
        
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        # get feature map
        model.eval()
        output, features_blobs = model(img_tensor)
        feature = features_blobs[-1].cpu().detach()
        
        # get confidence_score
        probs = F.softmax(output, dim=1)
        confidence_score, predicted_class = torch.max(probs, dim=1)
        confidence_score = confidence_score.item()
        predicted_class = predicted_class.item()
        
        heatmap = CounterFactual.returnCF(feature, weight_softmax, confidence_score, target_class_idx, ref_class_idx)
        return heatmap