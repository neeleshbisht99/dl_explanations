import numpy as np

class DiffCAM:
    @staticmethod
    def diffCAM(feature_conv, weight_softmax, target_class_idx, ref_class_idx):
        bz, nc, d, h, w = feature_conv.shape
        ws =  weight_softmax[target_class_idx] -  weight_softmax[ref_class_idx]
        ws = ws.reshape(1, 256)
        feature_conv = feature_conv.reshape((nc, d * h * w))
        heatmap = ws.dot(feature_conv)
        heatmap = heatmap.reshape(d, h, w)
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.transpose(heatmap, (1, 2, 0))
        return heatmap

    @staticmethod
    def make_diffcam_heatmap(img_tensor, model, target_class_idx, ref_class_idx): 
        """Generate differential class activation heatmap for 3D CNN model"""
        
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax_4 = np.squeeze(params[-4].data.cpu().numpy())
        weight_softmax_2 = np.squeeze(params[-2].data.cpu().numpy())
        weight_softmax = weight_softmax_2.dot(weight_softmax_4)

        model.eval()
        _, features_blobs = model(img_tensor)
        feature = features_blobs[-1].cpu().detach()
        
        heatmap = DiffCAM.diffCAM(feature, weight_softmax, target_class_idx, ref_class_idx)
        return heatmap
