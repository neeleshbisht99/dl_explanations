import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features_blobs = []
        self.hooks = []  # Store hooks to allow removal

    def hook_fn(self, module, input, output):
        self.features_blobs.append(output)

    def register_hooks(self, layer_names):
        for name, layer in self.model.named_modules():
            if name in layer_names:
                hook = layer.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)

    def get_features(self):
        return self.features_blobs

    def remove_hooks(self):
        """Removes all hooks after they have been used."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class DiffCAM:
    @staticmethod
    def diffCAM(feature_conv, weight_softmax, target_class_idx, ref_class_idx):
        _, nc, h, w = feature_conv.shape
        # Compute weights for target and reference classes
        ws = weight_softmax[target_class_idx] - weight_softmax[ref_class_idx]
        # Reshape weights for matrix multiplication
        ws = ws.reshape(1, nc)
        # Flatten the feature maps
        feature_conv = feature_conv.reshape((nc, h * w))
        # Compute heatmap
        heatmap = ws.dot(feature_conv)
        heatmap = heatmap.reshape(h, w)
        # Normalize the heatmap
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        return heatmap # The heatmap should already be in (height, width) format

    @staticmethod
    def make_diffcam_heatmap(img_tensor, model, last_conv_layer_name, target_class_idx, ref_class_idx): 
        """Generate differential class activation heatmap for 2D CNN model"""
        
        # Register hooks to capture feature maps
        feature_extractor = FeatureExtractor(model)
        feature_extractor.register_hooks([last_conv_layer_name])  # Adjust based on the last conv layer name

        model.eval()

        # Forward pass to get features and predictions
        # with torch.no_grad(): # use if memory issues are there
        output = model(img_tensor)
        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        # Obtain feature maps from the last convolutional layer
        feature_conv = feature_extractor.get_features()[-1].cpu().detach().numpy()
        heatmap = DiffCAM.diffCAM(feature_conv, weight_softmax, target_class_idx, ref_class_idx)

        # Remove hooks after obtaining the features
        feature_extractor.remove_hooks()
        
        return heatmap
    