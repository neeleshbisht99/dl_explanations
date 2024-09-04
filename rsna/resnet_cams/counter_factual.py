import numpy as np
import torch
import torch.nn.functional as F

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features_blobs = []
        self.hooks = []  # Store hook handles for removal

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
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class CounterFactual:
    @staticmethod
    def returnCF(feature_conv, weight_softmax, confidence_score, target_class_idx, ref_class_idx):
        _, nc, h, w = feature_conv.shape
        
        feature_conv = feature_conv.reshape((nc, h * w))
        
        # Attribution map for target class
        ws = weight_softmax[target_class_idx]
        heatmap = ws.dot(feature_conv)
        heatmap = heatmap.reshape(h, w)
        
        # Complement attribution map for reference class
        ws_ref = weight_softmax[ref_class_idx]
        heatmap_ref = ws_ref.dot(feature_conv)
        heatmap_ref = heatmap_ref.reshape(h, w)
        heatmap_ref = np.max(heatmap_ref) - heatmap_ref
        
        # Attribution map for the confidence score
        heatmap_conf = confidence_score * heatmap
        heatmap_conf = heatmap_conf.reshape(h, w)
        
        # Compute the discriminant map
        heatmap = heatmap * heatmap_ref * heatmap_conf
        
        # Normalize heatmap
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        return heatmap

    @staticmethod
    def make_counter_factual_heatmap(img_tensor, model, last_conv_layer_name, target_class_idx, ref_class_idx): 
        """Generate counterfactual heatmap for 2D CNN model"""
        
        # Register hooks to capture feature maps
        feature_extractor = FeatureExtractor(model)
        feature_extractor.register_hooks([last_conv_layer_name])  # Adjust based on the last conv layer name

        model.eval()
        # with torch.no_grad():  # use if memory issues are there # Avoid computing gradients
        output = model(img_tensor)
        output_probabilities = torch.softmax(output, dim=1)
        # Get the softmax weights
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        # Obtain feature maps from the last convolutional layer
        feature_conv = feature_extractor.get_features()[-1].cpu().detach().numpy()
        
        # Get confidence score
        probs = F.softmax(output, dim=1)
        confidence_score, predicted_class = torch.max(probs, dim=1)
        confidence_score = confidence_score.item()
        predicted_class = predicted_class.item()
        
        # Generate counterfactual heatmap
        heatmap = CounterFactual.returnCF(feature_conv, weight_softmax, confidence_score, target_class_idx, ref_class_idx)
        
        # Remove hooks after obtaining the features
        feature_extractor.remove_hooks()

        return heatmap, output_probabilities
