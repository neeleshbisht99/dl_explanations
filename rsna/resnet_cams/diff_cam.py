import numpy as np
import torch
import faiss  

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features_blobs = []
        self.hooks = []  # Store hooks to allow removal
        self.embedding = None

    def hook_fn(self, module, input, output):
        self.features_blobs.append(output)

    def _embedding_hook_fn(self, module, input, output):
        # This method will be called during the forward pass
        self.embedding = input[0]  # Storing the input to the fc layer

    def register_hooks(self, layer_names):
        for name, layer in self.model.named_modules():
            if name in layer_names:
                hook = layer.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)

        embedding_hook = self.model.fc.register_forward_hook(self._embedding_hook_fn)
        self.hooks.append(embedding_hook)

    def get_features(self):
        return self.features_blobs

    def get_embedding(self):
        return self.embedding

    def remove_hooks(self):
        """Removes all hooks after they have been used."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class DiffCAM:
    @staticmethod
    def diffCAM(feature_conv, embedding, target_class_idx, ref_class_idx, all_features, all_labels):
        _, nc, h, w = feature_conv.shape
        feature_dim = 2048
        K = 10
        N = 1
        index_flat = faiss.IndexFlatL2(feature_dim)  # build a flat (CPU) index
        index_flat.add(all_features) 
        D, I = index_flat.search(np.expand_dims(embedding, axis=0), K) # sanity check
        counter_center = None
        num = 0
        for i in range(1, K):
            idx = I[0, i]
            if all_labels[idx] != ref_class_idx:
                continue
            if counter_center is None:
                counter_center = all_features[idx]
            else:
                counter_center += all_features[idx]
            num += 1
            if num >= N:
                break
        # print("counter_center here", counter_center, num, all_features,all_labels )
        counter_center = counter_center / num if counter_center is not None and num else None
        ws = embedding - counter_center if counter_center is not None else embedding
        # Flatten the feature maps
        feature_conv = feature_conv.reshape((nc, h * w))
        # Compute heatmap
        heatmap = ws.dot(feature_conv)
        heatmap = heatmap.reshape(h, w)
        # Normalize the heatmap
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        return heatmap # The heatmap should already be in (height, width) format

    @staticmethod
    def make_diffcam_heatmap(img_tensor, model, last_conv_layer_name, target_class_idx, ref_class_idx, all_features, all_labels): 
        """Generate differential class activation heatmap for 2D CNN model"""
        
        # Register hooks to capture feature maps
        feature_extractor = FeatureExtractor(model)
        feature_extractor.register_hooks([last_conv_layer_name])  # Adjust based on the last conv layer name

        model.eval()

        # Forward pass to get features and predictions
        output = model(img_tensor)
        output_probabilities = torch.softmax(output, dim=1)

        # Obtain feature maps from the last convolutional layer
        feature_conv = feature_extractor.get_features()[-1].cpu().detach().numpy()
        embedding = feature_extractor.get_embedding()[0].detach().cpu().numpy()
        heatmap = DiffCAM.diffCAM(feature_conv, embedding, target_class_idx, ref_class_idx,  all_features, all_labels)

        # Remove hooks after obtaining the features
        feature_extractor.remove_hooks()
        
        return heatmap, output_probabilities


########################## OLD diff cam algorithm
    # @staticmethod
    # def diffCAM(feature_conv, weight_softmax, target_class_idx, ref_class_idx):
    #     _, nc, h, w = feature_conv.shape
    #     # Compute weights for target and reference classes
    #     ws = weight_softmax[target_class_idx] - weight_softmax[ref_class_idx]
    #     # Reshape weights for matrix multiplication
    #     ws = ws.reshape(1, nc)
    #     # Flatten the feature maps
    #     feature_conv = feature_conv.reshape((nc, h * w))
    #     # Compute heatmap
    #     heatmap = ws.dot(feature_conv)
    #     heatmap = heatmap.reshape(h, w)
    #     # Normalize the heatmap
    #     heatmap = heatmap - np.min(heatmap)
    #     heatmap = heatmap / np.max(heatmap)
    #     return heatmap # The heatmap should already be in (height, width) format

    # @staticmethod
    # def make_diffcam_heatmap(img_tensor, model, last_conv_layer_name, target_class_idx, ref_class_idx): 
    #     """Generate differential class activation heatmap for 2D CNN model"""
        
    #     # Register hooks to capture feature maps
    #     feature_extractor = FeatureExtractor(model)
    #     feature_extractor.register_hooks([last_conv_layer_name])  # Adjust based on the last conv layer name

    #     model.eval()

    #     # Forward pass to get features and predictions
    #     # with torch.no_grad(): # use if memory issues are there
    #     output = model(img_tensor)
    #     # get the softmax weight
    #     params = list(model.parameters())
    #     weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    #     # Obtain feature maps from the last convolutional layer
    #     feature_conv = feature_extractor.get_features()[-1].cpu().detach().numpy()
    #     heatmap = DiffCAM.diffCAM(feature_conv, weight_softmax, target_class_idx, ref_class_idx)

    #     # Remove hooks after obtaining the features
    #     feature_extractor.remove_hooks()
        
    #     return heatmap
