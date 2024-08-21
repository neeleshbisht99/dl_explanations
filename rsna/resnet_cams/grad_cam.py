import torch
import numpy as np

class SaveFeatures:
    def __init__(self, model, last_conv_layer_name):
        self.activations = None
        self.gradients = None
        # Retrieve the last convolutional layer
        last_conv_layer = getattr(model, last_conv_layer_name)
        # Register hooks
        self.forward_handle = last_conv_layer.register_forward_hook(self.activation_hook)
        self.backward_handle = last_conv_layer.register_full_backward_hook(self.gradients_hook)
        
    def activation_hook(self, model, input, output):
        self.activations = output 
    
    def gradients_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        """Removes the hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

class GradCam:

    @staticmethod
    def compute_gradcam_weights(gradients):
        # Compute the weights (mean over the spatial dimensions)
        pooled_grads = np.mean(gradients, axis=(0, 2, 3))
        return pooled_grads
    
    @staticmethod
    def get_activations_and_gradients(img_tensor, model, last_conv_layer_name, pred_index=None):
        """Generate class activation heatmap"""
        
        save_features = SaveFeatures(model, last_conv_layer_name)

        # Make a forward pass to get predictions and activations
        model.eval()
        output = model(img_tensor)
        
        if pred_index is None:
            pred_index = torch.argmax(output, dim=1)
        class_output = output[:, pred_index].squeeze()
        
        # Backward pass to get gradients
        model.zero_grad()
        class_output.backward(retain_graph=True)

        activations = save_features.activations.detach().cpu().numpy()
        gradients = save_features.gradients.detach().cpu().numpy()

        # Remove hooks after usage
        save_features.remove()

        return activations, gradients

    @staticmethod
    def make_gradcam_heatmap(activations, weights):
        # Compute the heatmap
        activation_ = activations.copy()
        for i in range(weights.shape[0]):
            activation_[:, i, :, :] *= weights[i]
        heatmap = np.mean(activation_, axis=1).squeeze()
        
        heatmap = np.maximum(heatmap, 0) # Apply ReLU
        heatmap /= np.max(heatmap) # Normalize to [0, 1]
        return heatmap

    @staticmethod
    def compute(img_tensor, model, last_conv_layer_name, target_class_idx, ref_class_idx):
        # Get activations and gradients for target and reference class
        activations, target_class_gradients = GradCam.get_activations_and_gradients(img_tensor, model, last_conv_layer_name, pred_index=target_class_idx)
        _, ref_class_gradients = GradCam.get_activations_and_gradients(img_tensor, model, last_conv_layer_name, pred_index=ref_class_idx)

        # Compute gradient weights from the gradients
        target_class_weights = GradCam.compute_gradcam_weights(target_class_gradients)
        ref_class_weights = GradCam.compute_gradcam_weights(ref_class_gradients)

        # Compute the differential weights
        diff_weights = target_class_weights - ref_class_weights

        # Generate class activation heatmap
        diff_grad_cam_heatmap = GradCam.make_gradcam_heatmap(activations, diff_weights)
        grad_cam_heatmap = GradCam.make_gradcam_heatmap(activations, target_class_weights)
        ref_class_heatmap = GradCam.make_gradcam_heatmap(activations, ref_class_weights)

        # Return heatmaps for target class, reference class, and differential heatmap
        return grad_cam_heatmap, ref_class_heatmap, diff_grad_cam_heatmap
