import random
import numpy as np
import os
import torch
from scipy import ndimage    

class Utils:
    @staticmethod
    def rotate(volume):
        """Rotate the volume by a few degrees using PyTorch and SciPy"""
        def scipy_rotate(volume):
            # define some rotation angles
            angles = [-20, -10, -5, 5, 10, 20]
            # pick an angle at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            return volume
        
        # Convert the tensor to a numpy array
        volume_np = volume.numpy()
        # Apply the rotation using SciPy
        rotated_volume = scipy_rotate(volume_np)
        # Convert the numpy array back to a tensor
        return torch.tensor(rotated_volume, dtype=torch.float32)

    @staticmethod
    def train_preprocessing(volume):
        """Process training data by rotating and adding a channel."""
        # Rotate volume
        volume = Utils.rotate(volume)
        volume = volume.permute(2, 0, 1)
        # Add a channel dimension
        volume = volume.unsqueeze(0)  # Add channel dimension at axis 0
        return volume # channel, depth, width, height

    @staticmethod
    def validation_preprocessing(volume):
        """Process validation data by only adding a channel."""
        volume = volume.permute(2, 0, 1)
        # Add a channel dimension
        volume = volume.unsqueeze(0)  # Add channel dimension at axis 0
        return volume # channel, depth, width, height
