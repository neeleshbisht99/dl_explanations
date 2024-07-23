import random
import numpy as np
import torch
from scipy import ndimage    
import torchio as tio

class Utils:

    # Define the training transformations
    train_transform = tio.Compose([
        tio.RandomAffine(
            scales=(0.8, 1.2),  # Scaling factor (0% ± 20%)
            degrees=10,  # Rotation (±10°)
            translation=(10, 10, 10),  # Horizontal and vertical translations (±10%)
            isotropic=False,  # Allow non-uniform scaling
            p=1.0  # Probability of applying the transform
        ),
        tio.RandomElasticDeformation(
            num_control_points=6,  # Number of control points for the deformation
            max_displacement=5,  # Maximum displacement of control points
            p=0.5  # Probability of applying the transform
        ),
        tio.RandomBiasField(p=0.2),  # Add bias field to simulate inhomogeneity
        tio.RandomNoise(mean=0, std=0.05, p=0.3),  # Add random noise
        tio.RandomGamma(p=0.3),  # Adjust gamma (brightness) randomly
        # tio.RandomContrast(p=0.3)  # Adjust contrast randomly
    ])
    
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
        volume = volume.permute(2, 1, 0)
        # Add a channel dimension
        volume = volume.unsqueeze(0)  # Add channel dimension at axis 0

        # Data augmentation #augmentation-change-experiment
        image = tio.ScalarImage(tensor=volume)
        image = Utils.train_transform(image)
        volume = image.tensor
        return volume # channel, depth, height, width

    @staticmethod
    def validation_preprocessing(volume):
        """Process validation data by only adding a channel."""
        volume = volume.permute(2, 1, 0)
        # Add a channel dimension
        volume = volume.unsqueeze(0)  # Add channel dimension at axis 0
        return volume # channel, depth, height, width

    @staticmethod
    def shuffle(arr1, arr2):
        #Shuffle train paths and labels together
        combined = list(zip(arr1, arr2))
        random.shuffle(combined)
        arr1, arr2 = zip(*combined)
        arr1 = list(arr1)
        arr2 = list(arr2)
        return arr1, arr2

