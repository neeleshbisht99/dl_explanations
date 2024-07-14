import nibabel as nib
from scipy.ndimage import zoom
from scipy import ndimage

from config import config

class MaskDataset():
    @staticmethod
    def read_nifti_file(filepath):
        """Read and load volume"""
        # Read file
        scan = nib.load(filepath)
        # Get raw data
        scan = scan.get_fdata()
        return scan

    @staticmethod
    def resize_mask_volume(img, desired_width=128, desired_height=128, desired_depth=64):
        """Resize the volume"""
        # Compute zoom factors
        width_factor = desired_width / img.shape[0]
        height_factor = desired_height / img.shape[1]
        depth_factor = desired_depth / img.shape[-1]

        # Rotate volume by 90 degrees
        img = ndimage.rotate(img, 90,  axes=(0, 1), reshape=False) # TODO why ??

        # Resize the volume using spline interpolated zoom (SIZ)
        img = zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img

    @staticmethod
    def process_mask(path):
        """Read and resize volume"""
        # Read mask
        volume = MaskDataset.read_nifti_file(path)
        # Resize width, height and depth
        volume = MaskDataset.resize_mask_volume(
            volume, config['img_size'], config['img_size'], config['depth']
        )
        return volume
