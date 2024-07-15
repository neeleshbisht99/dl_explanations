import nibabel as nib
from scipy.ndimage import zoom
from scipy import ndimage

from config import CnnConfig

class ImageDataset:
    @staticmethod
    def read_nifti_file(filepath):
        """Read and load volume"""
        # Read file
        scan = nib.load(filepath)
        # Get raw data
        scan = scan.get_fdata()
        return scan

    @staticmethod
    def normalize(volume):
        """Normalize the volume"""
        min_hu = -1000
        max_hu = 400
        volume[volume < min_hu] = min_hu
        volume[volume > max_hu] = max_hu
        volume = (volume - min_hu) / (max_hu - min_hu)
        return volume.astype('float32')

    @staticmethod
    def resize_volume(img, desired_width=128, desired_height=128, desired_depth=64):
        """Resize the volume"""
        # Compute zoom factors
        width_factor = desired_width / img.shape[0]
        height_factor = desired_height / img.shape[1]
        depth_factor = desired_depth / img.shape[-1]
        # Rotate volume by 90 degrees
        img = ndimage.rotate(img, 90, reshape=False)
        # Resize the volume using spline interpolated zoom (SIZ)
        img = zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img

    @staticmethod
    def process_image(path, is_mask_file = False):
        """Read and resize volume"""
        config = CnnConfig()
        # Read scan
        volume = ImageDataset.read_nifti_file(path)
        # Normalize
        volume = ImageDataset.normalize(volume)
        # Resize width, height and depth
        volume = ImageDataset.resize_volume(
            volume, config.img_size, config.img_size, config.depth
        )
        return volume
