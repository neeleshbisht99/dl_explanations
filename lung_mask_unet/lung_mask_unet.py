## run as script
import os # For System operations
import sys
sys.path.append('/shared/home/v_neelesh_bisht/local_scratch/3d-cnn')

import nibabel as nib # To load .nii.gz files
import cv2 # for image operations
import matplotlib.pyplot as plt # To plot the images
import numpy as np # Numpy operations\
from tensorflow.keras.models import load_model # To load the model
import numpy as np # Numpy operations
from utils import Device



if __name__ == "__main__":

    device_obj = Device()
    device_obj.set_device(6)
    device = device_obj.get_device()
    print(f"Using device: {device}")


    HOUNSFIELD_MIN = -3500 # assigned minimum HOUNSFIELD value
    HOUNSFIELD_MAX = 3500 # assigned maximum HOUNSFIELD value
    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN # difference of HOUNSFIELD max and min values

    # these variables define from what side/plane should be sliced
    SLICE_X = True # represents the sagittal slice
    SLICE_Y = True # represents the coronal slice
    SLICE_Z = True # represents the axial slice

    SLICE_DECIMATE_IDENTIFIER = 3

    # Function which normalises the values
    def normalizeImageIntensityRange(img):
        """
        This function normalizes the image intensity range from -3500 to 3500 to 0 to 1
        """
        img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
        img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
        return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


    def readImageVolume(imgPath, normalize=False):
        """
        This function function load the image and then if necessary normalise the image
        """
        img = nib.load(imgPath).get_fdata()
        if normalize:
            return normalizeImageIntensityRange(img)
        else:
            return img


    # Defining Constants for training

    SEED = 42 # Seed for reproducibility
    IMAGE_HEIGHT = 80 # Set the image height for training
    IMAGE_WIDTH = 40 # Set the image width for training
    IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


    model_path = os.path.join('/shared/home/v_neelesh_bisht/local_scratch/3d-cnn', "models/archive/lung_seg_unet/UNET_LungSegmentation_3D_10epochs_GPU_ALL.h5")
    model = load_model(model_path)


    # Scale the image accordingly
    def scaleImg(img, height, width):
        return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)


    def predictVolume(inImg, toBin=True):
        (xMax, yMax, zMax) = inImg.shape

        outImgX = np.zeros((xMax, yMax, zMax)) # Create imgx in the shape of inImgx
        outImgY = np.zeros((xMax, yMax, zMax)) # Create imgy in the shape of inImgy
        outImgZ = np.zeros((xMax, yMax, zMax)) # Create imgz in the shape of inImgz

        cnt = 0.0
        if SLICE_X:
            cnt += 1.0
            for i in range(xMax):
                img = scaleImg(inImg[i,:,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
                tmp = model.predict(img)[0,:,:,0]
                outImgX[i,:,:] = scaleImg(tmp, yMax, zMax)
        if SLICE_Y:
            cnt += 1.0
            for i in range(yMax):
                img = scaleImg(inImg[:,i,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
                tmp = model.predict(img)[0,:,:,0]
                outImgY[:,i,:] = scaleImg(tmp, xMax, zMax)
        if SLICE_Z:
            cnt += 1.0
            for i in range(zMax):
                img = scaleImg(inImg[:,:,i], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
                tmp = model.predict(img)[0,:,:,0]
                outImgZ[:,:,i] = scaleImg(tmp, xMax, yMax)

        outImg = (outImgX + outImgY + outImgZ)/cnt # Concatenate all the sides to form into one
        if(toBin):
            outImg[outImg>0.5] = 1.0 # Appying thresholding if > 0.5 then assign 1
            outImg[outImg<=0.5] = 0.0 # Appying thresholding if <= 0.5 then assign 0
        return outImg


    output_dir = os.path.join('/shared/home/v_neelesh_bisht/local_scratch/3d-cnn', "MosMedData")
    paths = [os.path.join(output_dir, "CT-0", x) for x in sorted(os.listdir(os.path.join(output_dir, "CT-0")))]
    save_path = os.path.join(output_dir, "CT-0-mask")


    # Load a sample data
    for path in paths[252:]:
        imgTargetNii = nib.load(path).get_fdata() # load the random 3d image
        imgTarget = normalizeImageIntensityRange(imgTargetNii) # Normalize the 3d ct scan image
        predImg = predictVolume(imgTarget) # get the volume prediction
        # print("imgTarget.shape: ",imgTarget.shape, ", predImg.shape: ", predImg.shape)
        a = nib.Nifti1Image(predImg, affine=np.eye(4))
        last_name = os.path.basename(path)
        nib.save(a, f"{save_path}/{last_name}")

    print("all done")