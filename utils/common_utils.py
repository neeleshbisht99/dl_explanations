import torch
from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt

class CommonUtils:
    @staticmethod
    def get_device(device_id = 0):
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def get_resized_heatmap(heatmap, shape):
        width = shape[0]
        height = shape[1]
        depth = shape[2]
        upscaled_heatmap = zoom(heatmap, (width / heatmap.shape[0], height / heatmap.shape[1], depth / heatmap.shape[2]), order=1)
        upscaled_heatmap = np.uint8(255 * upscaled_heatmap)
        return upscaled_heatmap

    @staticmethod
    def plot_slices(num_rows, num_columns, width, height, data):
        """Plot a montage of CT slices"""
        # rotate the volume at 90 deg to set the alignment
        data = np.rot90(np.array(data)) # (128, 128, 40) (width, height, depth)
        data = np.transpose(data) # (40, 128, 128) (depth, height, width )
        data = np.reshape(data, (num_rows, num_columns, width, height))  # (4, 10, 128, 128) (row, column, height, width)
        rows_data, columns_data = data.shape[0], data.shape[1]
        heights = [slc[0].shape[0] for slc in data]
        widths = [slc.shape[1] for slc in data[0]]
        fig_width = 12.0
        fig_height = fig_width * sum(heights) / sum(widths)
        _, axarr = plt.subplots(
            rows_data,
            columns_data,
            figsize=(fig_width, fig_height),
            gridspec_kw={'height_ratios': heights},
        )
        for i in range(rows_data):
            for j in range(columns_data):
                axarr[i, j].imshow(data[i][j], cmap='gray')
                axarr[i, j].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.show()

