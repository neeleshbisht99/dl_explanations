import matplotlib.pyplot as plt

from .grad_cam import GradCam
from .diff_cam import DiffCAM
from .counter_factual import CounterFactual
from .torch_cam import TorchCAM

from utils import CommonUtils

#TODO: move to config
shape = [224,224]

class ResnetCAMS:


    @staticmethod
    def get_cams(image, resnet_model, last_conv_layer_name, target_class_idx, ref_class_idx, show_cams=False):
        # Generate torch-cam score-cam
        # torch_cam_score_cam_heatmap = TorchCAM.compute_score_cam(image, resnet_model, last_conv_layer_name, target_class_idx)
        # torch_cam_score_cam_heatmap = torch_cam_score_cam_heatmap[0]
        # if show_cams:
        #     plt.matshow(torch_cam_score_cam_heatmap)
        #     plt.show()
        
        # Generate torch-cam grad-cam
        torch_cam_grad_cam_heatmap = TorchCAM.compute_grad_cam(image, resnet_model, last_conv_layer_name, target_class_idx)
        torch_cam_grad_cam_heatmap = torch_cam_grad_cam_heatmap[0]
        if show_cams:
            plt.matshow(torch_cam_grad_cam_heatmap)
            plt.show()

        # Generate torch-cam grad-cam plus
        torch_cam_grad_campp_heatmap = TorchCAM.compute_grad_campp(image, resnet_model, last_conv_layer_name, target_class_idx)
        torch_cam_grad_campp_heatmap = torch_cam_grad_campp_heatmap[0]
        if show_cams:
            plt.matshow(torch_cam_grad_campp_heatmap)
            plt.show()

        # Generate grad and diff-grad class activation heatmap
        grad_cam_heatmap, _,  diff_grad_cam_heatmap= GradCam.compute(image, resnet_model, last_conv_layer_name, target_class_idx, ref_class_idx)
        if show_cams:
            plt.matshow(grad_cam_heatmap)
            plt.show()

        # Generate diff class activation heatmap
        diff_cam_heatmap = DiffCAM.make_diffcam_heatmap(image, resnet_model, last_conv_layer_name, target_class_idx, ref_class_idx)
        if show_cams:
            plt.matshow(diff_cam_heatmap)
            plt.show()


        #Generate counter factual heatmap
        counter_factual_heatmap = CounterFactual.make_counter_factual_heatmap(image, resnet_model, last_conv_layer_name, target_class_idx, ref_class_idx)
        if show_cams:
            plt.matshow(counter_factual_heatmap)
            plt.show()

        ## TODO REMOVE HACK # score cam is throwing error and show score cam in the all cams heatmap  as well
        torch_cam_score_cam_heatmap = grad_cam_heatmap

        # Resize heatmap
        grad_cam_heatmap = CommonUtils.get_resized_heatmap(grad_cam_heatmap, shape)
        diff_grad_cam_heatmap = CommonUtils.get_resized_heatmap(diff_grad_cam_heatmap, shape)
        diff_cam_heatmap = CommonUtils.get_resized_heatmap(diff_cam_heatmap, shape)
        counter_factual_heatmap = CommonUtils.get_resized_heatmap(counter_factual_heatmap, shape)
        torch_cam_grad_cam_heatmap = CommonUtils.get_resized_heatmap(torch_cam_grad_cam_heatmap, shape)
        torch_cam_grad_campp_heatmap = CommonUtils.get_resized_heatmap(torch_cam_grad_campp_heatmap, shape)
        torch_cam_score_cam_heatmap = CommonUtils.get_resized_heatmap(torch_cam_score_cam_heatmap, shape)

        return {
            "grad_cam_heatmap": grad_cam_heatmap,
            "diff_grad_cam_heatmap": diff_grad_cam_heatmap,
            "diff_cam_heatmap": diff_cam_heatmap,
            "counter_factual_heatmap": counter_factual_heatmap,
            "torch_cam_grad_cam_heatmap": torch_cam_grad_cam_heatmap,
            "torch_cam_grad_campp_heatmap": torch_cam_grad_campp_heatmap,
            "torch_cam_score_cam_heatmap": torch_cam_score_cam_heatmap
        }