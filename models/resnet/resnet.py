
import torch
import torch.nn as nn
from utils import Device

from .model.renset6 import ResNet as ResNet6
from .model.renset8 import ResNet as ResNet8
from .model.renset_all import ResNet as ResNet
from .model.common import BasicBlock, Bottleneck

__all__ = [
    'ResNet', 'resnet6', 'resnet8', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


class resnet:
    def resnet6(**kwargs):
        """Constructs a ResNet-6 model.
        """
        model = ResNet6(BasicBlock, [1, 1], **kwargs)
        return model
    
    def resnet8(**kwargs):
        """Constructs a ResNet-8 model.
        """
        model = ResNet8(BasicBlock, [1, 1, 1], **kwargs)
        return model
    
    def resnet10(**kwargs):
        """Constructs a ResNet-10 model.
        """
        model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
        return model


    def resnet18(**kwargs):
        """Constructs a ResNet-18 model.
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        return model


    def resnet34(**kwargs):
        """Constructs a ResNet-34 model.
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        return model


    def resnet50(**kwargs):
        """Constructs a ResNet-50 model.
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        return model


    def resnet101(**kwargs):
        """Constructs a ResNet-101 model.
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
        return model


    def resnet152(**kwargs):
        """Constructs a ResNet-101 model.
        """
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        return model


    def resnet200(**kwargs):
        """Constructs a ResNet-101 model.
        """
        model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
        return model


class Resnet3D():
    @staticmethod
    def generate_model(config):
        assert config.model in [
            'resnet'
        ]
        device_obj = Device()
        device = device_obj.get_device()
        if config.model == 'resnet':
            assert config.model_depth in [6, 8, 10, 18, 34, 50, 101, 152, 200]
            
            if config.model_depth == 6:
                model = resnet.resnet6(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            if config.model_depth == 8:
                model = resnet.resnet8(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 10:
                model = resnet.resnet10(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 18:
                model = resnet.resnet18(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 34:
                model = resnet.resnet34(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 50:
                model = resnet.resnet50(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 101:
                model = resnet.resnet101(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 152:
                model = resnet.resnet152(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
            elif config.model_depth == 200:
                model = resnet.resnet200(
                    sample_input_W=config.input_W,
                    sample_input_H=config.input_H,
                    sample_input_D=config.input_D,
                    shortcut_type=config.resnet_shortcut,
                    no_cuda=config.no_cuda,
                    num_classes=config.n_classes)
        
        if not config.no_cuda:
            if len(config.gpu_id) > 1:
                model = model.to(device) 
                model = nn.DataParallel(model, device_ids=config.gpu_id)
                net_dict = model.state_dict() 
            else:
                import os
                os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_id[0])
                model = model.to(device) 
                ## set device_ids to config.gpu_id when running 3d_resnet_grad_cam, like below
                ## model = nn.DataParallel(model, device_ids=config.gpu_id)
                model = nn.DataParallel(model, device_ids=None)
                net_dict = model.state_dict()
        else:
            net_dict = model.state_dict()
        
        # load pretrain
        if config.phase != 'test' and config.pretrain_path:
            print ('loading pretrained model {}'.format(config.pretrain_path))
            pretrain = torch.load(config.pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)

            new_parameters = [] 
            for pname, p in model.named_parameters():
                for layer_name in config.new_layer_names:
                    if pname.find(layer_name) >= 0:
                        new_parameters.append(p)
                        break

            new_parameters_id = list(map(id, new_parameters))
            base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
            parameters = {'base_parameters': base_parameters, 
                        'new_parameters': new_parameters}

            return model, parameters

        return model, model.parameters()

