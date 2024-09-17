import torch.nn as nn
import torchvision

class Resnet(nn.Module):
    def __init__(self, device = 0):
        super(Resnet, self).__init__()
        model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, 2)
        model.to(device)
        self.model = model
        self.device = device
    
    def __call__(self, x):
        x = x.to(self.device)
        return self.model(x)