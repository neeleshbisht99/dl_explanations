import torch.nn as nn
import torchvision
import torch

class Inceptionv3(nn.Module):
    def __init__(self, device = 0, path=None):
        super(Inceptionv3, self).__init__()
        model = torchvision.models.inception_v3(weights='IMAGENET1K_V1')
        model.aux_logits = False 
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, 2)
        if path:
            model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        self.model = model
        self.device = device
    
    def __call__(self, x):
        x = x.to(self.device)
        return self.model(x)
