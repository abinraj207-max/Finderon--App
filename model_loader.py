import torch
import torchvision.models as models
import torch.nn as nn

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("model_weight.pth", map_location="cpu"))
    model.eval()
    return model