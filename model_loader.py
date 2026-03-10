import torch
import torch.nn as nn
from torchvision import models

def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("model_weight.pth", map_location="cpu"))
    model.eval()

    return model