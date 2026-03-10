import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, tensor, original_img):
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    original_img = cv2.resize(original_img, (224, 224))
    original_img = original_img.astype(np.float32) / 255

    cam_image = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)

    return cam_image