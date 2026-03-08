import torch
import cv2
import numpy as np

def generate_gradcam(model, image_tensor, original_image):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    handle_forward.remove()
    handle_backward.remove()

    return overlay