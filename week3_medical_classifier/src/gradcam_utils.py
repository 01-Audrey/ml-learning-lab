"""
Grad-CAM Utilities for Medical X-Ray Classification
Author: ML Learning Journey - Week 3
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam, target_class, output

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Apply heatmap overlay on original image"""
    heatmap = np.uint8(255 * activation_map)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if org_img.shape[:2] != heatmap.shape[:2]:
        org_img = cv2.resize(org_img, (heatmap.shape[1], heatmap.shape[0]))

    if org_img.dtype != np.uint8:
        org_img = np.uint8(255 * org_img)

    superimposed_img = cv2.addWeighted(org_img, 0.7, heatmap, 0.3, 0)

    return heatmap, superimposed_img

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    img = tensor.clone()

    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.uint8(255 * img)

    return img
