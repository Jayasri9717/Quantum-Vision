import torch
import numpy as np
import cv2
from PIL import Image

class CorrectGradCAM:
    """
    Fixed Grad-CAM for Hybrid Quantum Models using ShuffleNetV2.
    Hooks into 'conv5' to bypass broken quantum gradients.
    """
    def __init__(self, shufflenet_model):
        self.model = shufflenet_model
        self.gradients = None
        self.activations = None
        
        # TARGET: ShuffleNetV2's last conv layer is explicitly named 'conv5'
        self.target_layer = self.model.conv5
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        input_tensor.requires_grad = True
        
        # MANUAL FORWARD PASS: Stop exactly at conv5
        x = self.model.conv1(input_tensor)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        features = self.model.conv5(x) 
        
        # Global Average Pooling to get class scores proxy
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        pooled = pooled.view(pooled.size(0), -1)
        
        if target_class is None:
            target_class = torch.argmax(pooled, dim=1).item()
            
        # Backward pass from this point
        one_hot = torch.zeros_like(pooled)
        one_hot[0, target_class % pooled.shape[1]] = 1.0
        pooled.backward(gradient=one_hot, retain_graph=True)
        
        # Generate Heatmap
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap) 
        
        # Normalize
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap.squeeze().cpu().numpy()


def simple_activation_heatmap(shufflenet_model, input_tensor):
    """
    BULLETPROOF FALLBACK: Uses raw activation magnitude. 
    No gradients needed. Works 100% of the time if Grad-CAM fails.
    """
    shufflenet_model.eval()
    activations = {}
    
    def hook_fn(module, input, output):
        activations['features'] = output.detach()
        
    handle = shufflenet_model.conv5.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        # Manual forward pass matching CorrectGradCAM
        x = shufflenet_model.conv1(input_tensor)
        x = shufflenet_model.maxpool(x)
        x = shufflenet_model.stage2(x)
        x = shufflenet_model.stage3(x)
        x = shufflenet_model.stage4(x)
        _ = shufflenet_model.conv5(x)
        
    handle.remove()
    
    # Average across all channels
    heatmap = activations['features'].mean(dim=1, keepdim=True)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap.squeeze().cpu().numpy()


def create_overlay_from_pil(pil_image, heatmap_array, target_size=(224, 224)):
    """
    Creates the 3-panel visualization using PIL (Streamlit friendly).
    """
    original = pil_image.convert('RGB').resize(target_size)
    original_np = np.array(original)
    
    # Ensure heatmap is 2D
    if len(heatmap_array.shape) > 2:
        heatmap_array = heatmap_array.squeeze()
        
    heatmap_resized = cv2.resize(heatmap_array, target_size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend: 60% Original, 40% Heatmap
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_colored_rgb, 0.4, 0)
    
    return original_np, heatmap_colored_rgb, overlay