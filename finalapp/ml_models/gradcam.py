import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.feature = None

        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature = output

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.handlers.append(module.register_forward_hook(forward_hook))
                self.handlers.append(module.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, index=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        
        target = output[0][index]
        target.backward()

        gradient = self.gradient.cpu().data.numpy()[0]
        feature = self.feature.cpu().data.numpy()[0]

        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(feature.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * feature[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        
        return cam

def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor
 