# utils.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from torch.nn import functional as F
import torch.nn as nn

# ============================================
# MODELO ARQUITECTURA - IDÉNTICO AL ENTRENADO
# ============================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class CNN_DesdeCero(nn.Module):
    def __init__(self, num_classes, img_size=128, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================
# PREPROCESAMIENTO
# ============================================
def preprocess_image(image: Image.Image, size=128):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


# ============================================
# CARGA DE MODELO
# ============================================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Obtener parámetros del checkpoint
    num_classes = checkpoint.get("num_classes", 38)
    img_size = checkpoint.get("img_size", 128)
    
    # Crear modelo con la arquitectura original
    model = CNN_DesdeCero(num_classes=num_classes, img_size=img_size)
    
    # Cargar los pesos
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, img_size


# ============================================
# NOMBRES DE CLASES
# ============================================
def load_class_names(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except:
        # Si no existe, crear nombres genéricos
        return [f"Clase_{i}" for i in range(38)]


# ============================================
# GRAD-CAM
# ============================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_gradcam(model, input_tensor, target_layer, target_class, original_image, alpha=0.5):
    model.eval()
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor, target_class)
    
    orig_w, orig_h = original_image.size
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    
    img_np = np.array(original_image).astype(np.float32) / 255.0
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    overlayed = alpha * heatmap + (1 - alpha) * img_np
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed


# ============================================
# UTILIDADES
# ============================================
def get_top_predictions(probabilities, class_names, top_k=5):
    probs_np = probabilities.cpu().numpy()
    top_indices = np.argsort(probs_np)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        class_name = class_names[idx] if idx < len(class_names) else f"Clase_{idx}"
        prob = probs_np[idx]
        predictions.append((class_name, prob))
    
    return predictions

def format_confidence(confidence):
    return f"{confidence*100:.2f}%"