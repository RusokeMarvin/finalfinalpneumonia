from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ml_models.gradcam import GradCAM, preprocess_image
import torch
from torchvision import transforms
import numpy as np
import cv2
from django.http import JsonResponse
import torch.nn as nn
import os
import firebase_admin
from firebase_admin import credentials, storage

# Path to your service account key file
SERVICE_ACCOUNT_KEY_PATH = 'CODE/serviceAccountKey.json'
LOCAL_MODEL_PATH = 'finalapp/ml_models/oursecondmodel.pth'
REMOTE_MODEL_PATH = 'oursecondmodel.pth'

# Initialize Firebase Admin SDK with the storage bucket name
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'djangopneumonia-b28e1.appspot.com'
})

def download_model():
    """Download the model from Firebase Storage if not present locally."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        bucket = storage.bucket()
        blob = bucket.blob(REMOTE_MODEL_PATH)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print(f"Model downloaded to {LOCAL_MODEL_PATH}.")
    else:
        print(f"Model already present at {LOCAL_MODEL_PATH}.")

# Download the model if not already present
download_model()

# Define your custom VGG19 model class
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the model state dictionary
model = VGG19()
state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        # Load and preprocess the image
        img_tensor = preprocess_image(fs.path(filename))

        # Instantiate GradCAM with your custom model and specify target layer
        grad_cam = GradCAM(model, 'features.16')

        # Generate Grad-CAM
        cam = grad_cam(img_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = cv2.imread(fs.path(filename))
        img = cv2.resize(img, (224, 224))
        cam_img = heatmap + np.float32(img) / 255
        cam_img = cam_img / np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)

        cam_filename = 'cam_' + filename
        cv2.imwrite(fs.path(cam_filename), cam_img)
        cam_file_url = fs.url(cam_filename)

        response_data = {
            'uploaded_file_url': uploaded_file_url,
            'cam_file_url': cam_file_url
        }

        return JsonResponse(response_data)

    return render(request, 'index.html')
