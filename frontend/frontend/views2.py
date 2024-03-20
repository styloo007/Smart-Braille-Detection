from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
from PIL import Image
import torch
from torchvision import models
import cv2
import numpy as np
import os

from django.conf import settings


def index(request):
    return render(request, 'index.html')



def analyze(request):
    if request.method == 'POST':
        # Assuming the image is sent as 'image' in the request
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        static_image_path = 'static/' + uploaded_image.name
        fs.save(static_image_path, uploaded_image)

        model = models.resnet152(pretrained=False, num_classes=5)
        model_path = 'E:/Haegl/Haegl ML Projects/Braille/best.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Define transformations for the image
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define class mapping
        class_mapping = {
            0: 'CIRCLE',
            1: 'HEART',
            2: 'SQUARE',
            3: 'STAR',
            4: 'TRIANGLE'
        }
        
        with torch.no_grad():
            # Open the image with PIL and convert it to RGB
            pil_image = Image.open(uploaded_image).convert('RGB')
            # Convert the PIL Image to a NumPy array
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (224, 224))
            # Convert the NumPy array back to a PIL Image for transformations
            image = Image.fromarray(image)
            input_tensor = test_transform(image).unsqueeze(0)

            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = class_mapping[predicted.item()]
            
            print(predicted_label)
            
            params = {'label': predicted_label, 'img_pth': static_image_path}
            os.remove(os.path.join(settings.MEDIA_ROOT, filename))
            return render(request, 'result.html', params)
    else:
        return redirect('index')
