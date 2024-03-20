from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
from PIL import Image
import torch
from torchvision import models
from io import BytesIO
import os
import cv2
import numpy as np

from django.conf import settings
import glob
from gtts import gTTS
from playsound import playsound



def index(request):
    return render(request, 'index.html')

prediction = ""
image_path = ""

def analyze(request):
    global prediction
    global image_path
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
            
            prediction = predicted_label
            print(predicted_label)

            # After processing, get the most recent file in the static folder
            static_folder_path = os.path.join(settings.BASE_DIR, 'static')
            most_recent_file = max(glob.glob(os.path.join(static_folder_path, '*')), key=os.path.getctime)
            image_path = most_recent_file

            params = {'label': predicted_label, 'img_pth': static_image_path}
            os.remove(os.path.join(settings.MEDIA_ROOT, filename))
            return render(request, 'index.html', params)
        
        
        
        
        
        
        
        
        
def speak(quote):
        # Create a gTTS object
        tts = gTTS(text=quote, lang='en')

        # Save the speech to a temporary file
        tts.save("speech.mp3")

        # Play the speech directly
        playsound("speech.mp3")
        


    
def delete(request):
    global prediction
    global image_path
    
    quote =""
    if(prediction=='SQUARE'):
        quote='This is a square, four sides like a frame'
    if(prediction=='CIRCLE'):
        quote='This is a circle, round like a ball'
    if(prediction=='TRIANGLE'):
        quote='This is a triangle, three sides in all'
    if(prediction=='HEART'):
        quote='This is a heart, made with love'
    if(prediction=='STAR'):
        quote='This is a star, from up above'
        
        
    

    # Function to convert text to speech and play it
    

    # Test the speak function
    speak(quote)
    # speak(caption)
        

    params = {'label': quote, 'img_pth': image_path}
    return render(request, 'result.html', params)
