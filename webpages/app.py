from flask import Flask, request, jsonify
import torch
import timm
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load models
cat_model = timm.create_model('inception_v4', pretrained=False, num_classes=3)
cat_model.load_state_dict(torch.load('model/sample_cat_inception_v4_model.pth', map_location=torch.device('cpu')))
cat_model.eval()

dog_model = timm.create_model('inception_v4', pretrained=False, num_classes=5)
dog_model.load_state_dict(torch.load('model/sample_dog_inception_v4_model.pth', map_location=torch.device('cpu')))
dog_model.eval()

# Define image preprocessing for the model
preprocess = transforms.Compose([
    transforms.Resize(299),  # Inception_v4 expects 299x299 images
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_image(image_url, selected_animal):
    # Fetch the image from the provided URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    # Preprocess the image for the model
    img = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Choose the correct model based on the selected animal
    if selected_animal == '고양이':
        model = cat_model
    elif selected_animal == '강아지':
        model = dog_model
    else:
        return "Unknown animal"

    # Run the image through the model and get the output
    with torch.no_grad():
        output = model(img)
    
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    return int(predicted_class.item())  # Return class index

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_url = data.get('imageURL')
    selected_animal = data.get('animal')
    
    result = analyze_image(image_url, selected_animal)
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
