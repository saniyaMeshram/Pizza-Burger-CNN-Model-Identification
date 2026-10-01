import streamlit as st
import torch
import torch.nn as nn
import torch.quantization # Added for quantization
from torchvision import transforms
from PIL import Image
import os

# Define the CNN model architecture (must be the same as trained)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128), # Adjust based on your image size after pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Set device to CPU for deployment (Streamlit typically runs on CPU)
device = torch.device("cpu")

# Load the trained model
@st.cache_resource
def load_model(model_path='model_quantized.pt', num_classes=2): # Changed to model_quantized.pt
    model = SimpleCNN(num_classes=num_classes).to(device)
    # Apply dynamic quantization before loading state_dict
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
    quantized_model.load_state_dict(torch.load(model_path, map_location=device))
    quantized_model.eval()
    return quantized_model # Return the quantized model

# Image transformations (must be the same as training/testing)
preprocess = transforms.Compose([
    transforms.Resize((250, 250)), # Ensure image is 250x250, though already resized
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model and class names
model = load_model()
# Ensure class_names matches your training data order
class_names = ['keyboard', 'mouse'] # Assuming this order from full_dataset.classes

st.title("Image Classifier: Mouse vs. Keyboard")
st.write("Upload an image to classify it as a mouse or a keyboard.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}")
