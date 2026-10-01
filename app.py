import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path


# =========================================================
# 1. CNN MODEL ARCHITECTURE
# =========================================================

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
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 2. DEVICE
# =========================================================

device = torch.device("cpu")


# =========================================================
# 3. MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_quantized.pth"


# =========================================================
# 4. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    model = SimpleCNN(num_classes=2)

    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Load trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    quantized_model.load_state_dict(state_dict)

    quantized_model.eval()

    return quantized_model


# =========================================================
# 5. IMAGE PREPROCESSING
# =========================================================

preprocess = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 6. LOAD MODEL
# =========================================================

model = load_model()


# =========================================================
# 7. CLASS NAMES
# =========================================================

class_names = [
    "Burger",
    "Pizza"
]


# =========================================================
# 8. STREAMLIT UI
# =========================================================

st.title("⌨️🖱️ Image Classifier: Burger vs Pizza")

st.write(
    "Upload an image and the CNN model will classify it "
    "as either a Burger or a Pizza."
)


uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# 9. PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("### Classifying...")

    # Preprocess
    input_tensor = preprocess(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        output = model(input_batch)

        probabilities = torch.nn.functional.softmax(
            output[0],
            dim=0
        )

        predicted_idx = torch.argmax(
            probabilities
        ).item()

        predicted_class = class_names[predicted_idx]

        confidence = probabilities[
            predicted_idx
        ].item()

    # Result
    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )
