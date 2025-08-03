import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Definition ---
class SpectraXPlus(nn.Module):
    def __init__(self, num_classes=16, input_bands=194):
        super(SpectraXPlus, self).__init__()
        self.input_bands = input_bands
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(16)

        self.conv2d_1 = nn.Conv2d(16 * (input_bands - 6 - 4), 128, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)
        x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# --- Load Model Function ---
@st.cache_resource
def load_model(path, input_bands):
    model = SpectraXPlus(num_classes=16, input_bands=input_bands).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# --- Patch Extraction Function ---
def extract_patch(cube, center_x, center_y, patch_size=25, bands=194):
    half = patch_size // 2
    cube = cube[:bands, :, :]  # truncate spectral bands
    patch = cube[:, center_x - half:center_x + half + 1, center_y - half:center_y + half + 1]
    return patch

# --- Inference Function ---
def predict(model, patch_np):
    patch_tensor = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(patch_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# --- Streamlit UI ---
st.title("SpectraXPlus: HSI Classification Viewer")

uploaded_file = st.file_uploader("Upload HSI .npy cube file", type=["PNG", "npy"])

if uploaded_file:
    cube = np.load(uploaded_file)
    st.write(f"Loaded HSI array with shape: {cube.shape}")  # e.g., (145, 145, 200)

    if cube.shape[2] < 194:
        st.error("Need at least 194 spectral bands")
    else:
        # Transpose to (Bands, H, W)
        cube = cube.transpose(2, 0, 1)

        input_bands = 194
        model = load_model("SpectraXPlus_best.pth", input_bands)

        st.write("Select a location to classify")
        x = st.slider("X (row)", 12, cube.shape[1] - 13, 72)
        y = st.slider("Y (col)", 12, cube.shape[2] - 13, 72)

        patch = extract_patch(cube, x, y)

        if patch.shape == (194, 25, 25):
            pred_class = predict(model, patch)
            st.success(f"Predicted Class: {pred_class}")

            # Optional: show the RGB-like view
            rgb_img = cube[[30, 20, 10], :, :]  # False color
            rgb_img = np.stack([rgb_img[i] / np.max(rgb_img[i]) for i in range(3)], axis=-1)
            fig, ax = plt.subplots()
            ax.imshow(rgb_img)
            ax.scatter(y, x, c='red', marker='x')
            ax.set_title("False Color Image with Selected Point")
            st.pyplot(fig)
        else:
            st.error(f"Invalid patch shape: {patch.shape}")
            st.write("Note: The model expects patches of shape (194, 25, 25).")