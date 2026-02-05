import os
import streamlit as st
import torch
import numpy as np
import cv2

from src.models.cnn_baseline import CNNBaseline
from src.segmentation.unet import UNet

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@st.cache_resource
def load_models():
    cnn = CNNBaseline(num_classes=4)
    cnn.load_state_dict(torch.load("saved_models/cnn_gan_augmented.pth", map_location="cpu"))
    cnn.eval()

    unet = UNet()
    unet.load_state_dict(torch.load("saved_models/unet_segmentation.pth", map_location="cpu"))
    unet.eval()

    return cnn, unet

cnn_model, unet_model = load_models()

st.title("🧬 MRI Diagnosis")

if "patient" not in st.session_state:
    st.warning("⚠️ Please register patient first.")
    st.stop()

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)

    st.image(img, caption="Uploaded MRI", width=300)

    # CNN prediction
    img_r = cv2.resize(img, (224, 224)) / 255.0
    tensor = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        out = cnn_model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    tumor = CLASS_NAMES[pred.item()]
    confidence = conf.item() * 100

    st.success(f"Tumor Type: **{tumor}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    # U-Net mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    gray_tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        mask = unet_model(gray_tensor).squeeze().numpy()

    overlay = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    combined = cv2.addWeighted(
    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0
)

    st.image(combined, caption="Tumor Mask Overlay")

# 🔽 SAVE IMAGE FOR REPORT
    os.makedirs("reports/figures", exist_ok=True)
    seg_path = "reports/figures/segmentation_overlay.png"
    cv2.imwrite(seg_path, combined)

    st.session_state["segmentation_image"] = seg_path


    st.session_state["diagnosis"] = {
        "tumor": tumor,
        "confidence": confidence
    }
