import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from PIL import Image

import numpy as np
import pandas as pd
import cv2
import joblib

import plotly.express as px

from scipy.stats import skew

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

from skimage.measure import shannon_entropy


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DermaScan",
    layout="wide"
)

# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "VASC"
]

# ============================================================
# CNN TRANSFORM
# ============================================================

cnn_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])

# ============================================================
# RESNET TRANSFORM
# ============================================================

resnet_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ============================================================
# CUSTOM CNN
# ============================================================

class CustomCNN(nn.Module):

    def __init__(self, num_classes):

        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256,512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512,num_classes)
        )

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD CNN
# ============================================================

@st.cache_resource
def load_cnn():

    model = CustomCNN(7)

    model.load_state_dict(
        torch.load(
            "cnn_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


# ============================================================
# LOAD RESNET50
# ============================================================

@st.cache_resource
def load_resnet():

    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Linear(in_features,512),

        nn.ReLU(),

        nn.Dropout(0.48),

        nn.Linear(512,7)
    )

    model.load_state_dict(
        torch.load(
            "resnet50_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    svm_model = joblib.load(
        "svm_baseline.pkl"
    )

    label_encoder = joblib.load(
        "label_encoder_svm.pkl"
    )

    return svm_model, label_encoder



# ============================================================
# EDGE FEATURES
# ============================================================

def extract_edge_features(image):

    gray = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2GRAY
    )

    sobelx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobely = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobelx**2 + sobely**2
    )

    edge_mean = np.mean(magnitude)

    edge_std = np.std(magnitude)

    edge_energy = (
        np.sum(magnitude**2)
        /
        (gray.shape[0] * gray.shape[1])
    )

    edge_skew = skew(
        magnitude.flatten()
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    lap_var = np.var(lap)

    return [
        edge_mean,
        edge_std,
        edge_energy,
        edge_skew,
        lap_var
    ]


# ============================================================
# HSV FEATURES
# ============================================================

def extract_color_features_hsv(image):

    image_np = np.array(image)

    hsv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2HSV
    )

    features = []

    channel_ranges = [
        (0,180),
        (0,256),
        (0,256)
    ]

    for i, channel_range in enumerate(channel_ranges):

        channel = hsv[:,:,i]

        hist = np.histogram(
            channel,
            bins=16,
            range=channel_range
        )[0].astype(np.float32)

        hist = hist / (
            np.sum(hist) + 1e-7
        )

        features.extend(
            hist.tolist()
        )

        features.append(
            np.mean(channel)
        )

        features.append(
            np.std(channel)
        )

        features.append(
            skew(channel.flatten())
        )

    return features


# ============================================================
# LAB FEATURES
# ============================================================

def extract_color_features_lab(image):

    image_np = np.array(image)

    lab = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2LAB
    )

    features = []

    for i in range(3):

        channel = lab[:,:,i]

        features.append(
            np.mean(channel)
        )

        features.append(
            np.std(channel)
        )

        features.append(
            skew(channel.flatten())
        )

    return features


# ============================================================
# GLCM FEATURES
# ============================================================

def extract_texture_features(image):

    gray = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2GRAY
    )

    gray_reduced = gray // 8

    glcm = graycomatrix(
        gray_reduced,
        distances=[1],
        angles=[
            0,
            np.pi/4,
            np.pi/2,
            3*np.pi/4
        ],
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    ).mean()

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    ).mean()

    correlation = graycoprops(
        glcm,
        "correlation"
    ).mean()

    energy = graycoprops(
        glcm,
        "energy"
    ).mean()

    dissimilarity = graycoprops(
        glcm,
        "dissimilarity"
    ).mean()

    entropy = shannon_entropy(
        gray
    )

    return [
        contrast,
        homogeneity,
        correlation,
        energy,
        dissimilarity,
        entropy
    ]


# ============================================================
# LBP FEATURES
# ============================================================

def extract_lbp_features(image):

    gray = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2GRAY
    )

    radius = 2

    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            n_points + 3
        ),
        range=(
            0,
            n_points + 2
        )
    )

    hist = hist.astype(
        "float"
    )

    hist /= (
        hist.sum() + 1e-7
    )

    return hist.tolist()


# ============================================================
# ALL FEATURES
# ============================================================

def extract_all_features(image):

    return (

        extract_edge_features(image)

        +

        extract_color_features_hsv(image)

        +

        extract_color_features_lab(image)

        +

        extract_texture_features(image)

        +

        extract_lbp_features(image)

    )


# ============================================================
# SVM PREDICTION
# ============================================================

def predict_svm(image):

    svm_model, label_encoder = load_svm()

    features = extract_all_features(
        image
    )

    features = np.array(
        features
    ).reshape(1,-1)

    probs = svm_model.predict_proba(
        features
    )[0]

    idx = np.argmax(probs)

    prediction = (
        label_encoder
        .inverse_transform([idx])[0]
    )

    confidence = float(
        probs[idx]
    )

    return (
        prediction,
        confidence,
        probs
    )



# ============================================================
# CNN PREDICTION
# ============================================================

def predict_cnn(image):

    model = load_cnn()

    x = cnn_transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(x)

        probs = torch.softmax(
            output,
            dim=1
        ).cpu().numpy()[0]

    idx = np.argmax(probs)

    prediction = CLASSES[idx]

    confidence = float(probs[idx])

    return (
        prediction,
        confidence,
        probs
    )


# ============================================================
# RESNET50 PREDICTION
# ============================================================

def predict_resnet(image):

    model = load_resnet()

    x = resnet_transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(x)

        probs = torch.softmax(
            output,
            dim=1
        ).cpu().numpy()[0]

    idx = np.argmax(probs)

    prediction = CLASSES[idx]

    confidence = float(probs[idx])

    return (
        prediction,
        confidence,
        probs
    )


# ============================================================
# GENERATE GRADCAM
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    target_layer
):

    gradients = []
    activations = []

    def forward_hook(
        module,
        inp,
        output
    ):
        activations.append(output)

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        gradients.append(
            grad_output[0]
        )

    fh = target_layer.register_forward_hook(
        forward_hook
    )

    bh = target_layer.register_full_backward_hook(
        backward_hook
    )

    output = model(image_tensor)

    pred_class = output.argmax(
        dim=1
    )

    model.zero_grad()

    output[:, pred_class].backward()

    grads = gradients[0]

    acts = activations[0]

    weights = grads.mean(
        dim=(2,3),
        keepdim=True
    )

    cam = (
        weights * acts
    ).sum(dim=1)

    cam = F.relu(cam)

    cam = cam.squeeze()

    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(
        cam,
        (224,224)
    )

    cam = cam - cam.min()

    cam = cam / (
        cam.max() + 1e-8
    )

    fh.remove()
    bh.remove()

    return cam


# ============================================================
# OVERLAY HEATMAP
# ============================================================

def overlay_heatmap(
    image,
    cam
):

    image_np = np.array(
        image.resize(
            (224,224)
        )
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_np,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


# ============================================================
# CNN GRADCAM
# ============================================================

def get_cnn_gradcam(image):

    model = load_cnn()

    x = cnn_transform(image).unsqueeze(0)

    cam = generate_gradcam(
        model,
        x,
        model.features[12]
    )

    heatmap = overlay_heatmap(
        image,
        cam
    )

    return heatmap


# ============================================================
# RESNET50 GRADCAM
# ============================================================

def get_resnet_gradcam(image):

    model = load_resnet()

    x = resnet_transform(image).unsqueeze(0)

    cam = generate_gradcam(
        model,
        x,
        model.layer4[2]
    )

    heatmap = overlay_heatmap(
        image,
        cam
    )

    return heatmap


# ============================================================
# PROBABILITY CHART
# ============================================================

def plot_probability_chart(
    probs,
    title
):

    prob_df = pd.DataFrame({

        "Class": CLASSES,

        "Probability": probs

    })

    fig = px.bar(

        prob_df,

        x="Class",

        y="Probability",

        title=title

    )

    return fig

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🩺 DermaScan Interactive Screening")

st.markdown(
    """
    Upload a dermoscopic skin lesion image and compare predictions from:

    - CNN
    - ResNet50
    - SVM (95 handcrafted features)

    Includes:
    - Probability distribution
    - Grad-CAM explainability
    - Model confidence comparison
    """
)

uploaded = st.file_uploader(
    "Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)

selected_models = st.multiselect(
    "Select Models",
    ["CNN", "ResNet50", "SVM"],
    default=["CNN", "ResNet50", "SVM"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    if st.button("Run Screening"):

        results = []

        # =====================================================
        # CNN
        # =====================================================

        if "CNN" in selected_models:

            st.header("CNN")

            pred, conf, probs = predict_cnn(image)

            results.append([
                "CNN",
                pred,
                round(conf, 4)
            ])

            st.success(
                f"Prediction: {pred}"
            )

            st.write(
                f"Confidence: {conf:.4f}"
            )

            fig = plot_probability_chart(
                probs,
                "CNN Probability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            try:

                heatmap = get_cnn_gradcam(
                    image
                )

                st.subheader(
                    "CNN Grad-CAM"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        image,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        heatmap,
                        caption="CNN Grad-CAM",
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"CNN Grad-CAM Error: {e}"
                )

        # =====================================================
        # RESNET50
        # =====================================================

        if "ResNet50" in selected_models:

            st.header("ResNet50")

            pred, conf, probs = predict_resnet(
                image
            )

            results.append([
                "ResNet50",
                pred,
                round(conf, 4)
            ])

            st.success(
                f"Prediction: {pred}"
            )

            st.write(
                f"Confidence: {conf:.4f}"
            )

            fig = plot_probability_chart(
                probs,
                "ResNet50 Probability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            try:

                heatmap = get_resnet_gradcam(
                    image
                )

                st.subheader(
                    "ResNet50 Grad-CAM"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        image,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        heatmap,
                        caption="ResNet50 Grad-CAM",
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"ResNet50 Grad-CAM Error: {e}"
                )

        # =====================================================
        # SVM
        # =====================================================

        if "SVM" in selected_models:

            st.header("SVM")

            try:

                pred, conf, probs = predict_svm(
                    image
                )

                results.append([
                    "SVM",
                    pred,
                    round(conf, 4)
                ])

                st.success(
                    f"Prediction: {pred}"
                )

                st.write(
                    f"Confidence: {conf:.4f}"
                )

                svm_model, encoder = load_svm()

                svm_df = pd.DataFrame({

                    "Class": encoder.classes_,

                    "Probability": probs

                })

                fig = px.bar(
                    svm_df,
                    x="Class",
                    y="Probability",
                    title="SVM Probability Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"SVM Error: {e}"
                )

        # =====================================================
        # RESULTS TABLE
        # =====================================================

        st.header("Model Comparison")

        result_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Prediction",
                "Confidence"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # =====================================================
        # CONFIDENCE COMPARISON
        # =====================================================

        if len(result_df) > 0:

            fig = px.bar(

                result_df,

                x="Model",

                y="Confidence",

                color="Model",

                title="Confidence Comparison"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =====================================================
        # FINAL DECISION
        # =====================================================

        if len(result_df) > 0:

            final_pred = result_df.loc[
                result_df["Confidence"].idxmax()
            ]["Prediction"]

            st.success(
                f"Final Highest Confidence Prediction: {final_pred}"
            )

