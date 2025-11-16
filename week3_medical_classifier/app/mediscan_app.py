"""
================================================================================
MEDISCAN - AI-Powered Chest X-Ray Classifier
================================================================================
Week 3 Project: Medical Image Classification with Transfer Learning & Grad-CAM
Author: ML Learning Journey
Model: ResNet50 Transfer Learning
Accuracy: 94.48% on validation set
================================================================================
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MediScan - AI X-Ray Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .normal-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# GRAD-CAM IMPLEMENTATION
# ================================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam, target_class, output

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Apply heatmap overlay on original image"""
    heatmap = np.uint8(255 * activation_map)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if org_img.shape[:2] != heatmap.shape[:2]:
        org_img = cv2.resize(org_img, (heatmap.shape[1], heatmap.shape[0]))
    
    if org_img.dtype != np.uint8:
        org_img = np.uint8(255 * org_img)
    
    superimposed_img = cv2.addWeighted(org_img, 0.7, heatmap, 0.3, 0)
    
    return heatmap, superimposed_img

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    img = tensor.clone()
    
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.uint8(255 * img)
    
    return img

# ================================================================================
# MODEL LOADING
# ================================================================================

@st.cache_resource
def load_model():
    """Load trained ResNet50 model"""
    device = torch.device('cpu')  # Use CPU for deployment
    
    # Load model architecture
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load trained weights - FIXED ABSOLUTE PATH
    model_path = 'C:/Users/audrey/Documents/ml_learning_lab/week3_medical_classifier/models/resnet50_best.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

@st.cache_resource
def get_gradcam(_model):
    """Initialize Grad-CAM"""
    target_layer = _model.layer4[-1]
    return GradCAM(_model, target_layer)

# ================================================================================
# IMAGE PREPROCESSING
# ================================================================================

def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ================================================================================
# PREDICTION FUNCTION
# ================================================================================

def predict_image(image, model, gradcam, transform, device):
    """Make prediction with Grad-CAM visualization"""
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    cam, pred_class, output = gradcam.generate_cam(input_tensor)
    
    # Get probabilities
    probs = F.softmax(output, dim=1).squeeze().cpu().detach().numpy()
    
    # Denormalize
    vis_img = denormalize_image(input_tensor.squeeze())
    
    # Apply colormap
    heatmap, overlay = apply_colormap_on_image(vis_img, cam)
    
    # Class names
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
    
    results = {
        'prediction': class_names[pred_class],
        'confidence': float(probs[pred_class] * 100),
        'normal_prob': float(probs[0] * 100),
        'pneumonia_prob': float(probs[1] * 100),
        'original_img': vis_img,
        'heatmap': heatmap,
        'overlay': overlay
    }
    
    return results

# ================================================================================
# MAIN APP
# ================================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🏥 MediScan</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Chest X-Ray Classification with Explainable AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=100)
        st.title("About MediScan")
        st.info("""
        **MediScan** uses deep learning to classify chest X-rays as **NORMAL** or **PNEUMONIA**.
        
        **Technology:**
        - ResNet50 Transfer Learning
        - Grad-CAM Visualization
        - 94.48% Validation Accuracy
        
        **Model trained on:**
        - 5,863 chest X-ray images
        - Transfer learning from ImageNet
        - 7 epochs of fine-tuning
        """)
        
        st.warning("⚠️ **Medical Disclaimer**: This is an educational project. NOT for clinical use. Always consult healthcare professionals.")
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload a Chest X-Ray Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image for classification"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        
        # Predict button
        if st.button("🔬 Analyze X-Ray", type="primary"):
            with st.spinner("Analyzing image... This may take a moment..."):
                # Load model
                model, device = load_model()
                gradcam = get_gradcam(model)
                transform = get_transforms()
                
                # Make prediction
                results = predict_image(image, model, gradcam, transform, device)
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Prediction box
                if results['prediction'] == 'NORMAL':
                    st.markdown(f"""
                    <div class="prediction-box normal-prediction">
                        ✅ Prediction: NORMAL
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box pneumonia-prediction">
                        ⚠️ Prediction: PNEUMONIA DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence scores
                st.subheader("🎯 Confidence Scores")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="NORMAL",
                        value=f"{results['normal_prob']:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="PNEUMONIA",
                        value=f"{results['pneumonia_prob']:.2f}%",
                        delta=None
                    )
                
                # Progress bars
                st.progress(results['normal_prob'] / 100)
                st.progress(results['pneumonia_prob'] / 100)
                
                # Visualizations
                st.markdown("---")
                st.subheader("🔍 Model Attention Visualization (Grad-CAM)")
                
                st.markdown("""
                <div class="info-box">
                <b>What is this?</b> Grad-CAM shows WHERE the AI model is looking. 
                Red/yellow areas indicate high attention, blue areas indicate low attention.
                </div>
                """, unsafe_allow_html=True)
                
                # Display three images side by side
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(results['original_img'], caption="Original X-Ray", use_container_width=True)
                
                with col2:
                    st.image(results['heatmap'], caption="Attention Heatmap", use_container_width=True)
                
                with col3:
                    st.image(results['overlay'], caption="Overlay", use_container_width=True)
                
                # Interpretation
                st.markdown("---")
                st.subheader("💡 Interpretation")
                
                if results['prediction'] == 'NORMAL':
                    st.success("""
                    **Normal X-Ray Detected**
                    
                    The model has classified this X-ray as NORMAL with {:.2f}% confidence.
                    The Grad-CAM visualization shows the model's attention is distributed across the lung fields,
                    indicating no concentrated areas of concern.
                    """.format(results['confidence']))
                else:
                    st.error("""
                    **Pneumonia Detected**
                    
                    The model has detected signs of pneumonia with {:.2f}% confidence.
                    The Grad-CAM visualization highlights the areas where the model detected abnormalities.
                    Red/yellow regions indicate areas of concern that influenced the prediction.
                    
                    **⚠️ Important:** This is an AI screening tool. Please consult a healthcare professional
                    for proper diagnosis and treatment.
                    """.format(results['confidence']))
    
    else:
        # Instructions
        st.info("""
        ### 👆 Get Started
        
        1. Upload a chest X-ray image using the file uploader above
        2. Click "Analyze X-Ray" to get the prediction
        3. View the results with Grad-CAM visualization
        
        **Supported formats:** PNG, JPG, JPEG
        """)
        
        # Example section
        st.markdown("---")
        st.subheader("📸 Example Results")
        
        st.markdown("""
        Here's what MediScan can do:
        
        - **Classify** chest X-rays as NORMAL or PNEUMONIA
        - **Explain** predictions with Grad-CAM heatmaps
        - **Show confidence** scores for transparency
        - **Visualize** where the AI is looking
        """)

if __name__ == "__main__":
    main()