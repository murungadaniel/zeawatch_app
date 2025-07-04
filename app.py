import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from datetime import datetime
import base64
from streamlit_option_menu import option_menu
import io
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download, login

# Import basic libraries for email functionality
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# # Add these lines at the top of the file, right after the imports
# import os
# # Force TensorFlow to use CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# # Limit TensorFlow memory growth
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(f"Memory growth setting error: {e}")
# # Set memory limit for TensorFlow
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# Set page configuration
st.set_page_config(
    page_title="ZeaWatch",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load background image if it exists, otherwise use a color
try:
    add_bg_from_local('background.png')
except:
    st.markdown(
        """
        <style>
        /* Main background color - light green */
        /* background-color: #e8f5e9; */

        .stApp {
            background-color: #ccffff; 
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #e8f5e9;
            padding-top: 1rem;
            border-right: 1px solid #1b5e20;
        }
        
        /* Sidebar text color */
        [data-testid="stSidebar"] .css-1d391kg, 
        [data-testid="stSidebar"] .css-1wrcr25,
        [data-testid="stSidebar"] h2 {
            color: black !important;
        }
        
        /* Sidebar menu styling */
        .nav-link {
            background-color: rgba(255, 255, 255, 0.1) !important;
            margin-bottom: 0.3rem !important;
            border-radius: 4px !important;
        }
        
        .nav-link.active {
            background-color: #81c784 !important;
            color: #1b5e20 !important;
            font-weight: bold !important;
        }
        
        .nav-link:hover {
            background-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        /* Form input styling for better visibility */
        [data-testid="stTextInput"] input, 
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div,
        .stFileUploader {
            background-color: white !important;
            border: 2px solid #81c784 !important;
            border-radius: 4px !important;
            padding: 0.5rem !important;
            font-size: 1rem !important;
            color: #333 !important;
        }
        
        [data-testid="stTextInput"] input:focus, 
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stSelectbox"] div:focus {
            border-color: #2e7d32 !important;
            box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.2) !important;
        }
        
        /* Radio buttons styling */
        [data-testid="stRadio"] label {
            background-color: white;
            padding: 10px 15px;
            border-radius: 4px;
            border: 1px solid #81c784;
            margin-right: 10px;
            font-weight: 500;
        }
        
        /* File uploader styling */
        .stFileUploader button {
            background-color: #81c784 !important;
            color: #1b5e20 !important;
        }
        
        /* Headers styling */
        .main-header {
            font-size: 2.5rem;
            color: #1b5e20;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            animation: fadeIn 1.5s;
        }
        
        .sub-header {
            font-size: 1.8rem;
            color: #2e7d32;
            margin-bottom: 1rem;
            text-align: center;
            animation: slideIn 1s;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            animation: fadeIn 1s;
            border-left: 5px solid #4CAF50;
        }
        
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            animation: slideUp 0.5s;
            border-top: 5px solid #4CAF50;
        }
        
        /* Status colors */
        .healthy {
            color: #2e7d32;
            font-weight: bold;
        }
        
        .disease {
            color: #c62828;
            font-weight: bold;
        }
        
        .not-leaf {
            color: #ff6f00;
            font-weight: bold;
        }
        
        /* Recommendation box */
        .recommendation {
            background-color: #f1f8e9;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            border-left: 4px solid #2e7d32;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        /* Button styling */
        .stButton button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            padding: 10px 24px !important;
            text-align: center !important;
            text-decoration: none !important;
            display: inline-block !important;
            font-size: 16px !important;
            margin: 4px 2px !important;
            border-radius: 4px !important;
            cursor: pointer !important;
            transition: all 0.3s !important;
            font-weight: bold !important;
        }
        
        .stButton button:hover {
            background-color: #45a049 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        }
        
        /* Info, success, warning boxes */
        .stAlert {
            border-radius: 4px !important;
            border-left: 5px solid !important;
        }
        
        .stAlert.success {
            border-left-color: #4CAF50 !important;
        }
        
        .stAlert.warning {
            border-left-color: #ff9800 !important;
        }
        
        .stAlert.error {
            border-left-color: #f44336 !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f1f8e9 !important;
            border-radius: 4px !important;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            border-top: 3px solid #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Initialize session state for scan history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# Load the model
@st.cache_resource
# def load_cnn_model():
#     try:
#         # First try to load the model
#         model = load_model('disease_classifier.h5')
#         return model
#     except Exception as e:
#         # If loading fails, provide detailed error
#         st.error(f"Error loading model: {e}")
#         import traceback
#         st.error(f"Detailed error: {traceback.format_exc()}")
        
#         # Try to check if the model file exists
#         if not os.path.exists('disease_classifier.h5'):
#             st.error("Model file 'disease_classifier.h5' not found. Please make sure it's in the correct directory.")
        
#         # Return None to indicate failure
#         return None

# model = load_cnn_model()

# Authenticate with Hugging Face using API Key (safer via environment variable)
# HF_TOKEN = os.getenv("HF_TOKEN")   # Set this in GitHub Actions or local .env

# try:
#     MF_TOKEN = st.secrets.get("zenmatch", {}).get("token")
#     if not MF_TOKEN:
#         st.error("Missing Zenmatch token!")
#         st.stop()
        
# except Exception as e:
#     st.error(f"Failed to load token: {e}")
#     st.stop()

@st.cache_resource
def load_cnn_model():
    try:
        # Download the model from Hugging Face Hub
        # Removed HF_TOKEN since you mentioned you don't have it
        # The model will download if public, otherwise will fail
        model_path = hf_hub_download(
            repo_id="simuyu/zeawatch_model",
            filename="disease_classifier.h5",
            token=None  # No token needed for public models
        )
        
        # Load the model
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check:")
        st.error("1. The repository 'simuyu/zeawatch_model' exists and is public")
        st.error("2. The filename 'disease_classifier.h5' is correct")
        st.error("3. Your internet connection is working")
        import traceback
        st.code(traceback.format_exc())
        return None
try:
    MF_TOKEN = st.secrets.get("zenmatch", {}).get("token")
    if not MF_TOKEN:
        st.error("Missing Zenmatch token!")
        st.stop()
        
except Exception as e:
    st.error(f"Failed to load token: {e}")
    st.stop()

# Load model
model = load_cnn_model()
if model is None:
    st.stop()

# Class labels
class_labels = ['Healthy', 'Gray Spot Leaf', 'Blight', 'Common Rust']

# Disease recommendations
disease_recommendations = {
    'Gray Spot Leaf': """
    ### Recommendations for Gray Spot Leaf:
    1. **Remove infected leaves** to prevent spread
    2. **Apply fungicide** specifically designed for gray leaf spot
    3. **Improve air circulation** around plants
    4. **Rotate crops** in the next season
    5. **Maintain proper spacing** between plants
    """,
    
    'Blight': """
    ### Recommendations for Blight:
    1. **Remove and destroy infected plants** immediately
    2. **Apply copper-based fungicides** as a preventative measure
    3. **Ensure proper drainage** in your field
    4. **Avoid overhead watering** to keep foliage dry
    5. **Plant resistant varieties** in the future
    """,
    
    'Common Rust': """
    ### Recommendations for Common Rust:
    1. **Apply fungicide** at first sign of infection
    2. **Remove infected leaves** carefully
    3. **Increase plant spacing** for better air circulation
    4. **Avoid overhead irrigation**
    5. **Plant rust-resistant varieties** in future seasons
    """
}

# Function to check if image is a maize leaf
def is_maize_leaf(img):
    """
    Determine if the image contains a maize leaf using color and texture analysis.
    This is a simplified approach - in a production app, you would use a dedicated
    image classification model for this task.
    """
    try:
        # Convert PIL image to cv2 format
        img_cv = np.array(img.convert('RGB'))
        img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
        
        # Resize for consistent processing
        img_cv = cv2.resize(img_cv, (224, 224))
        
        # Convert to HSV color space for better color analysis
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define range for green color (typical for plant leaves)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green areas
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_percentage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
        
        # Calculate texture features using Laplacian
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Check if the image has leaf-like characteristics
        # These thresholds can be adjusted based on testing
        is_leaf = green_percentage > 15 and texture_variance > 50
        
        # Calculate confidence based on green percentage and texture
        confidence = min(green_percentage / 30, 1.0) * 0.7 + min(texture_variance / 500, 1.0) * 0.3
        
        return is_leaf, confidence
        
    except Exception as e:
        st.error(f"Error in leaf detection: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return False, 0.0

# Preprocess the image for the model
def preprocess_image(img):
    try:
        # Convert to RGB to ensure compatibility
        img = img.convert('RGB')
        # Resize with antialiasing for better results
        img = img.resize((224, 224), Image.LANCZOS)
        # Use numpy directly instead of keras preprocessing
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

# Predict the class of the image
def predict_disease(img):
    if model is None:
        return "Model not loaded", 0
    
    try:
        # First check if the image is a maize leaf
        is_leaf, leaf_confidence = is_maize_leaf(img)
        
        if not is_leaf:
            return "Not a Maize Leaf", leaf_confidence
        
        # If it is a leaf, proceed with disease classification
        img_array = preprocess_image(img)
        if img_array is None:
            return "Image preprocessing failed", 0
            
        # Add a check for model readiness
        if not hasattr(model, 'predict'):
            return "Model not properly initialized", 0
            
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class = class_labels[predicted_class_index]
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return "Error", 0

# Save scan to history
def save_to_history(image, prediction, confidence, timestamp):
    # Convert image to bytes for storage
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    st.session_state.scan_history.append({
        'image': img_bytes,
        'prediction': prediction,
        'confidence': confidence,
        'timestamp': timestamp
    })

# Main navigation
with st.sidebar:
    st.image("leaves/icon.png", width=100)
    st.markdown("<h2 style='text-align: center; color: black;'>ZeaWatch</h2>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Scan Leaf", "History", "Contact"],
        icons=["house", "camera", "clock-history", "envelope"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "black", "font-size": "16px"}, 
            "nav-link": {"color": "black", "font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#2e7d32", "color": "#1b5e20", "font-weight": "bold"},
        }
    )

# Home page
if selected == "Home":
    st.markdown("<h1 class='main-header'>Welcome to ZeaWatch</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        ## About This App
        
        This application helps farmers identify common maize leaf diseases using artificial intelligence. 
        Our CNN model can detect:
        
        - Healthy maize leaves
        - Gray Spot Leaf disease
        - Blight
        - Common Rust
        
        The app first checks if your image contains a maize leaf before analyzing for diseases.
        Simply upload an image or take a photo of a maize leaf, and the app will analyze it and provide recommendations if a disease is detected.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        ## How to Use
        
        1. Navigate to the **Scan Leaf** page
        2. Upload an image or take a photo of a maize leaf
        3. The app will first verify if it's a maize leaf
        4. If confirmed, it will analyze for diseases
        5. View recommendations if a disease is detected
        6. Check your scan history in the **History** page
        7. Contact us through the **Contact** page if you need assistance
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        # st.markdown("### Quick Scan")
        # uploaded_file = st.file_uploader("Upload a maize leaf image", type=["jpg", "jpeg", "png"])
        
        # if uploaded_file is not None:
        #     st.markdown(f"<a href='#' onclick=\"window.open('Scan Leaf'); return false;\">Go to Scan Page</a>", unsafe_allow_html=True)
        # st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Disease Statistics")
        
        # Simple chart showing disease distribution from history
        if st.session_state.scan_history:
            disease_counts = {}
            for scan in st.session_state.scan_history:
                disease = scan['prediction']
                if disease in disease_counts:
                    disease_counts[disease] += 1
                else:
                    disease_counts[disease] = 1
            
            chart_data = pd.DataFrame({
                'Disease': list(disease_counts.keys()),
                'Count': list(disease_counts.values())
            })
            
            st.bar_chart(chart_data.set_index('Disease'))
        else:
            st.info("No scan history available yet.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Scan Leaf page
elif selected == "Scan Leaf":
    st.markdown("<h1 class='main-header'>Scan Maize Leaf</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Upload or Capture Maize Leaf Image")
    
    # Option to upload image or take photo
    option = st.radio("Choose input method:", ["Upload Image", "Take Photo"])
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a maize leaf image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Leaf"):
                with st.spinner("Analyzing..."):
                    # Make prediction
                    prediction, confidence = predict_disease(image)
                    
                    # Save to history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_to_history(image, prediction, confidence, timestamp)
                    
                    # Display results
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("## Analysis Results")
                    
                    if prediction == "Not a Maize Leaf":
                        st.markdown(f"<h3>Status: <span class='not-leaf'>{prediction}</span></h3>", unsafe_allow_html=True)
                        st.markdown(f"Confidence: {confidence:.2%}")
                        st.warning("This image does not appear to contain a maize leaf. Please upload an image of a maize leaf for disease classification.")
                    elif prediction == "Healthy":
                        st.markdown(f"<h3>Status: <span class='healthy'>{prediction}</span></h3>", unsafe_allow_html=True)
                        st.markdown(f"Confidence: {confidence:.2%}")
                        st.success("Good news! Your maize plant appears to be healthy. Continue with your current care routine.")
                    else:
                        st.markdown(f"<h3>Status: <span class='disease'>{prediction}</span></h3>", unsafe_allow_html=True)
                        st.markdown(f"Confidence: {confidence:.2%}")
                        st.warning(f"Your maize plant shows signs of {prediction}.")
                        
                        # Show recommendations
                        if prediction in disease_recommendations:
                            st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
                            st.markdown(disease_recommendations[prediction], unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    else:  # Take Photo option
        st.markdown("""
        ### Take a Photo
        Please allow camera access when prompted.
        """)
        
        # Create columns
        col1, col2 = st.columns(2,border=True,vertical_alignment="bottom")
        with col1:
            # Camera input
            camera_image = st.camera_input("Take a picture")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                
                if st.button("Analyze Leaf"):
                    with st.spinner("Analyzing..."):
                        # Make prediction
                        prediction, confidence = predict_disease(image)
                        
                        # Save to history
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_to_history(image, prediction, confidence, timestamp)
                        
                        # Display results
                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        st.markdown("## Analysis Results")
                        
                        if prediction == "Not a Maize Leaf":
                            st.markdown(f"<h3>Status: <span class='not-leaf'>{prediction}</span></h3>", unsafe_allow_html=True)
                            st.markdown(f"Confidence: {confidence:.2%}")
                            st.warning("This image does not appear to contain a maize leaf. Please take a photo of a maize leaf for disease classification.")
                        elif prediction == "Healthy":
                            st.markdown(f"<h3>Status: <span class='healthy'>{prediction}</span></h3>", unsafe_allow_html=True)
                            st.markdown(f"Confidence: {confidence:.2%}")
                            st.success("Good news! Your maize plant appears to be healthy. Continue with your current care routine.")
                        else:
                            st.markdown(f"<h3>Status: <span class='disease'>{prediction}</span></h3>", unsafe_allow_html=True)
                            st.markdown(f"Confidence: {confidence:.2%}")
                            st.warning(f"Your maize plant shows signs of {prediction}.")
                            
                            # Show recommendations
                            if prediction in disease_recommendations:
                                st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
                                st.markdown(disease_recommendations[prediction], unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add examples of maize leaves
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Examples of Maize Leaves")
    st.markdown("""
    For best results, please upload images similar to these examples:
    
    - Clear, well-lit photos of maize leaves
    - Leaves should be the main focus of the image
    - Try to capture the entire leaf when possible
    - For diseased leaves, make sure the symptoms are visible
    """)
    
    # Example images would go here in a real app
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### Healthy")
        st.image("leaves/Corn_Health.jpg", width=150)

    with col2:
        st.markdown("### Gray Spot")
        st.image("leaves/Corn_Gray_Spot.jpg", width=150)
    with col3:
        st.markdown("### Blight")
        st.image("leaves/Corn_Blight.jpg", width=150)
    with col4:
        st.markdown("### Common Rust")
        st.image("leaves/Corn_Common_Rust.jpg", width=150)
    
    st.markdown("</div>", unsafe_allow_html=True)

# History page
elif selected == "History":
    st.markdown("<h1 class='main-header'>Scan History</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    if not st.session_state.scan_history:
        st.info("No scan history available yet. Start by scanning some maize leaves!")
    else:
        # Add clear history button
        if st.button("Clear History"):
            st.session_state.scan_history = []
            st.success("History cleared successfully!")
            # st.experimental_rerun()
            st.rerun()
        
        # Display history in reverse chronological order
        for i, scan in enumerate(reversed(st.session_state.scan_history)):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Convert bytes back to image
                img = Image.open(io.BytesIO(scan['image']))
                st.image(img, width=150)
            
            with col2:
                st.markdown(f"**Scan #{len(st.session_state.scan_history) - i}**")
                st.markdown(f"**Date:** {scan['timestamp']}")
                
                if scan['prediction'] == "Not a Maize Leaf":
                    st.markdown(f"**Result:** <span class='not-leaf'>{scan['prediction']}</span>", unsafe_allow_html=True)
                elif scan['prediction'] == "Healthy":
                    st.markdown(f"**Result:** <span class='healthy'>{scan['prediction']}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Result:** <span class='disease'>{scan['prediction']}</span>", unsafe_allow_html=True)
                
                st.markdown(f"**Confidence:** {scan['confidence']:.2%}")
                
                # Show recommendations button for diseases
                if scan['prediction'] != "Healthy" and scan['prediction'] != "Not a Maize Leaf" and scan['prediction'] in disease_recommendations:
                    with st.expander("View Recommendations"):
                        st.markdown(disease_recommendations[scan['prediction']], unsafe_allow_html=True)
            
            st.markdown("---")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Contact page
elif selected == "Contact":
    st.markdown("<h1 class='main-header'>Contact Us</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ## Get in Touch
    
    Have questions or need assistance with maize diseases? Fill out the form below and our team will get back to you.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
    
    with col2:
        message = st.text_area("Message", height=150)
    
    if st.button("Submit"):
        if not name or not email or not message:
            st.error("Please fill in all required fields (name, email, and message).")
        else:
            try:
                # Create email
                msg = MIMEMultipart()
                sender_email = "murungadaniel2002@gmail.com"  # Replace with your Gmail
                app_password = "tpiwxcrqkwkszmgq"  # Replace with your app password
                recipient_email = "murungadaniel2002@gmail.com"
                
                msg['From'] = f"ZeaWatch Contact Form <{sender_email}>"
                msg['To'] = recipient_email
                msg['Subject'] = f"New Contact Form: from {name}"
                msg['Reply-To'] = email  # Set reply-to as the user's email
                
                # Email body with HTML formatting
                html = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h2 style="color: #2e7d32;">New Contact Form Submission</h2>
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Name:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{name}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Email:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{email}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Phone:</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{phone if phone else "Not provided"}</td>
                        </tr>
                    </table>
                    <h3 style="color: #2e7d32; margin-top: 20px;">Message:</h3>
                    <p style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #2e7d32;">{message}</p>
                </body>
                </html>
                """
                
                # Attach HTML content
                part = MIMEText(html, "html")
                msg.attach(part)
                
                # Connect to SMTP server
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, app_password)
                    server.send_message(msg)
                
                st.success("Thank you! Your message has been sent successfully. We'll get back to you soon.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your email settings and try again.")
    
    # st.markdown("</div>", unsafe_allow_html=True)
    
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    # st.markdown("""
    # ## Other Ways to Reach Us

    # - **Email:** support@maizeleafclassifier.com
    # - **Phone:** (+254) 792-575-861
    # - **Hours:** Monday-Friday, 8am - 6pm

    # Follow us on social media for updates and tips on plant health:
    # - [Instagram](https://www.instagram.com/zeawatchapp/)
    # - [Facebook](https://web.facebook.com/profile.php?id=61576308151719)
    # - [TikTok](https://www.tiktok.com/@zeawatchapp?lang=en)
    # """)
    # st.markdown("</div>", unsafe_allow_html=True)

    # First, include Font Awesome in your app
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .social-icon {
            margin-right: 10px;
            font-size: 18px;
        }
        .card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ## Other Ways to Reach Us

    - **Email:** zeawatchapp@gmail.com
    - **Phone:** (+254) 792-575-861
    - **Hours:** Monday-Friday, 8am - 6pm

    Follow us on social media for updates and tips on plant health:
    - <i class="fab fa-instagram social-icon"></i> [Instagram](https://www.instagram.com/zeawatchapp/)
    - <i class="fab fa-facebook social-icon"></i> [Facebook](https://web.facebook.com/profile.php?id=61576308151719)
    - <i class="fab fa-tiktok social-icon"></i> [TikTok](https://www.tiktok.com/@zeawatchapp?lang=en)
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 ZeaWatch | Developed with ZeaWatch for Farmers</p>
</div>
""", unsafe_allow_html=True)
