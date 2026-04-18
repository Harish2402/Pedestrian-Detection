# pip install streamlit-option-menu

import streamlit as st
import json
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# ==================== CONFIGURATION ====================
USERS_DB = "users.json"
BEST_MODEL = "runs_citypersons/yolov11_bifpn_pedestrian/weights/best.pt"
DEFAULT_IMAGE_DIR = "Citypersons/images/test"

# ==================== STREAMLIT VERSION COMPATIBILITY ====================
def display_image(image, caption=None):
    """Display image with version-compatible parameters"""
    try:
        # Try newer parameter first
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        # Fall back to older parameter
        st.image(image, caption=caption, use_column_width=True)

# ==================== AUTHENTICATION SYSTEM ====================
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON database"""
    if os.path.exists(USERS_DB):
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON database"""
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=4)

def register_user(username, email, password):
    """Register a new user"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    # Check if email already exists
    for user_data in users.values():
        if user_data.get('email') == email:
            return False, "Email already registered"
    
    users[username] = {
        'email': email,
        'password': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    save_users(users)
    return True, "Registration successful"

def authenticate_user(username, password):
    """Authenticate user credentials"""
    users = load_users()
    
    if username not in users:
        return False, "Username not found"
    
    if users[username]['password'] == hash_password(password):
        # Update last login
        users[username]['last_login'] = datetime.now().isoformat()
        save_users(users)
        return True, "Login successful"
    
    return False, "Incorrect password"

# ==================== UI STYLING ====================
def load_custom_css():
    """Apply custom CSS for professional look"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #2193b0, #6dd5ed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        
        .sub-header {
            font-size: 1.5rem;
            color: #2c3e50;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
        }
        
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .feature-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .success-message {
            padding: 1rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            color: #155724;
            margin: 1rem 0;
        }
        
        .error-message {
            padding: 1rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            color: #721c24;
            margin: 1rem 0;
        }
        
        .stButton>button {
            width: 100%;
            background: linear-gradient(120deg, #2193b0, #6dd5ed);
            color: white;
            border: none;
            padding: 0.75rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

# ==================== PROJECT INFORMATION ====================
def show_project_info():
    """Display comprehensive project information"""
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>YOLOv11 + BiFPN Pedestrian Detection System</h3>
        <p style="font-size: 1.1rem; line-height: 1.8;">
        This advanced computer vision system leverages the cutting-edge YOLOv11 architecture 
        enhanced with Bidirectional Feature Pyramid Network (BiFPN) for superior pedestrian detection 
        in urban environments. Trained on the Citypersons dataset, it achieves state-of-the-art 
        performance in challenging real-world scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>🏗️ Architecture Highlights</h4>
            <ul style="line-height: 2;">
                <li><strong>Model:</strong> YOLOv11n (Nano variant)</li>
                <li><strong>Classes:</strong> 1 (Pedestrian)</li>
                <li><strong>Input Resolution:</strong> Multi-scale (P3-P5)</li>
                <li><strong>Backbone:</strong> C3k2 + C2PSA modules</li>
                <li><strong>Neck:</strong> BiFPN-style fusion</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>🚀 Key Features</h4>
            <ul style="line-height: 2;">
                <li>Top-down + Bottom-up pathways</li>
                <li>Multi-scale feature fusion</li>
                <li>SPPF spatial pyramid pooling</li>
                <li>C2PSA attention mechanism</li>
                <li>Optimized for real-time inference</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==================== MODEL INFERENCE ====================
@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    if not os.path.exists(BEST_MODEL):
        st.error(f"❌ Model not found at: {BEST_MODEL}")
        st.stop()
    return YOLO(BEST_MODEL)

def run_inference(image_path, conf_threshold=0.4):
    """Run YOLOv11 inference on image"""
    model = load_model()
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, "Failed to load image"
    
    # Run inference
    results = model(img, conf=conf_threshold, verbose=False)
    
    # Get annotated image
    annotated = results[0].plot()
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Get detection statistics
    boxes = results[0].boxes
    num_detections = len(boxes)
    confidences = boxes.conf.cpu().numpy() if num_detections > 0 else []
    
    stats = {
        'num_detections': num_detections,
        'avg_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0,
        'max_confidence': float(np.max(confidences)) if len(confidences) > 0 else 0,
        'min_confidence': float(np.min(confidences)) if len(confidences) > 0 else 0
    }
    
    return original_rgb, annotated_rgb, stats

# ==================== LOGIN PAGE ====================
def login_page():
    """Render login/registration page"""
    st.markdown('<h1 class="main-header">🚶 Pedestrian Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if username and password:
                        success, message = authenticate_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            users = load_users()
                            st.session_state.email = users[username]['email']
                            st.success("✅ " + message)
                            st.rerun()
                        else:
                            st.error("❌ " + message)
                    else:
                        st.warning("⚠️ Please fill in all fields")
        
        with tab2:
            st.markdown("### Create New Account")
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="your.email@example.com")
                new_password = st.text_input("Password", type="password", placeholder="Choose a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                register = st.form_submit_button("Register")
                
                if register:
                    if new_username and new_email and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        elif '@' not in new_email:
                            st.error("❌ Invalid email format")
                        else:
                            success, message = register_user(new_username, new_email, new_password)
                            if success:
                                st.success("✅ " + message + " Please login now.")
                            else:
                                st.error("❌ " + message)
                    else:
                        st.warning("⚠️ Please fill in all fields")

# ==================== MAIN APPLICATION ====================
def main_app():
    """Main application interface"""
    load_custom_css()
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown(f'''
        <h1 class="main-header" style="text-align: left;"> 🚶 Detector
        </h1>
    ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f"**User:** {st.session_state.username}")
        if st.button("Logout", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Navigation
    # page = st.sidebar.radio("Navigation", ["📖 Project Info", "🔍 Detection", "👤 Profile"])
    from streamlit_option_menu import option_menu

    page = option_menu(
        menu_title=None,
        options=["Project Info", "Detection", "Profile"],
        icons=["book", "search", "person"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    
    if page == "Project Info":
        show_project_info()
        
    elif page == "Detection":
        st.markdown('<h2 class="sub-header">Pedestrian Detection Interface</h2>', unsafe_allow_html=True)
        
        # Image selection method
        input_method = st.radio("Select Input Method:", ["📁 Browse Directory", "📤 Upload Image"], horizontal=True)
        
        selected_image = None
        
        if input_method == "📁 Browse Directory":
            # Directory browser
            if os.path.exists(DEFAULT_IMAGE_DIR):
                image_files = list(Path(DEFAULT_IMAGE_DIR).rglob("*.png")) + \
                              list(Path(DEFAULT_IMAGE_DIR).rglob("*.jpg")) + \
                              list(Path(DEFAULT_IMAGE_DIR).rglob("*.jpeg"))
                
                if image_files:
                    # Create friendly display names
                    image_dict = {str(f.relative_to(DEFAULT_IMAGE_DIR)): str(f) for f in image_files}
                    
                    selected_name = st.selectbox(
                        "Select an image:",
                        options=list(image_dict.keys()),
                        format_func=lambda x: f"{x}"
                    )
                    
                    selected_image = image_dict[selected_name]
                else:
                    st.warning(f"⚠️ No images found in {DEFAULT_IMAGE_DIR}")
            else:
                st.error(f"❌ Directory not found: {DEFAULT_IMAGE_DIR}")
        
        else:  # Upload Image
            uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_image = temp_path
        
        # Detection settings
        with st.expander("⚙️ Detection Settings", expanded=True):
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Higher values = fewer but more confident detections"
            )
        
        # Run detection
        if selected_image:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_detection = st.button("🚀 Run Detection", type="primary")
            
            if run_detection:
                with st.spinner("🔄 Running inference..."):
                    original, annotated, stats = run_inference(selected_image, conf_threshold)
                    
                    if original is not None:
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📊 Detection Results")
                        
                        # Statistics
                        if isinstance(stats, dict):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Pedestrians Detected", stats['num_detections'])
                            with col2:
                                st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")
                            with col3:
                                st.metric("Max Confidence", f"{stats['max_confidence']:.2%}")
                            with col4:
                                st.metric("Min Confidence", f"{stats['min_confidence']:.2%}")
                        
                        st.markdown("---")
                        
                        # Side-by-side comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📷 Original Image")
                            display_image(original)
                        
                        with col2:
                            st.markdown("#### 🎯 Detection Results")
                            display_image(annotated)
                        
                        st.success("✅ Detection completed successfully!")
                    else:
                        st.error(f"❌ Error: {stats}")
    
    elif page == "Profile":
        st.markdown('<h2 class="sub-header">👤 User Profile</h2>', unsafe_allow_html=True)
        
        users = load_users()
        user_data = users.get(st.session_state.username, {})
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h1 style="font-size: 4rem; margin: 0;">👤</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="feature-box">
                <h3>Account Information</h3>
                <p><strong>Username:</strong> {st.session_state.username}</p>
                <p><strong>Email:</strong> {user_data.get('email', 'N/A')}</p>
                <p><strong>Member Since:</strong> {user_data.get('created_at', 'N/A')[:10]}</p>
                <p><strong>Last Login:</strong> {user_data.get('last_login', 'N/A')[:19] if user_data.get('last_login') else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== MAIN ENTRY POINT ====================
def main():
    """Application entry point"""
    st.set_page_config(
        page_title="Pedestrian Detection System",
        page_icon="🚶",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Route to appropriate page
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()