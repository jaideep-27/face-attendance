import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
from PIL import Image
import dlib
import json

# Set page config
st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")

# Constants
FACE_MATCH_THRESHOLD = 0.5
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
ATTENDANCE_FILE = "attendance_records.json"

# Initialize face detector and recognition models
try:
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(MODEL_PATH)
    face_rec_model = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    st.error(f"Error loading face recognition models: {str(e)}")
    st.error("Please ensure the model files are in the project directory")

def load_attendance_records():
    """Load attendance records from file"""
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                records = json.load(f)
                return pd.DataFrame(records)
        except Exception as e:
            st.error(f"Error loading attendance records: {str(e)}")
    return pd.DataFrame(columns=['Name', 'Time'])

def save_attendance_records(df):
    """Save attendance records to file"""
    try:
        records = df.to_dict('records')
        with open(ATTENDANCE_FILE, 'w') as f:
            json.dump(records, f)
    except Exception as e:
        st.error(f"Error saving attendance records: {str(e)}")

# Initialize session state variables
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = load_attendance_records()

def get_face_encoding(image):
    """Get face encoding using dlib"""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_detector(image)
        if not faces:
            return None
        
        # Get face landmarks
        shape = shape_predictor(image, faces[0])
        
        # Get face encoding
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
        return face_descriptor
    except Exception as e:
        st.error(f"Error getting face encoding: {str(e)}")
        return None

def load_known_faces():
    """Load known faces from the faces directory"""
    if not MODELS_AVAILABLE:
        return [], []
        
    known_faces = []
    known_names = []
    faces_dir = "known_faces"
    
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        return known_faces, known_names
    
    for filename in os.listdir(faces_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            try:
                image_path = os.path.join(faces_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    st.warning(f"Could not read image: {filename}")
                    continue
                
                encoding = get_face_encoding(image)
                if encoding is not None:
                    known_faces.append(encoding)
                    known_names.append(os.path.splitext(filename)[0])
                else:
                    st.warning(f"No face found in {filename}")
            except Exception as e:
                st.warning(f"Error processing {filename}: {str(e)}")
                continue
    
    return known_faces, known_names

def compare_faces(known_encoding, face_encoding, tolerance=0.6):
    """Compare face encodings"""
    try:
        if face_encoding is None or known_encoding is None:
            return False, 1.0
        distance = np.linalg.norm(known_encoding - face_encoding)
        return distance < tolerance, distance
    except Exception as e:
        st.error(f"Error comparing faces: {str(e)}")
        return False, 1.0

def mark_attendance(name):
    """Mark attendance for a recognized face"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    if not st.session_state.attendance_df.empty:
        last_entry = st.session_state.attendance_df[st.session_state.attendance_df['Name'] == name]
        if not last_entry.empty:
            last_time = datetime.strptime(last_entry.iloc[-1]['Time'], "%H:%M:%S")
            if (now - last_time).seconds < 60:
                return
    
    new_attendance = pd.DataFrame([[name, current_time]], columns=['Name', 'Time'])
    st.session_state.attendance_df = pd.concat([st.session_state.attendance_df, new_attendance], ignore_index=True)
    save_attendance_records(st.session_state.attendance_df)

def init_camera():
    """Initialize camera with fallback options"""
    camera = None
    for index in range(2):  # Try first two camera indices
        try:
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Try DirectShow
            if camera.isOpened():
                ret, frame = camera.read()
                if ret and frame is not None and frame.size > 0:
                    return camera
                camera.release()
        except Exception:
            if camera is not None:
                camera.release()
    
    # If DirectShow failed, try without it
    for index in range(2):
        try:
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                ret, frame = camera.read()
                if ret and frame is not None and frame.size > 0:
                    return camera
                camera.release()
        except Exception:
            if camera is not None:
                camera.release()
    
    return None

def main():
    st.title("Face Recognition Attendance System")
    
    if not MODELS_AVAILABLE:
        st.error("Face recognition models not available. Please check installation.")
        return
    
    menu = ["Home", "Add New Face", "View Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Take Attendance")
        
        known_faces, known_names = load_known_faces()
        
        if not known_faces:
            st.warning("No known faces found. Please add faces through the 'Add New Face' menu.")
            return
        
        try:
            camera = init_camera()
            if camera is None:
                st.error("No camera found. Please check your camera connection.")
                return
                
            FRAME_WINDOW = st.image([])
            
            # Add confidence threshold slider
            confidence_threshold = st.slider(
                "Recognition Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=FACE_MATCH_THRESHOLD,
                step=0.05,
                help="Lower value means stricter matching"
            )
            
            stop_button = st.button("Stop Camera")
            
            while not stop_button:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Get face encoding for current frame
                face_encoding = get_face_encoding(frame)
                
                if face_encoding is not None:
                    best_match_name = "Unknown"
                    best_match_distance = 1.0
                    
                    # Find best match
                    for known_face, known_name in zip(known_faces, known_names):
                        is_match, distance = compare_faces(known_face, face_encoding, confidence_threshold)
                        if is_match and distance < best_match_distance:
                            best_match_name = known_name
                            best_match_distance = distance
                    
                    # Draw rectangle around face
                    faces = face_detector(frame)
                    if faces:
                        face = faces[0]
                        if best_match_distance < confidence_threshold:
                            color = (0, 255, 0)  # Green for known faces
                            display_name = f"{best_match_name} ({(1-best_match_distance)*100:.1f}%)"
                            mark_attendance(best_match_name)
                        else:
                            color = (0, 0, 255)  # Red for unknown faces
                            display_name = "Unknown"
                        
                        cv2.rectangle(frame, 
                                    (face.left(), face.top()),
                                    (face.right(), face.bottom()),
                                    color, 2)
                        cv2.putText(frame, display_name,
                                  (face.left(), face.top() - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                  color, 2)
                
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            camera.release()
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")
        
    elif choice == "Add New Face":
        st.subheader("Add New Face")
        
        name = st.text_input("Enter Name")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None and name:
            try:
                if not os.path.exists("known_faces"):
                    os.makedirs("known_faces")
                
                # Convert uploaded file to image
                image = Image.open(image_file)
                img_array = np.array(image)
                
                # Verify face can be detected
                face_encoding = get_face_encoding(img_array)
                
                if face_encoding is None:
                    st.error("No face detected in the image. Please try another image.")
                else:
                    # Save image
                    image_path = os.path.join("known_faces", f"{name}.jpg")
                    image.save(image_path)
                    
                    # Show preview
                    faces = face_detector(img_array)
                    if faces:
                        face = faces[0]
                        cv2.rectangle(img_array,
                                    (face.left(), face.top()),
                                    (face.right(), face.bottom()),
                                    (0, 255, 0), 2)
                        st.image(img_array, caption=f"Processed image for {name}")
                    
                    st.success(f"Successfully added {name} to the system!")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                
    elif choice == "View Attendance":
        st.subheader("Attendance Records")
        
        if st.session_state.attendance_df.empty:
            st.write("No attendance records yet.")
        else:
            st.write(st.session_state.attendance_df)
            
            if st.button("Download Attendance CSV"):
                csv = st.session_state.attendance_df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f'attendance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )

if __name__ == '__main__':
    main()
