import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
from PIL import Image
import dlib

# Set page config
st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")

# Initialize session state variables
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = pd.DataFrame(columns=['Name', 'Time'])

# Initialize face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    st.error("Please download the shape predictor file from dlib's website and place it in the project directory.")
    st.stop()

face_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_encoding(image):
    """Get face encoding using dlib directly"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    # Detect faces
    faces = face_detector(image)
    if not faces:
        return None
    
    # Get face landmarks
    shape = face_predictor(image, faces[0])
    
    # Get face encoding
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

def load_known_faces():
    """Load known faces from the faces directory"""
    known_faces = []
    known_names = []
    faces_dir = "known_faces"
    
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        return known_faces, known_names
    
    for filename in os.listdir(faces_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(faces_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            encoding = get_face_encoding(image)
            if encoding is not None:
                known_faces.append(encoding)
                known_names.append(os.path.splitext(filename)[0])
    
    return known_faces, known_names

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

def compare_faces(known_encoding, face_encoding, tolerance=0.6):
    """Compare face encodings"""
    if face_encoding is None:
        return False
    diff = np.linalg.norm(known_encoding - face_encoding)
    return diff < tolerance

def main():
    st.title("Face Recognition Attendance System")
    
    menu = ["Home", "Add New Face", "View Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Take Attendance")
        
        known_faces, known_names = load_known_faces()
        
        if not known_faces:
            st.warning("No known faces found. Please add faces through the 'Add New Face' menu.")
            return
        
        camera = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        
        stop_button = st.button("Stop Camera")
        
        while not stop_button:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera")
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get face encoding for current frame
            current_face_encoding = get_face_encoding(rgb_frame)
            
            if current_face_encoding is not None:
                # Compare with known faces
                for known_face, known_name in zip(known_faces, known_names):
                    if compare_faces(known_face, current_face_encoding):
                        mark_attendance(known_name)
                        # Draw rectangle around face
                        faces = face_detector(rgb_frame)
                        if faces:
                            face = faces[0]
                            cv2.rectangle(frame, 
                                        (face.left(), face.top()),
                                        (face.right(), face.bottom()),
                                        (0, 255, 0), 2)
                            cv2.putText(frame, known_name,
                                      (face.left(), face.top() - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                      (0, 255, 0), 2)
            
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        camera.release()
        
    elif choice == "Add New Face":
        st.subheader("Add New Face")
        
        name = st.text_input("Enter Name")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None and name:
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
                st.success(f"Successfully added {name} to the system!")
                
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
