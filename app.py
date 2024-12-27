import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from PIL import Image

# Set page config
st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")

# Initialize session state variables
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = pd.DataFrame(columns=['Name', 'Time'])

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
            image = face_recognition.load_image_file(os.path.join(faces_dir, filename))
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])
    
    return known_faces, known_names

def mark_attendance(name):
    """Mark attendance for a recognized face"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    # Check if person is not already marked in the last minute
    if not st.session_state.attendance_df.empty:
        last_entry = st.session_state.attendance_df[st.session_state.attendance_df['Name'] == name]
        if not last_entry.empty:
            last_time = datetime.strptime(last_entry.iloc[-1]['Time'], "%H:%M:%S")
            if (now - last_time).seconds < 60:  # If less than 1 minute has passed
                return
    
    new_attendance = pd.DataFrame([[name, current_time]], columns=['Name', 'Time'])
    st.session_state.attendance_df = pd.concat([st.session_state.attendance_df, new_attendance], ignore_index=True)

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
        
        # Initialize camera
        camera = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        
        stop_button = st.button("Stop Camera")
        
        while not stop_button:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera")
                break
                
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1]
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Draw rectangles and names
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                    mark_attendance(name)
                
                # Draw rectangle and name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            FRAME_WINDOW.image(frame[:, :, ::-1])
        
        camera.release()
        
    elif choice == "Add New Face":
        st.subheader("Add New Face")
        
        name = st.text_input("Enter Name")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None and name:
            if not os.path.exists("known_faces"):
                os.makedirs("known_faces")
                
            # Save image
            image_path = os.path.join("known_faces", f"{name}.jpg")
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
            
            # Verify face can be detected
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                st.error("No face detected in the image. Please try another image.")
                os.remove(image_path)
            elif len(face_locations) > 1:
                st.error("Multiple faces detected. Please upload an image with only one face.")
                os.remove(image_path)
            else:
                st.success(f"Successfully added {name} to the system!")
                
    elif choice == "View Attendance":
        st.subheader("Attendance Records")
        
        if st.session_state.attendance_df.empty:
            st.write("No attendance records yet.")
        else:
            st.write(st.session_state.attendance_df)
            
            # Download attendance as CSV
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
