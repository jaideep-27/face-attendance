# Face Recognition Attendance System

A real-time attendance management system using face recognition technology, built with Python and Streamlit.

## Features

- Real-time face detection and recognition
- Easy addition of new faces to the system
- Attendance tracking with timestamp
- Downloadable attendance records in CSV format
- User-friendly interface

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Use the sidebar menu to:
   - Take attendance (Home)
   - Add new faces to the system
   - View and download attendance records

## Directory Structure

- `app.py`: Main application file
- `requirements.txt`: Required Python packages
- `known_faces/`: Directory containing registered face images
- `README.md`: Project documentation

## Notes

- Make sure your webcam is properly connected and accessible
- Good lighting conditions will improve face recognition accuracy
- Each person should be registered with a clear, front-facing photo
