# 🎯 AI Face Attendance System

A modern, intelligent face recognition-based attendance system built with Python and Streamlit.

![AI Face Attendance System](https://img.shields.io/badge/AI-Face%20Attendance-FF6B6B?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF8E8E?style=for-the-badge&logo=streamlit&logoColor=white)
![dlib](https://img.shields.io/badge/dlib-FFE66D?style=for-the-badge&logo=python&logoColor=black)

## ✨ Features

### 🎯 Core Functionality
- **Real-time Face Recognition** - Instant face detection and recognition
- **High Accuracy** - Powered by dlib's state-of-the-art face recognition model
- **Adjustable Confidence Threshold** - Fine-tune recognition accuracy
- **Persistent Storage** - Attendance records saved automatically

### 🎨 Modern UI/UX
- **Dark Theme** - Easy on the eyes with a modern dark interface
- **Responsive Design** - Works seamlessly on all screen sizes
- **Interactive Elements** - Smooth animations and transitions
- **Glassmorphism Effects** - Beautiful frosted glass aesthetics

### 📊 Data Management
- **CSV Export** - Download attendance records as CSV
- **Statistics Dashboard** - View attendance metrics at a glance
- **Persistent Storage** - Records saved between sessions
- **Easy Data Management** - Add or remove faces easily

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
Webcam
```

### 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/face-attendance-app.git
cd face-attendance-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required models**
```bash
python download_models.py
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🎮 Usage

### 1. 📸 Take Attendance
- Launch the app and navigate to "Take Attendance"
- Adjust recognition confidence if needed
- Face the camera and wait for recognition
- Attendance is recorded automatically

### 2. 👤 Add New Face
- Go to "Add New Face" section
- Enter the person's name
- Upload a clear face photo
- Follow the on-screen instructions

### 3. 📊 View Records
- Check "View Records" section
- See attendance statistics
- Download records as CSV
- Track attendance history

## ⚙️ Configuration

### Recognition Settings
- Adjust confidence threshold (0.0 - 1.0)
- Lower values = stricter matching
- Recommended: 0.5 - 0.7

### Camera Settings
- Automatic camera detection
- Multiple camera support
- Fallback options available

## 🛡️ Privacy & Security

- Local processing - No cloud uploads
- Data stored locally
- Encrypted face encodings
- No personal data sharing

## 🎨 Customization

The app features a modern UI with:
- Gradient accents
- Smooth animations
- Dark theme
- Responsive design
- Custom CSS styling

## 📝 Dependencies

- **streamlit** - Web interface
- **opencv-python-headless** - Image processing
- **dlib** - Face recognition
- **numpy** - Numerical operations
- **pandas** - Data management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- dlib for face recognition
- Streamlit for the web framework
- OpenCV for image processing

---

<p align="center">
Made with ❤️ by [Your Name]
</p>
