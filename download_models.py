import requests
import bz2
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename + '.bz2', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Extracting {filename}...")
    with bz2.open(filename + '.bz2', 'rb') as source, open(filename, 'wb') as dest:
        dest.write(source.read())
    
    os.remove(filename + '.bz2')
    print(f"Successfully downloaded and extracted {filename}")

# URLs for the model files
shape_predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
face_rec_model_url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

# Download and extract shape predictor
download_file(shape_predictor_url, "shape_predictor_68_face_landmarks.dat")

# Download and extract face recognition model
download_file(face_rec_model_url, "dlib_face_recognition_resnet_model_v1.dat")

print("All model files have been downloaded and extracted successfully!")
