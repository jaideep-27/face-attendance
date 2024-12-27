import urllib.request
import bz2
import os

def download_and_extract_model(url, filename):
    """Download and extract a model file"""
    print(f"Downloading {filename}...")
    
    # Download the compressed file
    compressed_path = f"{filename}.bz2"
    urllib.request.urlretrieve(url, compressed_path)
    
    # Extract the file
    print(f"Extracting {filename}...")
    with bz2.BZ2File(compressed_path) as fr, open(filename, "wb") as fw:
        fw.write(fr.read())
    
    # Remove the compressed file
    os.remove(compressed_path)
    print(f"Successfully downloaded and extracted {filename}")

def main():
    # URLs for the model files
    models = {
        "shape_predictor_68_face_landmarks.dat": 
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat":
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    for filename, url in models.items():
        if not os.path.exists(filename):
            try:
                download_and_extract_model(url, filename)
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
        else:
            print(f"{filename} already exists")

if __name__ == "__main__":
    main()
