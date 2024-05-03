import tensorflow
import keras
import numpy as np
import cv2
import os


model = keras.models.load_model("saved_models/realFaces_vs_faceSynthetics.keras")

def preprocess_image(image_path, target_size=(64, 64)):
    # Attempt to load the image from the specified path
    # print(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image at path {image_path}. Check file path and integrity.")
        return None  # Return None if the image could not be loaded
    # Resize, flatten, and normalize the image
    # image = cv2.resize(image, target_size)
    # image = image.flatten()
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

def scan_folder(folder_path):
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


images = scan_folder('sample_images')

input_vectors = np.array([img for img in (preprocess_image(path) for path in images) if img is not None])

predictions = np.argmax(model.predict(input_vectors), axis=1)

for i, image in enumerate(images):

    decision = "REAL IMAGE" if predictions[i] == 0 else "AI GENERATED IMAGE"
    output = f"given image {image}, this image is a {decision}"
    print(output) 

