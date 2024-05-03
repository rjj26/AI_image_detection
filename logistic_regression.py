# !pip install scikit-learn

# !pip3 install opencv-python-headless --prefer-binary
import cv2

print("computer vision cv2 version:", cv2.__version__)

# Importing a specific model (e.g., the RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier

# Importing a utility function (e.g., train_test_split)
from sklearn.model_selection import train_test_split

# Importing a metric to evaluate your model (e.g., accuracy_score)
from sklearn.metrics import accuracy_score

#classifying data based on real and AI generated (by folder)
import os

real_image_folders = ['afhq',
                      'celebahq',
                      'coco',
                      'ffhq',
                      'imagenet',
                      'landscape',
                      'lsun',
                      'metfaces']

real_image_filenames = []

AI_generated_folders = ['big_gan',
                        'cips',
                        'ddpm',
                        'denoising_diffusion_gan',
                        'diffusion_gan',
                        'face_synthetics',
                        'gansformer',
                        'gau_gan',
                        'generative_inpainting',
                        'glide',
                        'lama',
                        'latent_diffusion',
                        'mat',
                        'palette',
                        'pro_gan',
                        'projected_gan',
                        'sfhq',
                        'stable_diffusion',
                        'star_gan',
                        'stylegan1',
                        'stylegan2',
                        'stylegan3',
                        'taming_tranformer',
                        'vq_diffusion']

fake_image_filenames = []

def scan_folder(folder_path):
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

folder_path = '/Users/reesejohnson/Desktop/AI_vs_real_images/'

for folder in real_image_folders:
  real_image_filenames += scan_folder(folder_path+folder)

for folder in AI_generated_folders:
  fake_image_filenames += scan_folder(folder_path+folder)

print(len(real_image_filenames), "real images processed for training")
print(len(fake_image_filenames), "fake images processed for training")

#Generating Training Data
all_image_files = real_image_filenames + fake_image_filenames

import numpy as np
from sklearn.model_selection import train_test_split

#real images = 0, fake images = 1
ground_truth_labels = [0] * len(real_image_filenames) + [1]*len(fake_image_filenames)

#60% training data, 40% testing
x_train, x_test, y_train, y_test = train_test_split(all_image_files, ground_truth_labels, test_size=0.4, random_state=24)

#split testing data into 20% validation, 20% testing
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=24)


#attempting with a logistic regression first
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import cv2  # Assuming OpenCV is used for image processing


def preprocess_image(image_path, target_size=(64, 64)):
    # Attempt to load the image from the specified path
    # print(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image at path {image_path}. Check file path and integrity.")
        return None  # Return None if the image could not be loaded
    # Resize, flatten, and normalize the image
    # image = cv2.resize(image, target_size)
    image = image.flatten()
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image


# Assuming x_train, x_val, and x_test are lists of image paths
# Preprocess the images and filter out any that failed to load
#pulls batch images from all different types of 
def make_batch(x_data, y_data, batchsize):
  indices = np.random.permutation(len(y_data))
  x_data = np.array(x_data)[indices]
  y_data = np.array(y_data)[indices]

  real_count = 0
  fake_count = 0

  x_batch = []
  y_batch = []
  
  #ensuring # of real and fake images are equal in whatever batch
  index = 0
  while real_count < batchsize/2 and fake_count < batchsize/2:
    if y_data[index] == 0 and real_count < batchsize/2:
      x_batch.append(x_data[index])
      y_batch.append(y_data[index])
      real_count += 1
    elif y_data[index] == 1 and fake_count < batchsize/2:
      x_batch.append(x_data[index])
      y_batch.append(y_data[index])
      fake_count += 1
    
    index += 1

  return x_batch, y_batch
      

x_train_batch, y_train_batch = make_batch(x_train, y_train, 1000)
x_val_batch, y_val_batch = make_batch(x_val, y_val, 1000)
x_test_batch, y_test_batch = make_batch(x_test, y_test, 1000)


X_train_vect = np.array([img for img in (preprocess_image(path) for path in x_train_batch) if img is not None])
X_val_vect = np.array([img for img in (preprocess_image(path) for path in x_val_batch) if img is not None])
X_test_vect = np.array([img for img in (preprocess_image(path) for path in x_test_batch) if img is not None])

# Ensure the labels correspond to the filtered images
# This might require adjusting depending on how you handle images that failed to load
y_train_vect = np.array(y_train_batch)[:len(X_train_vect)]
y_val_vect = np.array(y_val_batch)[:len(X_val_vect)]
y_test_vect = np.array(y_test_batch)[:len(X_test_vect)]

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed for convergence
model.fit(X_train_vect, y_train_vect)

# Predict on the validation set
y_val_pred = model.predict(X_val_vect)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val_vect, y_val_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on the test set (after model selection and hyperparameter tuning)
y_test_pred = model.predict(X_test_vect)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test_vect, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")

