import tensorflow
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os


print("computer vision cv2 version:", cv2.__version__)

#classifying data based on real and AI generated (by folder)
real_image_folders = [#'afhq',
                      'celebahq',
                      # 'coco',
                      'ffhq'
                      # 'imagenet',
                      # 'landscape',
                      # 'lsun',
                      # 'metfaces'
]

##ffhq, celebahq  have real faces

real_image_filenames = []

AI_generated_folders = [#'big_gan',
                        # 'cips',
                        # 'ddpm',
                        'denoising_diffusion_gan',
                        'diffusion_gan',
                        'face_synthetics'
                        # 'gansformer',
                        # 'gau_gan',
                        # 'generative_inpainting',
                        # 'glide',
                        # 'lama',
                        # 'latent_diffusion',
                        # 'mat',
                        'palette',
                        # 'pro_gan',
                        # 'projected_gan',
                        'sfhq',
                        'stable_diffusion/stable-face',
                        'star_gan',
                        'stylegan1',
                        # 'stylegan2',
                        # 'stylegan3',
                        # 'taming_tranformer',
                        # 'vq_diffusion'
]
#pro gan has real and fake images, proj gan has an ffhq data set: projected_gan/proj/ffhq
#sfhq is all fake faces
#stablediffusion/stable-face
#star gan is good
#stylegan1

fake_image_filenames = []


def scan_folder(folder_path):
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

folder_path = '/Users/reesejohnson/Desktop/AI_vs_real_images/'
#/home/reese/artifact...

for folder in real_image_folders:
  real_image_filenames += scan_folder(folder_path+folder)

for folder in AI_generated_folders:
  fake_image_filenames += scan_folder(folder_path+folder)

print(len(real_image_filenames), "real images processed for training")
print(len(fake_image_filenames), "fake images processed for training")

#Generating Training Data
all_image_files = real_image_filenames + fake_image_filenames


#real images = 0, fake images = 1
ground_truth_labels = [0] * len(real_image_filenames) + [1]*len(fake_image_filenames)

#60% training data, 40% testing
x_train, x_test, y_train, y_test = train_test_split(all_image_files, ground_truth_labels, test_size=0.4, random_state=24)

#split testing data into 20% validation, 20% testing
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=24)


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

#initialize model
model = keras.models.Sequential()

# layer1 = keras.layers.Reshape( (28,28,1), input_shape=(28,28) )
#200x200pixel x RGB channel
# layer1 = keras.layers.Conv2D(10, [4, 4], activation='relu', input_shape=(200,200, 3))
# layer6 = keras.layers.Conv2D(20, [3, 3], activation='relu')
# layer2 = keras.layers.Conv2D(10, [4, 4], activation='relu')
# layer3 = keras.layers.Flatten()
# layer4 = keras.layers.Dense( 64, activation='relu')
# layer7 = keras.layers.Dropout(0.3)
# layer8 = keras.layers.Dense(16, activation='relu')
# layer5 = keras.layers.Dense( 2, activation='softmax' )


# layer1 = keras.layers.Reshape((40,100,1), input_shape=(40,100))
#40x100pixel x RGB channel
layer2 = keras.layers.Conv2D(32, [4, 4], activation='relu', input_shape=(200,200, 3))
# layer6 = keras.layers.Conv2D(6, [3, 3], activation='relu')
layer3 = keras.layers.Flatten()
# layer4 = keras.layers.Dense( 128, activation='relu' )
layer8 = keras.layers.Dense( 32, activation='relu' )
layer5 = keras.layers.Dense( 2, activation='softmax' )

# model.add( layer1 )
# model.add( layer6 )
model.add( layer2 )
model.add( layer3 )
# model.add( layer4 )
# model.add( layer7 )
model.add( layer8 )
model.add( layer5 )


model.compile( optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'] )

# early_stopping = keras.callbacks.EarlyStopping( monitor='val_loss', patience=30 )
# lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=10, verbose=1 )

model.summary()

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
  while real_count < batchsize/2 or fake_count < batchsize/2:
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
      

x_test_batch, y_test_batch = make_batch(x_test, y_test, 1000)
X_test_vect = np.array([img for img in (preprocess_image(path) for path in x_test_batch) if img is not None])
y_test_vect = np.array(y_test_batch)[:len(X_test_vect)]
y_test_vect = keras.utils.to_categorical(y_test_vect, num_classes=2)

#loop to generate multiple batches for the model
for i in range(50):

    x_train_batch, y_train_batch = make_batch(x_train, y_train, 1000)
    x_val_batch, y_val_batch = make_batch(x_val, y_val, 1000)

    X_train_vect = np.array([img for img in (preprocess_image(path) for path in x_train_batch) if img is not None])
    X_val_vect = np.array([img for img in (preprocess_image(path) for path in x_val_batch) if img is not None])

    # Ensure the labels correspond to the filtered images
    # This might require adjusting depending on how you handle images that failed to load
    y_train_vect = np.array(y_train_batch)[:len(X_train_vect)]
    y_val_vect = np.array(y_val_batch)[:len(X_val_vect)]

    #does the one hot encoding
    y_train_vect = keras.utils.to_categorical(y_train_vect, num_classes=2)
    y_val_vect = keras.utils.to_categorical(y_val_vect, num_classes=2)


    model.fit( X_train_vect, y_train_vect, validation_data=(X_val_vect, y_val_vect), batch_size=200, epochs=1, 
            verbose=1 ) #, callbacks=[early_stopping, lr_reduction]


model.save('realFaces_vs_allFakeFaces.keras')
test_loss, test_acc = model.evaluate( X_test_vect, y_test_vect, verbose=1 )

print('Final accuracy on test data: ', test_acc)
print('Final loss on test data: ', test_loss)

#best one so far:
#4 layers, input layer: layer1 = keras.layers.Conv2D(3, [4, 4], activation='relu', input_shape=(200,200, 3)), 50 tries, Final accuracy on test data:  0.6490147709846497


