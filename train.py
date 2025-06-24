import random
import os
import cv2
import numpy as np
import math
import pickle
from glob import glob
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import ReLU, Add, Multiply, Input, LeakyReLU, Conv2D, concatenate, Conv2DTranspose, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



from tensorflow.keras.utils import Sequence


class ImageProcessor:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        image = self.resize_and_normalize(image, add_batch_dim=True)  # Always add batch dimension
        
        return image

    def resize_and_normalize(self, image, add_batch_dim=False):
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        
        if add_batch_dim:
            image = np.expand_dims(image, axis=0)  # Add batch dimension if specified
        
        return image
class CustomDataGenerator(Sequence):
    def __init__(self, Cover_dir, Secret_dir, batch_size, image_size=(256, 256)):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_processor = ImageProcessor(image_size=image_size)
        
        self.Cover_paths = glob(Cover_dir + "\\*.jpg", recursive=True)
        self.Secret_paths = glob(Secret_dir + "\\*.jpg", recursive=True)

    def __len__(self):
        return math.ceil(len(self.Cover_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_Cover_paths = self.Cover_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_Secret_paths = self.Secret_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_Cover_images = []
        batch_Secret_images = []

        for Cover_path, Secret_path in zip(batch_Cover_paths, batch_Secret_paths):
            try:
                # Load and preprocess Cover image without adding batch dimension
                image_cover = self.image_processor.load_and_preprocess_image(Cover_path)
                
                batch_Cover_images.append(image_cover)
                
                # Load and preprocess Secret image without adding batch dimension
                image_secret = self.image_processor.load_and_preprocess_image(Secret_path)
                batch_Secret_images.append(image_secret)

            except Exception as e:
                print(f"Error loading or processing image: {e}")
                continue

        batch_Cover_images = tf.convert_to_tensor(batch_Cover_images, dtype=tf.float32)
        batch_Cover_images=np.squeeze(batch_Cover_images, axis=1)
        batch_Secret_images = tf.convert_to_tensor(batch_Secret_images, dtype=tf.float32)
        batch_Secret_images=np.squeeze(batch_Secret_images, axis=1)
        return [batch_Cover_images, batch_Secret_images], [batch_Cover_images, batch_Secret_images]

# Define the PSNR metric function

def psnr(y_true, y_pred):
    # Extract the luminance (Y) channels from the true and predicted YUV images
    y_true_y, _, _ = tf.split(y_true, num_or_size_splits=3, axis=-1)
    y_pred_y, _, _ = tf.split(y_pred, num_or_size_splits=3, axis=-1)
    
    # Compute PSNR for the luminance (Y) channels
    psnr_y = tf.image.psnr(y_true_y, y_pred_y, max_val=1.0)
    
    # Average PSNR across all channels
    return tf.reduce_mean(psnr_y)


# Define the Attention block function
def Attention(x, g, out):
    theta_x = Conv2D(out, (1,1), strides=(2,2), padding="same")(x)
    phi_g = Conv2D(out, (1,1), padding="same")(g)
    add = Add()([theta_x, phi_g])
    lrelu = LeakyReLU(alpha=0.3)(add)
    s = Conv2DTranspose(out, (1, 1), strides=(2,2), activation='sigmoid', padding="same")(lrelu)
    r = Multiply()([x, s])
    return r
# Define the function to split YUV channels within TensorFlow
def split_yuv_channels(input_tensor):
    # Convert input tensor to YUV color space
    yuv_tensor = tf.image.rgb_to_yuv(input_tensor)
    
    # Split YUV channels
    Y, U, V = tf.split(yuv_tensor, num_or_size_splits=3, axis=-1)
    
    return Y, U, V
# Define the Lightweight U-net model function
def Lightweight_U_net(input1, input2=None):
    if input2 is None: 
        Combined_Input = input1
    else:
        # Split YUV channels for input1
        Y1, U1, V1 = split_yuv_channels(input1)
        # Split YUV channels for input2
        Y2, U2, V2 = split_yuv_channels(input2)

        # Concatenate YUV channels
        Y_concatenated = concatenate([Y1, Y2], axis=-1)
        U_concatenated = concatenate([U1, U2], axis=-1)
        V_concatenated = concatenate([V1, V2], axis=-1)

        # Merge the concatenated channels
        Combined_Input = concatenate([Y_concatenated, U_concatenated, V_concatenated], axis=-1)
     

    act = None
    start_filter = 32
    alpha = 0.3

    conv1 = Conv2D(start_filter * 1, (3, 3), activation=act, padding="same")(Combined_Input)
    conv1 = Conv2D(start_filter * 1, (3, 3), activation=act, padding="same")(conv1)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(start_filter * 2, (3, 3), activation=act, padding="same")(pool1)
    conv2 = Conv2D(start_filter * 2, (3, 3), activation=act, padding="same")(conv2)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    pool2 = MaxPool2D((2, 2))(conv2)

    convm = Conv2D(start_filter * 4, (3, 3), activation=act, padding="same")(pool2)
    convm = Conv2D(start_filter * 4, (3, 3), activation=act, padding="same")(convm)
    convm = LeakyReLU(alpha=alpha)(convm)

    att2 = Attention(conv2, convm, start_filter*2)
    convm = Conv2DTranspose(start_filter * 2, (3, 3), strides=(2,2), activation=ReLU(), padding="same")(convm)
    concate2 = concatenate([convm, att2])
    upconv2 = Conv2D(start_filter * 2, (3, 3), activation=act, padding="same")(concate2)
    upconv2 = Conv2D(start_filter * 2, (3, 3), activation=act, padding="same")(upconv2)
    upconv2 = LeakyReLU(alpha=alpha)(upconv2)

    att1 = Attention(conv1, upconv2, start_filter*1)
    upconv2 = Conv2DTranspose(start_filter * 1, (3, 3), strides=(2,2), activation=ReLU(), padding="same")(upconv2)
    concate1 = concatenate([upconv2, att1])
    upconv1 = Conv2D(start_filter * 1, (3, 3), activation=act, padding="same")(concate1)
    upconv1 = Conv2D(start_filter * 1, (3, 3), activation=act, padding="same")(upconv1)
    upconv1 = LeakyReLU(alpha=alpha)(upconv1)

    output = Conv2D(3, (1,1), padding="same", activation='ELU')(upconv1)
    
    return output
#Current_dir =os.getcwd()
Current_dir='C:/Users/TUF/Desktop/'
# Load paths
pathsc = glob('C:/Users/TUF/Desktop/Mix_Cover_tr/*.jpg')
pathss = glob('C:/Users/TUF/Desktop/Mix_Secret_tr/*.jpg')
pathsvc = glob('C:/Users/TUF/Desktop/Mix_Cover_tv/*.jpg')
pathsvs = glob('C:/Users/TUF/Desktop/Mix_Secret_tv/*.jpg')
random.shuffle(pathss)
random.shuffle(pathsvs)
random.shuffle(pathsc)
random.shuffle(pathsvc)

# Training Path
Cover_Images_tr = pathsc[:60000]
Secret_Images_tr = pathss[:60000]

# Validation Path
Cover_Images_tv = pathsvc[:15000]
Secret_Images_tv = pathsvs[:15000]

# Create data generators
Images_tr = CustomDataGenerator(Current_dir+"\\Mix_Cover_tr",Current_dir+"\\Mix_Secret_tr",8)
Images_tv = CustomDataGenerator(Current_dir+"\\Mix_Cover_tv",Current_dir+"\\Mix_Secret_tv",8)

# Define model inputs
Input_CI = Input(shape=(256, 256, 3))
Input_SI = Input(shape=(256, 256, 3))

# Build the model
output1 = Lightweight_U_net(Input_CI, Input_SI)

output2 = Lightweight_U_net(output1)
Embedding_Model = Model(inputs=[Input_CI, Input_SI], outputs=output1)
Extraction_Model = Model(inputs=output1, outputs=output2)  
Combined_Model = Model(inputs=[Input_CI, Input_SI], outputs=[output1, output2])

# Compile the model
Combined_Model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), loss='mse', metrics=[psnr])

# Define callbacks
model_checkpoint_callback = ModelCheckpoint(
    filepath="optimized_weightsmix.h5",
    save_weights_only=True,
    monitor='val_psnr',
    mode='max',
    save_best_only=True)

# Display model summary
Combined_Model.summary()
# Train the model
History = Combined_Model.fit(Images_tr, epochs=90, validation_data=Images_tv, callbacks=[model_checkpoint_callback])

# Save models and training history
Embedding_Model.save('Embedding_Modelmix.h5')
Extraction_Model.save('Extraction_Modelemix.h5')
Combined_Model.save('Combined_modelemix.h5')