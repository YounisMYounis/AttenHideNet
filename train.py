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

# Note: We are currently updating and refactoring this codebase.
# For access to the older, unrefactored (legacy) version, please contact:
# younis.younis@uoz.edu.krd

# Save models and training history
Embedding_Model.save('Embedding_Modelmix.h5')
Extraction_Model.save('Extraction_Modelemix.h5')
Combined_Model.save('Combined_modelemix.h5')
