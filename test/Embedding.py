import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from glob import glob
from tensorflow.keras.models import load_model
from utils.processor import ImageProcessor
from utils.metrics import compute_psnr, compute_ssim

# Paths
cover_dir = 'data/Cover_ts'
secret_dir = 'data/Secret_ts'
output_dir = 'output/Stego'
os.makedirs(output_dir, exist_ok=True)

# Load model
Em_model = load_model('weights/Embedding_Model.h5')
processor = ImageProcessor()

# Load image paths
cover_paths = sorted(glob(os.path.join(cover_dir, "*.jpg")))
secret_paths = sorted(glob(os.path.join(secret_dir, "*.jpg")))

# Storage for metrics
Psnr_Embedding, Mse_Embedding, SS_Embedding = [], [], []

batch_size = 8
for i in range(0, len(cover_paths), batch_size):
    batch_cover = cover_paths[i:i+batch_size]
    batch_secret = secret_paths[i:i+batch_size]

    covers = processor.load_and_preprocess_images(batch_cover)
    secrets = processor.load_and_preprocess_images(batch_secret)

    embeddings = Em_model.predict([covers, secrets], verbose=0)

    for j in range(len(covers)):
        stego_img = processor.denormalize_images(embeddings[j])
        cv2.imwrite(f'{output_dir}/St_{i + j}.bmp', stego_img)

        Psnr_Embedding.append(compute_psnr(covers[j], embeddings[j]))
        Mse_Embedding.append(np.mean((covers[j] - embeddings[j])**2))
        SS_Embedding.append(compute_ssim(covers[j], embeddings[j]))

# Save results
with open('Results_ts.pkl', 'wb') as f:
    pickle.dump([Psnr_Embedding, Mse_Embedding, SS_Embedding], f)

print("Embedding done and results saved.")
