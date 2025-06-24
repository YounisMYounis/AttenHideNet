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
stego_dir = 'output/Stego'
secret_dir = 'data/Secret_ts'
output_dir = 'output/Recover'
os.makedirs(output_dir, exist_ok=True)

# Load model
Ex_model = load_model('weights/Extraction_Model.h5')
processor = ImageProcessor()

# Load image paths
stego_paths = sorted(glob(os.path.join(stego_dir, "*.jpg")))
secret_paths = sorted(glob(os.path.join(secret_dir, "*.jpg")))

# Storage for metrics
Psnr_Extraction, Mse_Extraction, SS_Extraction = [], [], []

batch_size = 8
for i in range(0, len(stego_paths), batch_size):
    batch_stego = stego_paths[i:i+batch_size]
    batch_secret = secret_paths[i:i+batch_size]

    stegos = processor.load_and_preprocess_images(batch_stego)
    secrets = processor.load_and_preprocess_images(batch_secret)

    extractions = Ex_model.predict(stegos, verbose=0)

    for j in range(len(secrets)):
        extracted_img = processor.denormalize_images(extractions[j])
        cv2.imwrite(f'{output_dir}/Rc_{i + j}.jpg', extracted_img)

        Psnr_Extraction.append(compute_psnr(secrets[j], extractions[j]))
        Mse_Extraction.append(np.mean((secrets[j] - extractions[j])**2))
        SS_Extraction.append(compute_ssim(secrets[j], extractions[j]))

# Append to previous results
with open('Results_ts.pkl', 'rb') as f:
    results = pickle.load(f)

results.extend([Psnr_Extraction, Mse_Extraction, SS_Extraction])

with open('Results_ts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Extraction done and results saved.")
