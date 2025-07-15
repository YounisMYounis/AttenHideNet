# AttenHideNet

**AttenHideNet: A Novel Deep Learning-Based Image Steganography Method Using a Lightweight U-Net with Soft Attention**

This repository provides the official implementation of **AttenHideNet**, a deep learning-based image-to-image steganography framework. It uses a lightweight U-Net with soft attention mechanisms to embed a secret image inside a cover image, and then extract it back with high fidelity.

---

## 🧠 Key Features

- Dual-stage architecture: embedding and extraction
- YUV channel separation for improved imperceptibility
- Soft attention blocks for spatial relevance
- Lightweight U-Net for fast inference
- Evaluated on multiple datasets (ImageNet, CelebA, FLW, COCO, etc.)
- PSNR, SSIM, and MSE metrics included

---

## 📁 Project Structure

AttenHideNet/
├── train.py # Model training script
├── requirements.txt # List of dependencies
├── .gitignore # Ignore unneeded files in repo
├── weights/ # Pretrained models (H5)
│ ├── Combmodelmix_weights.h5
│ ├── Embmodelmix_weights.h5
│ └── Extmodelmix_weights.h5
├── test/ # Inference and evaluation code
│ ├── Embedding.py
│ ├── Extraction.py
│ ├── processor.py
│ ├── metrics.py
│ └── Results/ # Output metrics
│ ├── Results_ImageNet_ts.pkl
│ ├── Results_Coco_ts.pkl
│ ├── Results_CelebA_ts.pkl
│ └── ...

---

## 🚀 Getting Started

### 🔧 Requirements

Install Python dependencies using:

```bash
pip install -r requirements.txt

Tested with:

	-Python 3.9+

	-TensorFlow 2.9+

	-NumPy, OpenCV, scikit-image

Pretrained Weights
All pretrained weights are available under:

weights/
├── Combmodelmix_weights.h5     # Full dual-output model
├── Embmodelmix_weights.h5      # Embedding-only model
└── Extmodelmix_weights.h5      # Extraction-only model
These are loaded automatically by the inference scripts.

🧪 Running the Code
🔹 Embedding (hide secret into cover)

python test/Embedding.py
🔹 Extraction (recover secret from stego)

python test/Extraction.py

📊 Evaluation Metrics
The following metrics are used to evaluate the quality of embedding and extraction:

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

MSE (Mean Squared Error)

Results are saved as .pkl files under test/Results/, one file per dataset:

Results_ImageNet_ts.pkl
Results_Coco_ts.pkl
Results_CelebA_ts.pkl
...
Each file is a Python list containing the per-image metrics.

📁 Datasets
⚠️ Datasets are not included in this repository due to size limitations.

To evaluate the model, prepare the following folder structure:

AttenHideNet/
└── data/
    ├── Cover_ts/        # Test cover images (e.g., 256×256 JPGs)
    └── Secret_ts/       # Test secret images (same dimensions)

📜 Citation

This work is published at *Applied Soft Computing*.
Doi:
https://doi.org/10.1016/j.asoc.2025.113583

URL:
-Full version free at first 50 days :https://kwnsfk27.r.eu-west-1.awstrack.me/L0/https:%2F%2Fauthors.elsevier.com%2Fa%2F1lR955aecSzAqm/1/010201980c2b6e49-22ee8e96-4257-4154-af0d-c6dd40470144-000000/kS7BgjBvRy7MQ5w86Lz66LDf0YI=434

-purchase version :https://www.sciencedirect.com/science/article/pii/S1568494625008944

📬 Contact
Younis M. Younis
PhD Student in Information Security (University of Zakho/College of Sciecne/Computer Science Department)
GitHub: @YounisMYounis
Email:younis.younis@uoz.edu.krd



