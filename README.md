# 📄 **README.md**

# Bag-of-Visual-Words Scene Classification  
### SIFT + Swin-B Hybrid Descriptor System  
CMPE 537 – Computer Vision (Fall 2025)

This repository implements a **Bag-of-Visual-Words (BoVW)** classification pipeline on the **MIT Indoor Scenes** dataset using both **traditional SIFT descriptors** and **modern Swin-B Transformer features**.  
The project includes feature extraction, quantization via k-means, histogram construction, and classification using SVMs with linear and Chi-Squared kernels.

---

## 📁 Project Structure

``` ngsx
project/
├── data/
│   ├── Images/                 # MIT Indoor Scenes dataset
│   ├── descriptors/            # Extracted SIFT/Swin-B descriptors
│   └── histograms/             # K-means BoVW histograms
├── src/
│   ├── feature_extraction/     # SIFT & Swin-B feature modules
│   ├── models/                 # SVM, classifiers, helpers
│   ├── utils/                  # Preprocessing, scaling
│   └── train.py                # End-to-end training script
├── results/                    # Evaluation outputs, confusion matrices
├── checkpoints/                # Saved models & cluster centers
├── predict.py                  # Single-image prediction script
└── README.md
``` 

---

## 📦 Installation

### 1. Clone the repository  
```bash
git clone https://github.com/username/bovw-sift-swinb.git
cd bovw-sift-swinb
````

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The project uses the **MIT Indoor Scenes** dataset:

* **67 scene classes**
* Train/test file lists: `TrainImages.txt`, `TestImages.txt`
* Each descriptor type stored separately

  * `data/descriptors/sift/`
  * `data/descriptors/swin/`

---

## 🧩 Feature Extraction

### ✔️ **SIFT Descriptors**

* Extracted using OpenCV
* 128-dimensional vectors
* Stored per image
* Used for clustering and histogram generation

### ✔️ **Swin-B Transformer Descriptors**

* Extracted using pretrained PyTorch Swin-B (backbone only)
* Global average pooled embedding
* Features normalized before clustering

---

## 🛠️ BoVW Pipeline

### **1. Feature Quantization (K-Means)**

* Tested cluster sizes: **50, 100, 500**
* Trained only on training descriptors
* Cluster centers saved in `checkpoints/`

### **2. Histogram Construction**

For each image:

1. Assign descriptors to nearest cluster centers
2. Build histogram
3. L1-normalize histogram

---

## 🤖 Classification

### **Linear SVM**

* One-vs-rest strategy
* GridSearchCV hyperparameter tuning

### **Chi-Squared Kernel SVM**

* Precomputed kernel
* Performs well for histogram features

---

## 📈 Evaluation Metrics

* Mean F1-score
* Per-class F1-scores
* Accuracy (balanced & standard)
* Confusion matrices
* Misclassification visualization

  * Shows best/worst-performing classes
  * Example misclassified images (20×20 thumbnails)

---

## 🧪 Example: Predict on Single Image

```bash
python predict.py --image path/to/image.jpg
```

`predict.py` uses:

* Saved model
* Stored scaler
* Vocabulary (cluster centers)
* Class names file

---

## 📝 Report

A full LaTeX report template is included (not in repo) covering:

* Dataset description
* SIFT vs Swin-B comparison
* Preprocessing
* Classifier analysis
* Misclassifications
* Model improvements

---

## 🚀 Future Improvements

* Fisher Vector encoding
* VLAD aggregation
* End-to-end Swin-B fine-tuning
* Spatial pyramid matching (SPM)

---

## 🧑‍💻 Author

**Amin Abu-Hilga**
Boğaziçi University – CMPE 537 (Fall 2025)
