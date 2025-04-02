# 🚗💤 Driver Fatigue Detection Using EEG Signals

## 📌 Overview
Driver fatigue is a major cause of road accidents worldwide. This project aims to detect driver fatigue using EEG (Electroencephalography) signals and machine learning models. By analyzing EEG signals and classifying them into alert and fatigued states, our system aims to predict driver drowsiness and trigger an alert.

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation & Setup](#-installation--setup)
- [Results](#-results)
- [Contributors](#-contributors)
- [License](#-license)
- [Contact](#-contact)

## 📊 Dataset
- EEG data collected from **twelve healthy subjects** using a **40-channel Neuroscan amplifier**.
- Data stored in `.cnt` files containing continuous EEG signals, timestamps, and metadata.
- Two states of driving recorded:
  - **Alert (awake)**
  - **Fatigued (drowsy)**
- EEG features extracted from relevant frequency bands (Alpha, Beta, Theta, Delta).

## 🔬 Methodology
### 🔍 Data Preprocessing
- Load and process `.cnt` EEG files.
- Apply bandpass filtering to remove noise.
- Extract meaningful EEG features for classification.

### 🎯 Feature Extraction
- Extract **Power Spectral Density (PSD)**, wavelet coefficients, and other statistical EEG features.
- Perform **dimensionality reduction** using PCA and feature selection.

### 🤖 Model Training & Evaluation
- Machine learning models used:
  - ✅ **Support Vector Machine (SVM)**
  - ✅ **Logistic Regression**
  - ✅ **Random Forest**
  - ✅ **K-Nearest Neighbors (KNN)**
  - ✅ **Artificial Neural Networks (ANNs)**
- **Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Techniques Applied:**
  - Hyperparameter tuning, Cross-validation, Feature importance analysis

### ⚡ Fatigue Detection & Alert System
- Real-time classification of EEG signals.
- Generates an **alert** when fatigue is detected.

## ⚙️ Installation & Setup
### 🔧 Prerequisites
Ensure you have Python 3.x installed. Install required libraries using:
```bash
pip install numpy pandas scipy scikit-learn tensorflow mne matplotlib seaborn
```

### 🚀 Running the Project
```bash
# Clone the repository
git clone https://github.com/yourusername/driver-fatigue-detection.git
cd driver-fatigue-detection

# Run the data preprocessing script
python preprocess.py

# Train the models
python train.py

# Evaluate the models
python evaluate.py

# Run the real-time detection system
python detect.py
```

## 📈 Results
- **Accuracy Achieved:** (To be filled after model evaluation)
- **Best Performing Model:** (To be determined)
- **Confusion Matrix & ROC Curves:** Included in `results/` directory.

## 🤝 Contributors
- [Harish J](https://github.com/harishjanarth)
- [Naveen Bijulal Menon](https://github.com/naveenbijulalmenon)
- [Varma](https://github.com/justvarma)

## 📜 License
This project is licensed under the MIT License.

## 📧 Contact
For queries, open an issue or reach out via email:
- 📩 [Varma](mailto:simplyvarma648@gmail.com)
- 📩 [Naveen](mailto:naveenbijulalmenon@gmail.com)
- 📩 [Harish](mailto:harishjanarth@gmail.com)

## 🙏 Acknowledgments
- **References:** Research papers on EEG-based fatigue detection.

