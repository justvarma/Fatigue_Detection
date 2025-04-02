🚗💤 Driver Fatigue Detection Using EEG Signals
📋 Overview
Driver fatigue is a major cause of road accidents worldwide. This project aims to detect driver fatigue using EEG (Electroencephalography) signals and machine learning models. By analyzing EEG data from subjects in different driving states, our system predicts signs of drowsiness and alerts the driver.

📊 Dataset
EEG data collected from twelve healthy subjects using a 40-channel Neuroscan amplifier.
Data stored in .cnt files containing continuous EEG signals, timestamps, and metadata.
Two states of driving process recorded: alert (awake) and fatigued (drowsy).
🔬 Methodology
Data Preprocessing
Load and process .cnt files.
Filter noise and extract relevant EEG features.
Feature Extraction
Extract power spectral density (PSD), wavelet coefficients, and other EEG features.
Model Training & Evaluation
Machine learning models used:
Support Vector Machine (SVM)
Logistic Regression
Random Forest
K-Nearest Neighbors (KNN)
Artificial Neural Networks (ANNs)
Evaluation Metrics:
Accuracy, Precision, Recall, F1-Score, ROC-AUC
Techniques: Hyperparameter tuning, Cross-validation, Feature importance analysis
Fatigue Detection & Alert System
Real-time classification of EEG signals.
Generates an alert when fatigue is detected.
⚙️ Installation & Setup
Prerequisites
Python 3.x
Required Libraries:
pip install numpy pandas scipy scikit-learn tensorflow mne matplotlib seaborn
Running the Project
Clone the repository:
git clone https://github.com/yourusername/driver-fatigue-detection.git
Navigate to the project directory:
cd driver-fatigue-detection
Run the data preprocessing script:
python preprocess.py
Train the models:
python train.py
Evaluate the models:
python evaluate.py
Run the real-time detection system:
python detect.py
📈 Results
Accuracy Achieved: (To be filled after model evaluation)
Best Performing Model: (To be determined)
Confusion Matrix & ROC Curves: Included in results/ directory.
🤝 Contributors
[Harish J ] (@GitHub)
[Naveen Bijulal Menon] (@GitHub)
📜 License
This project is licensed under the MIT License.

🙏 Acknowledgments
References: Research papers on EEG-based fatigue detection.
📧 Contact
For queries, open an issue or reach out at [, simplyvarma648@gmail.com, naveenbijulalmenon@gmail.com, harishjanarth@gmail.com].
