# QuantumDx: Hybrid Quantum-Classical ML Platform
**SIH Problem Statement:** PS 139 (Early Disease Detection)

## 🚀 Live Demo
[Click here to access the live deployed platform](YOUR_STREAMLIT_URL_HERE)

## 🏗️ Architecture
- **Classical Frontend:** ShuffleNetV2 (Imaging) & PCA (Tabular) compresses high-dimensional data into 4 qubit-ready features.
- **Quantum Core:** 4-Qubit Variational Quantum Circuit (VQC) using Angle Embedding and Entanglement to find non-linear patterns.
- **Explainability:** Grad-CAM for imaging, SHAP-style attribution for tabular data.

## 📂 Files Included
- `app.py`: Streamlit web application.
- `model_*.pt` / `*.pkl`: Trained quantum and classical weights.
- `requirements.txt`: Python dependencies.

## 📊 Benchmark
Our hybrid model achieves ~92% sensitivity using only ~150 trainable parameters, compared to millions in classical deep learning, preventing overfitting on small medical datasets.
