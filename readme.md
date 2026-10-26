# QuantumDx: Hybrid Quantum-Classical ML Platform

**Domain:** AI/ML · Quantum Machine Learning · Healthcare  
**Problem Statement:** Early Disease Detection — SIH

## Overview

QuantumDx is a prototype exploring a hybrid quantum-classical machine-learning approach for healthcare screening. It combines classical data preprocessing with machine-learning models and quantum machine-learning concepts to investigate patterns in medical images and structured health data.

The project is intended for educational and research purposes. It is not a substitute for professional medical diagnosis.

## Features

- **Brain MRI Analysis:** Uses a trained model for brain-image classification.
- **Pneumonia Image Analysis:** Includes a trained model intended for pneumonia-image classification.
- **Heart-Disease Prediction:** Uses a structured-data prediction pipeline with preprocessing components.
- **Explainability:** Includes Grad-CAM utilities to help visualise image regions relevant to model predictions.
- **Interactive Interface:** Provides a Streamlit application for interacting with the prototype.

*Feature availability and model performance should be verified by testing the application.*

## Technical Architecture

### Classical Processing
Classical machine-learning and image-processing components prepare input data for model inference. The heart-disease pipeline includes saved preprocessing components.

### Quantum Machine Learning
The broader project explores a hybrid quantum-classical approach using a variational quantum circuit. The exact quantum components used by the currently published application should be confirmed against its source code.

### Explainability
Grad-CAM utilities are included for visualising regions of interest in supported image models.

## Technology Stack

- Python
- Streamlit
- PyTorch
- scikit-learn
- Grad-CAM utilities

Additional dependencies should be confirmed from `requirements.txt`.

## Repository Structure

- `app.py` — Streamlit application
- `gradcam_utils.py` — Grad-CAM utilities
- `requirements.txt` — Python dependencies
- `models/` or model files — Saved model weights and preprocessing artifacts

## Getting Started

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Run the application using Streamlit.

```bash
git clone https://github.com/Jayasri9717/Quantum-Vision.git
cd Quantum-Vision
pip install -r requirements.txt
streamlit run app.py
```

These commands assume that the repository contains all required files and that the dependencies install successfully.

## Evaluation

Model performance should be reported only after testing on a documented evaluation dataset. Future evaluation should include appropriate classification metrics, such as accuracy, precision, recall, F1-score, and sensitivity where applicable.

No numerical performance claim is made here until the evaluation results have been verified.

## Limitations

- This is a research prototype, not a clinically validated medical system.
- Predictions may be incorrect and must not be used independently for medical decisions.
- Model performance depends on the datasets, preprocessing, and evaluation methodology.
- Quantum machine-learning claims should be interpreted according to the components actually implemented in the published code.

## Future Improvements

- Verify and document evaluation results for each supported task.
- Add example inputs and screenshots.
- Document the dataset sources and preprocessing steps.
- Improve testing, reproducibility, and deployment instructions.

## Author

**Jayasri R**  
[GitHub Profile](https://github.com/Jayasri9717)
