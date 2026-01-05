# NeuroDetect AI - Intelligent Multi-Disease Medical Imaging System

**NeuroDetect AI** is an advanced web-based medical diagnostic platform that harnesses the power of Artificial Intelligence (AI) and Deep Learning to assist healthcare professionals in diagnosing serious conditions from medical images.

The system integrates **three distinct diagnostic models** into a unified interface:

1.  **Brain Tumors**: Classifies MRI scans into Glioma, Meningioma, Pituitary tumor, or No tumor.
2.  **Diabetic Retinopathy (DR)**: Detects stages of retinal damage from eye fundus images (0-4 scale).
3.  **Lung Cancer**: Identifies lung cancer types (Adenocarcinoma, Squamous cell carcinoma, etc.) from CT scans.

It acts as an **intelligent second opinion**, providing instant analysis, confidence scores, and **Grad-CAM visual explanations** (heatmaps) to support clinical decision-making.

---

## 🚀 Features

*   **Multi-Modal Diagnosis**: Single platform for Brain, Eye, and Lung disease analysis.
*   **State-of-the-Art AI**: Powered by **EfficientNetB0** (Transfer Learning) for high-accuracy classification.
*   **Explainable AI (XAI)**: Generates **Grad-CAM Heatmaps** to visualize the specific regions of the image that influenced the AI's decision, solving the "Black Box" problem.
*   **Advanced Format Support**: Native support for **DICOM (.dcm)** and **NIfTI (.nii)** medical formats, alongside standard images (.jpg, .png).
*   **Comprehensive Reporting**: Automatically generates professional **PDF Reports** with patient details, analysis results, and medical insights.
*   **Clinical Workflow**: Includes patient history tracking, secure user authentication, and doctor feedback loops.
*   **Responsive Interface**: Modern, user-friendly UI built with Flask and Bootstrap.

## 🛠️ Tech Stack

*   **Backend**: Python 3.10+, Flask
*   **Deep Learning**: TensorFlow, Keras, EfficientNetB0
*   **Image Processing**: OpenCV, NumPy, Pillow, Pydicom, Nibabel
*   **Database**: MongoDB (Flexible NoSQL storage)
*   **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
*   **Utilities**: ReportLab (PDF Generation)

## 📋 Prerequisites

*   Python 3.10 or higher
*   MongoDB (Optional, but recommended for full functionality like history and user management)

## 🔧 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/s0nxt/NeuroDetect-AI.git
    cd NeuroDetect-AI
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up MongoDB** (Optional):
    *   Install and run MongoDB locally or use a cloud instance string in `.env` (if applicable).
    *   The app defaults to `mongodb://localhost:27017/`.
    *   *Note: Without MongoDB, the functionality will be limited to pure analysis without saving history.*

## ▶️ Usage

1.  **Start the application**:
    ```bash
    python run.py
    ```

2.  **Access the Dashboard**:
    Open your browser and navigate to `http://127.0.0.1:5000`.

3.  **Workflow**:
    *   **Login/Register** to access the system.
    *   Select the **Disease Type** (Brain, Eye, or Lung).
    *   **Upload** the medical scan (Image, DICOM, or NIfTI).
    *   View the **Prediction**, **Confidence Score**, and **Explainability Heatmap**.
    *   **Download Report** as PDF.

## 📂 Project Structure

```
NeuroDetect-AI/
├── app/                        # Main Application Package
│   ├── __init__.py             # App Factory & Configuration
│   ├── routes.py               # Web Routes & Controllers
│   ├── utils.py                # Helper Functions (PDF, File Handling)
│   ├── gradcam.py              # XAI / Heatmap Generation Logic
│   ├── static/                 # CSS, JS, and Temporary Uploads
│   └── templates/              # HTML Templates (Jinja2)
├── models/                     # AI Models
├── scripts/                    # Training Scripts for Deep Learning Models
├── run.py                      # Application Entry Point
├── requirements.txt            # Project Dependencies
└── README.md                   # Documentation
```

## ⚠️ Disclaimer

This tool is a **prototype** intended for **research and educational purposes only**. It comes with no warranty and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider.
