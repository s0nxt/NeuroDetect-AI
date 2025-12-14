# NeuroDetect AI - Brain Tumor Analysis System

NeuroDetect AI is an advanced web application designed to assist medical professionals in the early and accurate diagnosis of brain tumors from MRI scans. Leveraging state-of-the-art Deep Learning (VGG16 architecture), the system classifies MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

## 🚀 Features

*   **AI-Powered Analysis**: Utilizes a Convolutional Neural Network (CNN) based on VGG16 to classify brain MRI scans with high accuracy.
*   **Instant Diagnosis**: Provides real-time prediction results with confidence scores for each tumor type.
*   **Comprehensive Reports**: Generates detailed PDF reports including patient details, analysis results, tumor information, and medical insights.
*   **Secure User System**: Includes user registration and login functionality to manage access.
*   **History Tracking**: Stores past analysis records for easy retrieval and review (requires MongoDB).
*   **Smart Validation**: Implements strict image validation to ensure only valid MRI scans (grayscale, specific formats) are processed, rejecting invalid or non-medical images.
*   **Responsive Design**: Modern, user-friendly interface built with Flask and Vanilla CSS.

## 🛠️ Tech Stack

*   **Backend**: Python, Flask
*   **AI/ML**: TensorFlow, Keras, NumPy, Pillow
*   **Database**: MongoDB (Optional, for user management and history)
*   **Frontend**: HTML5, CSS3 (Custom styling)
*   **PDF Generation**: ReportLab

## 📋 Prerequisites

*   Python 3.10+
*   MongoDB (Optional, but recommended for full functionality)

## 🔧 Installation

1.  **Clone the repository** (or download the source code):
    ```bash
    git clone <repository-url>
    cd BrainTumor1.0
    ```

2.  **Create a virtual environment** (recommended):
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
    *   Install and run MongoDB locally or use a cloud instance.
    *   The app connects to `mongodb://localhost:27017/` by default.
    *   If MongoDB is not available, the app will run in a limited mode (no history saving).

## ▶️ Usage

1.  **Start the application**:
    ```bash
    python run.py
    ```

2.  **Access the web interface**:
    Open your browser and navigate to `http://127.0.0.1:5000`.

3.  **Register/Login**:
    *   Create a new account or login.
    *   If MongoDB is not running, you can use the demo mode (some features might be restricted).

4.  **Perform Analysis**:
    *   Go to the **Dashboard**.
    *   Enter the **Patient Name**.
    *   Upload a **Brain MRI Image** (Supported formats: .jpg, .png, .dcm, .nii).
    *   Click **Analyze Scan**.

5.  **View Results**:
    *   See the predicted tumor type and confidence scores.
    *   Read detailed medical information about the result.
    *   Download the **PDF Report** for documentation.

## 📂 Project Structure

```
BrainTumor1.0/
├── app/                        # Main Application Package
│   ├── __init__.py             # App Factory & Configuration
│   ├── routes.py               # Web Routes & Controllers
│   ├── utils.py                # Helper Functions
│   ├── gradcam.py              # AI Explainability
│   ├── static/                 # Static Assets (CSS, Images)
│   └── templates/              # HTML Templates
├── models/                     # AI Models Directory
│   └── brain_tumor_classifier_v2.keras
├── run.py                      # Entry Point
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for clinical decision-making.
