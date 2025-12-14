# 🧠 NeuroDetect AI - Codebase Explanation

This document provides a detailed technical overview of the **NeuroDetect AI** application. It explains the project structure, key components, and the execution flow of the system.

## 📂 Project Structure

The project follows a modular Flask application structure:

```
BrainTumor1.0/
├── app/                        # Core Application Package
│   ├── __init__.py             # App Initialization, Database & Model Loading
│   ├── routes.py               # Web Routes (Controllers)
│   ├── utils.py                # Utility Functions (Validation, Image Processing)
│   ├── gradcam.py              # AI Explainability (Heatmap Generation)
│   ├── static/                 # CSS, Images, and User Uploads
│   └── templates/              # HTML Templates (Frontend)
├── models/                     # Directory for AI Models
│   └── brain_tumor_classifier_v2.keras
├── run.py                      # Application Entry Point
├── requirements.txt            # Python Dependencies
└── README.md                   # General Documentation
```

---

## 🔧 Key Components

### 1. Entry Point (`run.py`)
This is the file you execute to start the server. It imports the Flask `app` instance from the `app/` package and runs it.
*   **Command**: `python run.py`

### 2. Initialization (`app/__init__.py`)
This file sets up the application environment:
*   **Flask App**: Initializes the Flask instance.
*   **Database**: Connects to MongoDB (if available) for storing user data and history.
*   **AI Model**: Loads the pre-trained Keras model (`brain_tumor_classifier_v2.keras`) into memory once when the app starts. This prevents reloading the model for every request, ensuring speed.

### 3. Routing & Logic (`app/routes.py`)
This file defines the URL endpoints and handles user requests:
*   `/`: Landing page.
*   `/register` & `/login`: User authentication.
*   `/dashboard`: The main interface for uploading MRI scans.
*   `/download_report`: Generates and serves the PDF report.
*   **Logic Flow**: It receives the image, calls validation functions, triggers the AI prediction, and renders the results.

### 4. Utilities (`app/utils.py`)
Contains helper functions to keep the code clean:
*   **Validation**: `is_valid_mri_file()` checks file types, sizes, and ensures the image is actually a grayscale MRI (not a random photo).
*   **Image Conversion**: `convert_dicom_to_image()` and `convert_nifti_to_image()` handle complex medical file formats (`.dcm`, `.nii`), converting them to standard images the AI can process.
*   **Preprocessing**: `preprocess_image()` resizes and normalizes images to match the model's expected input (128x128 pixels).
*   **PDF Generation**: `generate_pdf_report()` uses ReportLab to create professional medical reports.

### 5. AI Explainability (`app/gradcam.py`)
Implements **Grad-CAM (Gradient-weighted Class Activation Mapping)**.
*   **Purpose**: It looks at the last convolutional layer of the AI model to see *where* the model is looking.
*   **Output**: Generates a heatmap overlay that highlights the tumor region, providing visual proof for the diagnosis.

---

## 🔄 Execution Flow: How an Analysis Works

1.  **Upload**: User uploads a file (JPG, PNG, DICOM, or NIfTI) on the Dashboard.
2.  **Validation**:
    *   `app/utils.py` checks extensions and file headers.
    *   It analyzes pixel intensity (histogram) to ensure it looks like an MRI (mostly black background).
3.  **Preprocessing**:
    *   If it's a medical file (DICOM/NIfTI), it's converted to a JPG.
    *   The image is resized to 128x128 and normalized (pixel values scaled 0-1).
4.  **Prediction**:
    *   The loaded Keras model predicts the class probabilities (e.g., `[0.01, 0.98, 0.01, 0.00]`).
    *   **Softmax Correction**: If the model output doesn't sum to 1 (common in raw logits), a softmax function is applied to get proper percentages.
5.  **Explainability**:
    *   `gradcam.py` runs a backward pass to generate the attention heatmap.
6.  **Storage**:
    *   The result, patient name, and file paths are saved to MongoDB.
7.  **Result**:
    *   The `result.html` template displays the diagnosis, confidence scores, and the Grad-CAM heatmap.

---

## 🧠 AI Model Details

*   **Architecture**: VGG16 (Visual Geometry Group) based CNN.
*   **Input**: 128x128x3 RGB Images.
*   **Classes**:
    1.  Glioma Tumor
    2.  Meningioma Tumor
    3.  No Tumor
    4.  Pituitary Tumor
*   **Training**: Trained on a dataset of thousands of MRI scans to recognize specific texture and shape patterns associated with brain tumors.
