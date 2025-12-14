
# NeuroDetect AI - Intelligent Multi-Disease Medical Imaging System

## 1. Project Overview
**NeuroDetect AI** is an advanced web-based medical diagnostic platform that harnesses the power of Artificial Intelligence (AI) and Deep Learning to assist healthcare professionals in diagnosing serious conditions from medical images.

The system specifically targets three critical areas:
*   **Brain Tumors:** Classifying MRI scans into Glioma, Meningioma, Pituitary tumor, or No tumor.
*   **Diabetic Retinopathy (DR):** Detecting stages of retinal damage from eye fundus images (No DR, Mild, Moderate, Severe, Proliferative).
*   **Lung Cancer:** Identifying lung cancer types (Adenocarcinoma, Squamous cell carcinoma, etc.) from CT scans.

It acts as an **intelligent second opinion**, providing instant analysis, confidence scores, and visual explanations to support clinical decision-making.

---

## 2. Why This Project is Needed (Problem Statement)
*   **Diagnostic Delays:** Manual analysis of MRIs and CT scans is time-consuming. In critical cases like brain tumors or cancer, every hour counts.
*   **Expert Shortage:** Remote and rural areas often lack specialized radiologists or ophthalmologists.
*   **Human Error:** Fatigue or subtle patterns in complex images can lead to missed diagnoses.
*   **Consistency:** Different experts may interpret the same image differently. AI provides a standardized baseline.

**Impact:** By automating the initial screening, NeuroDetect AI allows doctors to focus on complex cases, reduces waiting times for patients, and potentially catches diseases earlier when they are treatable.

---

## 3. Unique Features (USP)
1.  **Multi-Modal Architecture:** Unlike most systems that focus on just one disease, this platform integrates **three distinct diagnostic models** (Brain, Eye, Lung) into a single, unified interface.
2.  **Explainable AI (XAI) with Grad-CAM:** It doesn't just give a diagnosis; it visualizing **Heatmaps**. These heatmaps highlight the exact regions in the MRI/CT scan that influenced the AI’s decision (e.g., glowing red around a tumor). This "Black Box" transparency builds trust with doctors.
3.  **Advanced Medical Format Support:** It supports raw medical imaging formats directly:
    *   **DICOM (.dcm):** The standard for medical imaging.
    *   **NIfTI (.nii):** Common in neuroimaging research.
    *   Standard image formats (.jpg, .png).
    *   *Includes a 3D Slicing Viewer for NIfTI brain scans.*
4.  **Clinical Workflow Integration:**
    *   **PDF Reports:** Automatically generates professional medical reports with patient details and scan analysis.
    *   **Email Integration:** direct emailing of reports to patients or specialists.
    *   **Patient History:** Tracks analysis history over time in a secure database.
    *   **Feedback Loop:** Allows doctors to correct the AI, collecting data for future model improvements.

---

## 4. Technical Architecture (How it Works)

The project follows a modern **Model-View-Controller (MVC)** influenced architecture:

### A. The "Flow"
1.  **Input:** User (Doctor/Patient) logs in and uploads a medical image via the **Dashboard**.
2.  **Preprocessing (The "Cleaner"):** 
    *   Backend checks file type.
    *   If **DICOM/NIfTI**: It extracts pixel data and converts it to a standardized image format.
    *   **Image Preprocessing:** The image is resized to **224x224 pixels**, normalized (pixel values scaled), and formatted to match the AI model's input requirements.
3.  **AI Inference (The "Brain"):**
    *   The system selects the appropriate model based on user input (Brain/Eye/Lung).
    *   The processed image is passed through a **Deep Convolutional Neural Network (EfficientNetB0)**.
    *   The model outputs a probability array (e.g., `[0.02, 0.95, 0.03, 0.00]`).
4.  **Post-Processing:**
    *   The system identifies the class with the highest probability.
    *   **Confidence Score:** Calculated to inform the user how certain the AI is.
    *   **Grad-CAM:** The system performs a backward pass through the neural network to generate a heatmap indicating valid regions of interest.
5.  **Storage:** Results, file paths, and patient data are stored in **MongoDB**.
6.  **Presentation:** The diagnosis, confidence score, and interactive heatmap are rendered on the results page using **Jinja2 templates** and **Bootstrap**.

### B. Technology Stack
*   **Deep Learning (The Core):**
    *   **TensorFlow & Keras:** For building and running the neural networks.
    *   **EfficientNetB0:** A state-of-the-art Transfer Learning architecture known for high accuracy with fewer parameters than older models like VGG or ResNet. Used for all three disease modules.
    *   **OpenCV & NumPy:** For high-performance image manipulation and matrix operations.
*   **Backend (The Server):**
    *   **Python:** The primary programming language.
    *   **Flask:** A lightweight, robust web framework handling routing, API requests, and application logic.
*   **Database:**
    *   **MongoDB:** A NoSQL database chosen for its flexibility in handling unstructured data and varying patient record formats.
*   **Frontend (The Interface):**
    *   **HTML5, CSS3, Bootstrap 5:** For a responsive, mobile-friendly medical interface.
    *   **JavaScript:** For dynamic interactions (drag-and-drop uploads, UI alerts).
*   **Utilities:**
    *   **pydicom / nibabel:** Libraries specialized for reading medical file formats.
    *   **ReportLab:** For programmatic PDF generation.

---

## 5. Summary for Interview
"NeuroDetect AI is a comprehensive solution effectively bridging the gap between advanced Deep Learning and practical healthcare. By leveraging EfficientNetB0 architectures and solving the 'Black Box' problem with Grad-CAM explainability, it offers a distinct advantage over standard classification tools. It's designed not just as a code project, but as a viable prototype for modern Consultation/Decision Support Systems (CDSS)."
