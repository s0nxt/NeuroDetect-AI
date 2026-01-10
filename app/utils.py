import os
import uuid
import re
import io
import numpy as np
import cv2
from PIL import Image as PILImage
import pydicom
import nibabel as nib
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime

# Constants
ALLOWED_MRI_EXTENSIONS = {'.dcm', '.dicom', '.nii', '.nii.gz', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

TUMOR_INFO = {
    'glioma_tumor': {
        'name': 'Glioma Tumor', 
        'description': 'Gliomas are tumors that arise from glial cells in the brain and spinal cord. They are the most common type of primary brain tumor.',
        'stages': {
            'Grade I': 'Low-grade, slow-growing, well-differentiated',
            'Grade II': 'Low-grade but may progress to higher grades',
            'Grade III': 'Anaplastic, malignant, faster growing',
            'Grade IV': 'Glioblastoma, most aggressive form'
        },
        'symptoms': ['Headaches', 'Seizures', 'Memory problems', 'Personality changes', 'Weakness or numbness'],
        'treatment': ['Surgery', 'Radiation therapy', 'Chemotherapy', 'Targeted therapy'],
        'prognosis': 'Varies significantly based on grade, location, and patient factors'
    },
    'meningioma_tumor': {
        'name': 'Meningioma Tumor',
        'description': 'Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord. Most are benign.',
        'stages': {
            'Grade I': 'Benign, slow-growing, well-defined borders',
            'Grade II': 'Atypical, higher recurrence risk',
            'Grade III': 'Anaplastic/malignant, aggressive growth'
        },
        'symptoms': ['Headaches', 'Vision problems', 'Hearing loss', 'Memory loss', 'Seizures'],
        'treatment': ['Observation', 'Surgery', 'Radiation therapy', 'Stereotactic radiosurgery'],
        'prognosis': 'Generally good for Grade I tumors, varies for higher grades'
    },
    'no_tumor': {
        'name': 'No Tumor Detected',
        'description': 'The AI analysis indicates normal brain tissue with no detectable tumor presence.',
        'stages': {
            'Normal': 'Healthy brain tissue with no abnormal growths detected'
        },
        'symptoms': ['No tumor-related symptoms expected'],
        'treatment': ['No treatment required', 'Regular monitoring if symptoms persist'],
        'prognosis': 'Excellent - normal brain tissue'
    },
    'pituitary_tumor': {
        'name': 'Pituitary Tumor',
        'description': 'Pituitary tumors develop in the pituitary gland and can affect hormone production. Most are benign.',
        'stages': {
            'Grade I': 'Small tumor (<10mm), usually benign',
            'Grade II': 'Larger tumor (>10mm), may cause compression',
            'Grade III': 'Extends beyond sella turcica'
        },
        'symptoms': ['Hormonal imbalances', 'Vision problems', 'Headaches', 'Fatigue', 'Mood changes'],
        'treatment': ['Medication', 'Surgery', 'Radiation therapy', 'Hormone replacement'],
        'prognosis': 'Generally good with appropriate treatment'
    }
}

EYE_INFO = {
    '0': {
        'name': 'No DR',
        'description': 'No signs of Diabetic Retinopathy detected. The retina appears healthy.',
        'stages': {'Normal': 'Healthy retina'},
        'symptoms': ['None'],
        'treatment': ['Regular annual eye exams'],
        'prognosis': 'Excellent'
    },
    '1': {
        'name': 'Mild DR',
        'description': 'Mild Non-Proliferative Diabetic Retinopathy. Microaneurysms are present.',
        'stages': {'Stage 1': 'Microaneurysms only'},
        'symptoms': ['Usually none', 'Possible slight blurriness'],
        'treatment': ['Control blood sugar', 'Control blood pressure', 'Monitor closely'],
        'prognosis': 'Good with management'
    },
    '2': {
        'name': 'Moderate DR',
        'description': 'Moderate Non-Proliferative Diabetic Retinopathy. More microaneurysms and other signs like cotton wool spots.',
        'stages': {'Stage 2': 'Multiple microaneurysms, dot/blot hemorrhages'},
        'symptoms': ['Blurry vision', 'Floaters'],
        'treatment': ['Strict blood sugar control', 'Regular monitoring (every 3-6 months)'],
        'prognosis': 'Fair, can progress if uncontrolled'
    },
    '3': {
        'name': 'Severe DR',
        'description': 'Severe Non-Proliferative Diabetic Retinopathy. Many blood vessels are blocked, depriving retina of blood supply.',
        'stages': {'Stage 3': 'Severe hemorrhages, venous beading'},
        'symptoms': ['Vision loss', 'Dark spots', 'Poor night vision'],
        'treatment': ['Laser surgery (photocoagulation)', 'Anti-VEGF injections'],
        'prognosis': 'Risk of vision loss, requires immediate attention'
    },
    '4': {
        'name': 'Proliferative DR',
        'description': 'Proliferative Diabetic Retinopathy. Advanced stage where new, fragile blood vessels grow (neovascularization).',
        'stages': {'Stage 4': 'Neovascularization, vitreous hemorrhage, retinal detachment risk'},
        'symptoms': ['Severe vision loss', 'Blindness', 'Red vision (bleeding)'],
        'treatment': ['Panretinal photocoagulation', 'Vitrectomy surgery'],
        'prognosis': 'Serious threat to vision, requires urgent treatment'
    }
}

LUNG_INFO = {
    'Adenocarcinoma': {
        'name': 'Adenocarcinoma',
        'description': 'Adenocarcinoma is a type of non-small cell lung cancer (NSCLC) that usually develops in the outer part of the lung. It is the most common type of lung cancer.',
        'stages': {
            'Stage I': 'Cancer is in the lung only',
            'Stage II': 'Cancer is in the lung and nearby lymph nodes',
            'Stage III': 'Cancer has spread to lymph nodes in the middle of the chest',
            'Stage IV': 'Cancer has spread to other parts of the body'
        },
        'symptoms': ['Persistent cough', 'Shortness of breath', 'Chest pain', 'Coughing up blood'],
        'treatment': ['Surgery', 'Chemotherapy', 'Targeted therapy', 'Immunotherapy'],
        'prognosis': 'Better if caught early'
    },
    'Large cell carcinoma': {
        'name': 'Large Cell Carcinoma',
        'description': 'Large cell carcinoma is a type of non-small cell lung cancer that can appear in any part of the lung. It tends to grow and spread quickly.',
        'stages': {
            'Stage I-IV': 'Similar staging to other NSCLCs'
        },
        'symptoms': ['Cough', 'Fatigue', 'Chest pain', 'Shortness of breath'],
        'treatment': ['Surgery', 'Radiation', 'Chemotherapy'],
        'prognosis': 'Can be aggressive'
    },
    'Squamous cell carcinoma': {
        'name': 'Squamous Cell Carcinoma',
        'description': 'Squamous cell carcinoma is a type of non-small cell lung cancer that usually starts in the squamous cells that line the airways inside the lungs.',
        'stages': {
            'Stage I-IV': 'Similar staging to other NSCLCs'
        },
        'symptoms': ['Cough', 'Coughing up blood', 'Shortness of breath', 'Wheezing'],
        'treatment': ['Surgery', 'Radiation', 'Chemotherapy', 'Immunotherapy'],
        'prognosis': 'Varies by stage'
    },
    'Normal': {
        'name': 'Normal Lung',
        'description': 'The AI analysis indicates healthy lung tissue with no detectable abnormalities.',
        'stages': {'Normal': 'Healthy lung tissue'},
        'symptoms': ['None'],
        'treatment': ['No treatment required'],
        'prognosis': 'Excellent'
    }
}

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def validate_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'[a-zA-Z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    return True

def validate_username(username):
    return re.match(r'^[a-zA-Z0-9_]{3,20}$', username) is not None

def detect_file_type(file_path):
    try:
        try:
            with PILImage.open(file_path) as img:
                return f'image/{img.format.lower()}'
        except:
            pass
        
        with open(file_path, 'rb') as f:
            header = f.read(132)
            if len(header) >= 132 and header[128:132] == b'DICM':
                return 'application/dicom'
        
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header in [b'\x5c\x01\x00\x00', b'\x00\x00\x01\x5c']:
                return 'application/nifti'
        
        return 'unknown'
    except Exception:
        return 'unknown'

def is_valid_mri_file(file, analysis_type='brain'):
    if not file or not file.filename:
        return False, "No file selected"
    
    filename = file.filename.lower()
    file_ext = None
    for ext in ALLOWED_MRI_EXTENSIONS:
        if filename.endswith(ext.lower()):
            file_ext = ext
            break
    
    if not file_ext:
        return False, "Invalid file format. Please upload MRI images in DICOM (.dcm), NIfTI (.nii), JPEG, or PNG format."
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File size too large. Maximum allowed size is {MAX_FILE_SIZE // (1024*1024)}MB."
    
    if file_size == 0:
        return False, "File is empty."
    
    temp_filename = f"temp_{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join('app', 'static', temp_filename)
    os.makedirs(os.path.join('app', 'static'), exist_ok=True)
    
    try:
        file.save(temp_path)
        file.seek(0)
        
        detected_type = detect_file_type(temp_path)
        
        if file_ext in ['.dcm', '.dicom']:
            try:
                dicom_data = pydicom.dcmread(temp_path, force=True)
                modality = getattr(dicom_data, 'Modality', '').upper()
                body_part = getattr(dicom_data, 'BodyPartExamined', '').upper()
                
                if modality and modality not in ['MR', 'MRI']:
                    return False, "File is not an MRI scan. Please upload MRI images only."
                
                if body_part:
                    brain_keywords = ['BRAIN', 'HEAD', 'CRANIUM', 'CEREBR', 'NEURO']
                    if not any(keyword in body_part for keyword in brain_keywords):
                        return False, "MRI scan does not appear to be of the brain. Please upload brain MRI images only."
            except Exception:
                return False, "Invalid DICOM file format or corrupted file."
        
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            try:
                if detected_type not in ['image/jpeg', 'image/png', 'image/jpg']:
                    return False, "File is not a valid image format."
                
                img = PILImage.open(temp_path)
                if img.mode not in ['L', 'RGB', 'RGBA', 'P']:
                    return False, "Image format not supported for MRI analysis."
                
                width, height = img.size
                if width < 64 or height < 64:
                    return False, "Image resolution too low. Minimum 64x64 pixels required."
                
                if width > 8192 or height > 8192:
                    return False, "Image resolution too high. Maximum 8192x8192 pixels allowed."
                
                # Only check for grayscale if analysis type is brain or lung
                if analysis_type in ['brain', 'lung'] and img.mode != 'L':
                    hsv_img = img.convert('HSV')
                    _, s, _ = hsv_img.split()
                    avg_saturation = np.mean(np.array(s))
                    if avg_saturation > 25:
                        return False, f"Invalid image. {analysis_type.capitalize()} scans should be grayscale. Please upload a valid medical scan."
                
                # Specific validation for Eye Scans (Retinal Fundus)
                if analysis_type == 'eye':
                    # Convert to RGB if not already
                    if img.mode != 'RGB':
                        img_rgb = img.convert('RGB')
                    else:
                        img_rgb = img
                    
                    img_array = np.array(img_rgb)
                    
                    # Create mask for non-black pixels (intensity > 20)
                    # We use grayscale for masking
                    gray_temp = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    mask = gray_temp > 20
                    
                    if np.sum(mask) == 0:
                         return False, "Invalid scan. Image is too dark."

                    # Get mean R, G, B of the foreground
                    r_mean = np.mean(img_array[:,:,0][mask])
                    g_mean = np.mean(img_array[:,:,1][mask])
                    b_mean = np.mean(img_array[:,:,2][mask])
                    
                    # Ratios
                    rg_ratio = r_mean / (g_mean + 1e-5)
                    gb_ratio = g_mean / (b_mean + 1e-5)
                    
                    # Heuristic: Eye scans are reddish/orange. 
                    # R >> G >> B. Skin is R > G > B but ratios are smaller.
                    # Valid Eye: R/G > 1.4, G/B > 1.2
                    is_valid_color = (rg_ratio > 1.4) and (gb_ratio > 1.2)
                    
                    if not is_valid_color:
                         return False, "Invalid scan. Please upload a valid Retinal Fundus image (red/orange dominant)."
                    
                    # Check HSV (Hue)
                    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    h_channel = hsv_img[:,:,0][mask]
                    
                    # Hue 0-40 or 140-179 (Red/Orange)
                    valid_hue_mask = ((h_channel >= 0) & (h_channel <= 40)) | ((h_channel >= 140) & (h_channel <= 179))
                    valid_hue_ratio = np.sum(valid_hue_mask) / len(h_channel)
                    
                    if valid_hue_ratio < 0.5:
                         return False, "Invalid scan. Color profile does not match a Retinal Fundus image."

                if analysis_type == 'brain':
                    # Open image
                    gray_img = img.convert('L')
                    histogram = gray_img.histogram()
                    # Count pixels with value < 15 (very dark/black)
                    black_pixels = sum(histogram[:15])
                    total_pixels = width * height
                    black_ratio = black_pixels / total_pixels
                    
                    # Enforce minimum black background for Brain MRI (lowered to 1% to allow cropped images)
                    if black_ratio < 0.01:
                        return False, "Invalid scan. Brain MRIs typically have a black background. Please check your image."
                

                    # Check for Lung CT characteristics (large internal dark areas - lungs)
                    # Only perform this check if the image has low black background (potential Lung CT)
                    # Valid Brain MRIs usually have > 15% black background.
                    if black_ratio < 0.15:
                        img_array = np.array(gray_img)
                        # Threshold the image to find the "body" (non-black area)
                        _, binary_body = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY)
                        body_mask = binary_body > 0
                        total_body_pixels = np.sum(body_mask)
                        
                        if total_body_pixels > 0:
                            # Count dark pixels INSIDE the body (e.g., lungs are dark)
                            # We look for pixels that are dark (< 40) but are inside the body mask
                            internal_dark_pixels = np.sum((img_array < 40) & body_mask)
                            internal_dark_ratio = internal_dark_pixels / total_body_pixels
                            
                            # If more than 35% of the "body" is dark, it's likely lungs, not a brain
                            if internal_dark_ratio > 0.35:
                                return False, "Invalid scan. This appears to be a Lung CT scan (large air pockets detected). Please upload a Brain MRI."
            
            except Exception:
                return False, "Invalid image file format or corrupted file."
        
        elif file_ext in ['.nii', '.nii.gz']:
            if detected_type != 'application/nifti':
                return False, "File does not appear to be a valid NIfTI format."
    
    except Exception as e:
        return False, f"Error validating file: {str(e)}"
    
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
    
    return True, "Valid medical scan"

def convert_dicom_to_image(dicom_path, output_path):
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array
        
        if len(pixel_array.shape) == 3:
            pixel_array = pixel_array[pixel_array.shape[0] // 2]
        
        p_low = np.percentile(pixel_array, 1)
        p_high = np.percentile(pixel_array, 99)
        pixel_array = np.clip(pixel_array, p_low, p_high)
        
        pixel_array = pixel_array.astype(float)
        if pixel_array.max() > pixel_array.min():
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        
        pixel_array = pixel_array.astype(np.uint8)
        
        img = PILImage.fromarray(pixel_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        return False

def convert_nifti_to_image(nifti_path, output_path):
    try:
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        
        if len(data.shape) == 3:
            middle_slice_idx = data.shape[2] // 2
            slice_data = data[:, :, middle_slice_idx]
            slice_data = np.rot90(slice_data)
        elif len(data.shape) == 4:
            middle_slice_idx = data.shape[2] // 2
            slice_data = data[:, :, middle_slice_idx, 0]
            slice_data = np.rot90(slice_data)
        else:
            return False
            
        p_low = np.percentile(slice_data, 1)
        p_high = np.percentile(slice_data, 99)
        slice_data = np.clip(slice_data, p_low, p_high)
        
        if slice_data.max() > slice_data.min():
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            
        slice_data = slice_data.astype(np.uint8)
        
        img = PILImage.fromarray(slice_data)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error converting NIfTI: {e}")
        return False

def process_nifti_slices(nifti_path, output_dir):
    """
    Extracts all slices from a NIfTI file and saves them as images.
    Returns a list of relative paths to the saved images.
    """
    try:
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        
        # Handle 4D data (time series) by taking the first volume
        if len(data.shape) == 4:
            data = data[:, :, :, 0]
            
        if len(data.shape) != 3:
            return []
            
        num_slices = data.shape[2]
        slice_paths = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit to max 100 slices to prevent overload, skip if needed
        step = max(1, num_slices // 100)
        
        for i in range(0, num_slices, step):
            slice_data = data[:, :, i]
            slice_data = np.rot90(slice_data)
            
            # Normalize
            p_low = np.percentile(slice_data, 1)
            p_high = np.percentile(slice_data, 99)
            slice_data = np.clip(slice_data, p_low, p_high)
            
            if slice_data.max() > slice_data.min():
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
                
            slice_data = slice_data.astype(np.uint8)
            
            img = PILImage.fromarray(slice_data)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            filename = f"slice_{i:03d}.jpg"
            file_path = os.path.join(output_dir, filename)
            img.save(file_path)
            
            # Store relative path for frontend
            # Assuming output_dir is inside static/
            rel_path = os.path.relpath(file_path, start=os.path.join('app', 'static'))
            slice_paths.append(rel_path)
            
        return slice_paths
    except Exception as e:
        print(f"Error processing NIfTI slices: {e}")
        return []

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_prediction_confidence(prediction_array, class_labels):
    confidence_scores = {}
    for i, class_name in enumerate(class_labels):
        confidence_scores[class_name] = float(prediction_array[0][i] * 100)
    return confidence_scores

def generate_pdf_report(patient_name, image_path, prediction, confidence_scores, analysis_date, heatmap_path=None, info_dict=None, analysis_type='brain', predicted_stage=None):
    if info_dict is None:
        info_dict = TUMOR_INFO

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30, alignment=TA_CENTER, textColor=colors.HexColor('#2c3e50'))
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16, spaceAfter=12, spaceBefore=20, textColor=colors.HexColor('#3498db'))
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=11, spaceAfter=12, alignment=TA_JUSTIFY)
    
    # Title
    report_title = "NeuroDetect AI - Medical Analysis Report"
    if analysis_type == 'eye':
        report_title = "NeuroDetect AI - Diabetic Retinopathy Report"
    elif analysis_type == 'lung':
        report_title = "NeuroDetect AI - Lung Cancer Analysis Report"
    elif analysis_type == 'brain':
        report_title = "NeuroDetect AI - Brain Tumor Analysis Report"

    elements.append(Paragraph(report_title, title_style))
    elements.append(Spacer(1, 20))
    
    # Determine Model Name
    model_name = "Brain Tumor Classifier v1.0"
    if analysis_type == 'eye':
        model_name = "Diabetic Retinopathy Detector v1.0"
    elif analysis_type == 'lung':
        model_name = "Lung Cancer Detector v1.0"

    header_data = [
        ['Patient Name:', patient_name],
        ['Analysis Date:', analysis_date.strftime('%Y-%m-%d %H:%M:%S')],
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['AI Model:', model_name]
    ]
    
    header_table = Table(header_data, colWidths=[2*inch, 4*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 30))
    
    scan_title = "MRI Scan Analysis"
    image_label = "Original MRI"
    if analysis_type == 'eye':
        scan_title = "Retinal Scan Analysis"
        image_label = "Original Eye Scan"
    elif analysis_type == 'lung':
        scan_title = "CT Scan Analysis"
        image_label = "Original CT Scan"
        
    elements.append(Paragraph(scan_title, heading_style))
    
    image_data = []
    try:
        img1 = Image(image_path, width=3*inch, height=3*inch)
        img1.hAlign = 'CENTER'
        
        if heatmap_path and os.path.exists(heatmap_path):
            img2 = Image(heatmap_path, width=3*inch, height=3*inch)
            img2.hAlign = 'CENTER'
            image_data = [[img1, img2], [image_label, "AI Attention Map"]]
        else:
            image_data = [[img1], [image_label]]
            
        img_table = Table(image_data, colWidths=[3.5*inch, 3.5*inch] if len(image_data[0]) > 1 else [6*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Oblique'),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.gray),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 20))
    except Exception as e:
        elements.append(Paragraph(f"Error loading images: {str(e)}", normal_style))
        elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("AI Prediction Results", heading_style))
    prediction_text = f"<b>Primary Diagnosis:</b> {info_dict.get(prediction, {}).get('name', prediction)}"
    if predicted_stage:
        prediction_text += f"<br/><b>Predicted Stage/Grade:</b> {predicted_stage}"
    elements.append(Paragraph(prediction_text, normal_style))
    elements.append(Spacer(1, 10))
    
    confidence_data = [['Tumor Type', 'Confidence Score']]
    for tumor_type, confidence in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
        tumor_name = info_dict.get(tumor_type, {}).get('name', tumor_type)
        confidence_data.append([tumor_name, f"{confidence:.2f}%"])
    
    confidence_table = Table(confidence_data, colWidths=[3*inch, 2*inch])
    confidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
    ]))
    elements.append(confidence_table)
    elements.append(Spacer(1, 30))
    
    tumor_info = info_dict.get(prediction, {})
    if tumor_info:
        elements.append(Paragraph("Detailed Analysis", heading_style))
        elements.append(Paragraph(f"<b>Description:</b> {tumor_info.get('description', 'N/A')}", normal_style))
        elements.append(Spacer(1, 10))
        
        if tumor_info.get('stages'):
            elements.append(Paragraph("<b>Possible Stages/Grades:</b>", normal_style))
            for stage, description in tumor_info['stages'].items():
                elements.append(Paragraph(f"• <b>{stage}:</b> {description}", normal_style))
            elements.append(Spacer(1, 10))
        
        if tumor_info.get('symptoms'):
            symptoms_text = "• " + "<br/>• ".join(tumor_info['symptoms'])
            elements.append(Paragraph(f"<b>Common Symptoms:</b><br/>{symptoms_text}", normal_style))
            elements.append(Spacer(1, 10))
            
        if tumor_info.get('treatment'):
            treatment_text = "• " + "<br/>• ".join(tumor_info['treatment'])
            elements.append(Paragraph(f"<b>Treatment Options:</b><br/>{treatment_text}", normal_style))
            
    elements.append(Paragraph("AI Model Information", heading_style))
    model_info = []
    if analysis_type == 'eye':
        model_info = [
            "This analysis was performed using a deep learning model trained on retinal fundus images.",
            "The model classifies Diabetic Retinopathy into 5 stages: No DR, Mild, Moderate, Severe, and Proliferative.",
            "Current model accuracy: Approximately 85-90% on validation data.",
            "The confidence scores represent the model's certainty for each stage."
        ]
    elif analysis_type == 'lung':
        model_info = [
            "This analysis was performed using a deep learning model (VGG16) trained on Chest CT scans.",
            "The model classifies lung conditions into: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal.",
            "Current model accuracy: Approximately 85-90% on validation data.",
            "The confidence scores represent the model's certainty for each classification."
        ]
    else: # Brain
        model_info = [
            "This analysis was performed using a deep learning convolutional neural network trained on thousands of MRI brain scans.",
            "The model has been trained to classify four categories: Glioma, Meningioma, Pituitary tumors, and No tumor.",
            "Current model accuracy: Approximately 85-90% on validation data.",
            "The confidence scores represent the model's certainty for each classification."
        ]
    for info in model_info:
        elements.append(Paragraph(f"• {info}", normal_style))
    
    elements.append(Spacer(1, 30))
    elements.append(Spacer(1, 30))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor('#7f8c8d'))
    elements.append(Paragraph("Generated by NeuroDetect AI ", footer_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

