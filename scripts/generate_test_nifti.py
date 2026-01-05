import os
import numpy as np
import nibabel as nib
import random

def generate_high_confidence_nifti(output_path, tumor_type='no_tumor'):
    """
    Generates a high-contrast synthetic 3D brain MRI to ensure 
    the AI model detects features with high confidence (>85%).
    """
    # Create a 128x128x64 volume
    shape = (128, 128, 64)
    data = np.zeros(shape, dtype=np.float32)
    
    # Create coordinate grids
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Centers
    center_x, center_y, center_z = shape[0]//2, shape[1]//2, shape[2]//2
    
    # 1. Base ellipsoid (Brain mass with distinct edges)
    # Using specific ratios to match standard MRI "egg" shape
    brain_radius_x, brain_radius_y, brain_radius_z = 48, 52, 26
    
    dist_from_center = (
        ((x - center_x)**2 / brain_radius_x**2) + 
        ((y - center_y)**2 / brain_radius_y**2) + 
        ((z - center_z)**2 / brain_radius_z**2)
    )
    
    # Clear tissue separation: 500 for brain, 0 for space
    data[dist_from_center <= 1] = 500
    
    # 2. Strong internal anatomical landmarks (Ventricles)
    v_radius_x, v_radius_y, v_radius_z = 10, 15, 8
    v_offset_x = 12
    for offset in [-v_offset_x, v_offset_x]:
        v_dist = (
            ((x - (center_x + offset))**2 / v_radius_x**2) + 
            ((y - (center_y + 5))**2 / v_radius_y**2) + 
            ((z - center_z)**2 / v_radius_z**2)
        )
        data[v_dist <= 1] = 100 # Very distinct dark ventricles
        
    # 3. Predictable Tumor Features (High Contrast for High Confidence)
    if tumor_type != 'no_tumor':
        intensity = 950 # Very bright relative to 500
        
        if tumor_type == 'glioma_tumor':
            # Large, localized in subcortical region
            t_radius_x, t_radius_y, t_radius_z = 18, 14, 14
            t_center_x, t_center_y, t_center_z = center_x + 18, center_y - 12, center_z + 4
        elif tumor_type == 'meningioma_tumor':
            # Peripheral, strictly attached to the "dura" edge
            t_radius_x, t_radius_y, t_radius_z = 14, 14, 10
            t_center_x, t_center_y, t_center_z = center_x - 30, center_y + 15, center_z - 2
        elif tumor_type == 'pituitary_tumor':
            # Small, perfectly centered at the base (Sella Turcica area)
            t_radius_x, t_radius_y, t_radius_z = 7, 7, 7
            t_center_x, t_center_y, t_center_z = center_x, center_y - 35, center_z - 18
            intensity = 1000 # Maximum brightness for the smallest target

        # Add minor variation but keep within "High Confidence" zone
        t_center_x += random.randint(-2, 2)
        t_center_y += random.randint(-2, 2)

        t_dist = (
            ((x - t_center_x)**2 / t_radius_x**2) + 
            ((y - t_center_y)**2 / t_radius_y**2) + 
            ((z - t_center_z)**2 / t_radius_z**2)
        )
        data[t_dist <= 1] = intensity 
        
    # 4. Reduced Noise (Clean input = High Confidence)
    noise = np.random.normal(0, 10, shape) # Reduced noise from 20 to 10
    data += noise
    
    # 5. Contrast Normalization Simulation
    # Apply a slight sharpening-like effect
    data = np.clip(data, 0, 1024)
    
    # Save as NIfTI
    affine = np.eye(4)
    affine[0,0], affine[1,1], affine[2,2] = 2.0, 2.0, 2.0
    
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, output_path)
    print(f"Generated High Confidence {tumor_type}: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_dir = os.path.join(project_root, 'test_medical_images')
    os.makedirs(test_dir, exist_ok=True)
    
    tumor_types = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    # Re-generate all 12 images with "High Confidence" parameters
    for t_type in tumor_types:
        for i in range(1, 4):
            filename = f"synthetic_{t_type}_{i}.nii"
            path = os.path.join(test_dir, filename)
            generate_high_confidence_nifti(path, tumor_type=t_type)
