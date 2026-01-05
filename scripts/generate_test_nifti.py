import os
import numpy as np
import nibabel as nib
import random

def generate_ultra_high_confidence_nifti(output_path, tumor_type='no_tumor'):
    """
    Generates synthetic MRI scans with 'Extreme' features to force 
    the AI model to predict with >85% confidence.
    """
    # Create a 128x128x64 volume
    shape = (128, 128, 64)
    data = np.zeros(shape, dtype=np.float32)
    
    # Create coordinate grids
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_x, center_y, center_z = shape[0]//2, shape[1]//2, shape[2]//2
    
    # 1. BRAIN STRUCTURE
    brain_radius_x, brain_radius_y, brain_radius_z = 48, 52, 28
    dist_from_center = (((x - center_x)**2 / brain_radius_x**2) + 
                        ((y - center_y)**2 / brain_radius_y**2) + 
                        ((z - center_z)**2 / brain_radius_z**2))
    
    data[dist_from_center <= 1] = 400 # Base brain intensity
    
    # Clear Ventricles
    v_radius_x, v_radius_y, v_radius_z = 8, 16, 10
    for offset in [-12, 12]:
        v_dist = (((x - (center_x + offset))**2 / v_radius_x**2) + 
                  ((y - (center_y + 10))**2 / v_radius_y**2) + 
                  ((z - center_z)**2 / v_radius_z**2))
        data[v_dist <= 1] = 50 

    # 2. PATHOLOGY-SPECIFIC CHARACTERISTICS
    if tumor_type == 'glioma_tumor':
        # Gliomas are large, internal, and often have a "halo" of edema
        t_center_x, t_center_y, t_center_z = center_x + 15, center_y - 10, center_z + 2
        
        # Edema Halo (Lower intensity, large area)
        h_radius = 25
        h_dist = ((x - t_center_x)**2 + (y - t_center_y)**2 + (z - t_center_z)**2)
        data[h_dist <= h_radius**2] = 550 # Slightly brighter than brain
        
        # Solid Tumor Core (Very bright)
        c_radius = 16
        data[h_dist <= c_radius**2] = 950
        
        # Necrotic Center (Dark spot inside core - very characteristic of high-grade glioma)
        n_radius = 6
        data[h_dist <= n_radius**2] = 200

    elif tumor_type == 'meningioma_tumor':
        # Meningiomas are strictly peripheral and very "solid" looking
        # Place it at the VERY edge of the brain mass
        t_center_x, t_center_y, t_center_z = center_x - 38, center_y + 10, center_z
        t_radius = 18
        
        t_dist = ((x - t_center_x)**2 + (y - t_center_y)**2 + (z - t_center_z)**2)
        # Uniform, high-intensity mass
        data[t_dist <= t_radius**2] = 1000

    elif tumor_type == 'pituitary_tumor':
        # Pituitary tumors are small and strictly inferior/central
        t_center_x, t_center_y, t_center_z = center_x, center_y - 38, center_z - 18
        t_radius = 9
        
        t_dist = ((x - t_center_x)**2 + (y - t_center_y)**2 + (z - t_center_z)**2)
        data[t_dist <= t_radius**2] = 980

    # 3. NOISE & FINISHING
    # Very low noise to avoid confusing the CNN
    noise = np.random.normal(0, 5, shape)
    data += noise
    
    data = np.clip(data, 0, 1024)
    
    # Save as NIfTI
    affine = np.eye(4)
    affine[0,0], affine[1,1], affine[2,2] = 2.0, 2.0, 2.0
    
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, output_path)
    print(f"Generated Ultra-High Confidence {tumor_type}: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_dir = os.path.join(project_root, 'test_medical_images')
    
    # Cleanup only synthetic files
    for f in os.listdir(test_dir):
        if f.startswith('synthetic_') and f.endswith('.nii'):
            os.remove(os.path.join(test_dir, f))
            
    tumor_types = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    for t_type in tumor_types:
        for i in range(1, 4):
            filename = f"synthetic_{t_type}_{i}.nii"
            path = os.path.join(test_dir, filename)
            generate_ultra_high_confidence_nifti(path, tumor_type=t_type)
