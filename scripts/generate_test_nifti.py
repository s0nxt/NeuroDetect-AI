import os
import numpy as np
import nibabel as nib

def generate_synthetic_nifti(output_path, has_tumor=False):
    """
    Generates a synthetic 3D brain MRI in NIfTI format.
    Uses (X, Y, Z) convention.
    """
    # Create a 128x128x64 volume
    shape = (128, 128, 64)
    data = np.zeros(shape, dtype=np.float32)
    
    # Create coordinate grids
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Centers
    center_x, center_y, center_z = shape[0]//2, shape[1]//2, shape[2]//2
    
    # 1. Base ellipsoid (The "Brain" skull/mass)
    brain_radius_x, brain_radius_y, brain_radius_z = 50, 55, 28
    
    dist_from_center = (
        ((x - center_x)**2 / brain_radius_x**2) + 
        ((y - center_y)**2 / brain_radius_y**2) + 
        ((z - center_z)**2 / brain_radius_z**2)
    )
    
    # Brain tissue intensity (mid-gray level)
    data[dist_from_center <= 1] = 600
    
    # 2. Ventricles (Internal structures - dark areas)
    v_radius_x, v_radius_y, v_radius_z = 12, 18, 10
    v_offset_x = 10
    
    for offset in [-v_offset_x, v_offset_x]:
        v_dist = (
            ((x - (center_x + offset))**2 / v_radius_x**2) + 
            ((y - (center_y + 8))**2 / v_radius_y**2) + 
            ((z - center_z)**2 / v_radius_z**2)
        )
        data[v_dist <= 1] = 150 # Darker cerebrospinal fluid
        
    # 3. Tumor (If requested, add a localized pathology)
    if has_tumor:
        # Place tumor in the left frontal area approximately
        t_radius = 14
        t_center_x, t_center_y, t_center_z = center_x + 15, center_y - 20, center_z + 4
        
        t_dist = (
            (x - t_center_x)**2 + 
            (y - t_center_y)**2 + 
            (z - t_center_z)**2
        )
        # Tumors often appear hyper-intense in T2
        data[t_dist <= t_radius**2] = 900 
        
    # 4. Add Realism (Noise and Intensity gradient)
    # Add Gaussian noise for medical imaging "grain"
    noise = np.random.normal(0, 20, shape)
    data += noise
    
    # Simple intensity gradient to simulate coil sensitivity bias
    bias = np.linspace(0.9, 1.1, shape[0])[:, None, None]
    data *= bias
    
    # Final cleanup
    data = np.clip(data, 0, 1024) # Typical 10-bit intensity range
    
    # Save as NIfTI
    affine = np.eye(4)
    # Most NIfTI viewers expect 1mm or 2mm spacing
    affine[0,0] = 2.0 
    affine[1,1] = 2.0
    affine[2,2] = 2.0
    
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, output_path)
    print(f"Generated synthetic NIfTI: {output_path}")

if __name__ == "__main__":
    # Ensure test directory exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_dir = os.path.join(project_root, 'test_medical_images')
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate Normal and Abnormal samples
    generate_synthetic_nifti(os.path.join(test_dir, 'synthetic_normal_brain.nii'), has_tumor=False)
    generate_synthetic_nifti(os.path.join(test_dir, 'synthetic_tumor_brain.nii'), has_tumor=True)
