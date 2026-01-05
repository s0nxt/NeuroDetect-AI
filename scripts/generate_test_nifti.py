import os
import numpy as np
import nibabel as nib
import random

def generate_synthetic_nifti(output_path, tumor_type='no_tumor'):
    """
    Generates a synthetic 3D brain MRI in NIfTI format.
    tumor_type options: 'no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor'
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
        
    # 3. Tumor Simulation
    if tumor_type != 'no_tumor':
        t_radius_x, t_radius_y, t_radius_z = 10, 10, 10
        t_center_x, t_center_y, t_center_z = center_x, center_y, center_z
        intensity = 900
        
        if tumor_type == 'glioma_tumor':
            # Gliomas: Large, often in white matter
            t_radius_x, t_radius_y, t_radius_z = 15, 12, 12
            t_center_x, t_center_y, t_center_z = center_x + 15, center_y - 10, center_z + 5
            intensity = 850
        elif tumor_type == 'meningioma_tumor':
            # Meningiomas: Peripheral, near skull
            t_radius_x, t_radius_y, t_radius_z = 12, 12, 8
            t_center_x, t_center_y, t_center_z = center_x - 35, center_y + 10, center_z - 5
            intensity = 950
        elif tumor_type == 'pituitary_tumor':
            # Pituitary: Central, base of brain
            t_radius_x, t_radius_y, t_radius_z = 6, 6, 6
            t_center_x, t_center_y, t_center_z = center_x, center_y - 30, center_z - 15
            intensity = 980
            
        # Add some randomness to position for variety
        t_center_x += random.randint(-5, 5)
        t_center_y += random.randint(-5, 5)
        t_center_z += random.randint(-3, 3)

        t_dist = (
            ((x - t_center_x)**2 / t_radius_x**2) + 
            ((y - t_center_y)**2 / t_radius_y**2) + 
            ((z - t_center_z)**2 / t_radius_z**2)
        )
        # Apply intensity
        data[t_dist <= 1] = intensity 
        
    # 4. Add Realism (Noise and Intensity gradient)
    noise = np.random.normal(0, 20, shape)
    data += noise
    
    # Simple intensity gradient 
    bias = np.linspace(0.95, 1.05, shape[0])[:, None, None]
    data *= bias
    
    # Final cleanup
    data = np.clip(data, 0, 1024) 
    
    # Save as NIfTI
    affine = np.eye(4)
    # 2mm spacing
    affine[0,0] = 2.0 
    affine[1,1] = 2.0
    affine[2,2] = 2.0
    
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, output_path)
    print(f"Generated {tumor_type} synthetic NIfTI: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_dir = os.path.join(project_root, 'test_medical_images')
    os.makedirs(test_dir, exist_ok=True)
    
    tumor_types = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    for t_type in tumor_types:
        for i in range(1, 4):
            filename = f"synthetic_{t_type}_{i}.nii"
            path = os.path.join(test_dir, filename)
            generate_synthetic_nifti(path, tumor_type=t_type)
