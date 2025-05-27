import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'vtk_utils'))
from datasets.create_sdfdataset import create_from_segmentation
import glob
from tqdm import tqdm
import pickle
import numpy as np

def regenerate_pickles():
    # Input and output directories
    seg_dir = 'data/segmentations'
    out_dir = 'data/processed_segs'
    
    # Create output directories if they don't exist
    for folder in ['pytorch', 'vtk', 'vtk_ls']:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    
    # Get all NIFTI files
    seg_files = sorted(glob.glob(os.path.join(seg_dir, '*.nii.gz')))
    print(f"Found {len(seg_files)} NIFTI files to process")
    
    # Process each file
    for seg_file in tqdm(seg_files):
        try:
            name = os.path.basename(seg_file).split('.nii.gz')[0]
            output_pkl = os.path.join(out_dir, 'pytorch', f'{name}.pkl')
            output_vtk = os.path.join(out_dir, 'vtk', f'{name}.vtp')
            output_vtk_ls = os.path.join(out_dir, 'vtk_ls', f'{name}.vtp')
            
            # Skip if file already exists and is valid
            if os.path.exists(output_pkl):
                try:
                    with open(output_pkl, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, np.ndarray):
                        print(f"Skipping {name} - valid file already exists")
                        continue
                except:
                    pass  # File is corrupted, regenerate it
            
            print(f"Processing {name}")
            mesh, mesh_ls, sdf, sdf_py = create_from_segmentation(
                seg_file,
                hs_size=(512, 512, 512),
                ds_size=(128, 128, 128),
                ref_fn=None,
                r_ids=[1]  # Process only artery
            )
            
            if mesh is None:
                print(f"Warning: Could not create mesh for {name}")
                continue
                
            # Save the files
            with open(output_pkl, 'wb') as f:
                pickle.dump(sdf_py.astype(np.float32), f)
                
            print(f"Successfully processed {name}")
            
        except Exception as e:
            print(f"Error processing {seg_file}: {str(e)}")
            continue

if __name__ == '__main__':
    regenerate_pickles() 