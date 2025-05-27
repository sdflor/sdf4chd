import os
import sys
import numpy as np
import torch
import pickle
import vtk
from vtk_utils.vtk_utils import *
import argparse
import glob
from tqdm import tqdm

def process_mesh_to_sdf(input_mesh_file, output_sdf_file, grid_size=128):
    """Process a single mesh file to SDF format"""
    # Read the mesh
    reader = vtk.vtkPolyDataReader() if input_mesh_file.endswith('.vtk') else vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_mesh_file)
    reader.Update()
    mesh = reader.GetOutput()
    
    # Convert to SDF
    sdf = mesh_to_sdf(mesh, grid_size)
    
    # Save SDF
    with open(output_sdf_file, 'wb') as f:
        pickle.dump(sdf, f)

def process_directory(input_dir, output_dir, grid_size=128):
    """Process all mesh files in a directory to SDF format"""
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'pytorch'), exist_ok=True)
    
    # Get all mesh files - support both .nii.gz and mesh files
    mesh_files = []
    for ext in ['.vtk', '.vtp', '.nii.gz']:
        files = glob.glob(os.path.join(input_dir, f'pulse_*_artery{ext}'))
        mesh_files.extend(files)
    
    print(f"Found {len(mesh_files)} files to process")
    
    # Process each file
    for input_file in tqdm(mesh_files):
        basename = os.path.splitext(os.path.basename(input_file))[0]
        if basename.endswith('.nii'): # Handle .nii.gz files
            basename = basename[:-4]
        output_file = os.path.join(output_dir, 'pytorch', f'{basename}.pkl')
        
        if not os.path.exists(output_file):
            try:
                if input_file.endswith('.nii.gz'):
                    # Handle NIFTI files - you'll need to implement this based on your needs
                    print(f"NIFTI processing not implemented yet for {input_file}")
                    continue
                else:
                    process_mesh_to_sdf(input_file, output_file, grid_size)
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Create SDF dataset from mesh or NIFTI files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for SDF files')
    parser.add_argument('--grid_size', type=int, default=128, help='Grid size for SDF computation')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.grid_size)

if __name__ == '__main__':
    main() 