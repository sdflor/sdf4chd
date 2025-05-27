#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch

def get_dimensionality(data):
    """Determines the dimensionality of the loaded data."""
    if isinstance(data, (np.ndarray, torch.Tensor)):
        return f"Shape: {data.shape}"
    elif isinstance(data, list):
        return f"Length: {len(data)}"
    elif isinstance(data, dict):
        return f"Number of keys: {len(data.keys())}"
    else:
        return f"Type: {type(data)}"

def main():
    target_dir = 'data/processed_segs/pytorch'
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' not found.")
        return

    pkl_files = [f for f in os.listdir(target_dir) if f.endswith('.pkl')]
    
    num_pkl_files = len(pkl_files)
    print(f"Found {num_pkl_files} .pkl files in '{target_dir}'.")
    
    if num_pkl_files == 0:
        return

    # print("\nDimensionality of each .pkl file:")
    # #for filename in pkl_files:
    #     filepath = os.path.join(target_dir, filename)
    #     try:
    #         with open(filepath, 'rb') as f:
    #             data = pickle.load(f)
    #         dimensionality = get_dimensionality(data)
    #         print(f"  - {filename}: {dimensionality}")
    #     except Exception as e:
    #         print(f"  - {filename}: Error loading or processing file - {e}")

if __name__ == '__main__':
    main() 