import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def get_axis_slice(data_3d, axis_name, slice_index):
    """
    Extracts a 2D slice from a 3D NumPy array along the specified axis and index.

    Args:
        data_3d (np.ndarray): The 3D NumPy array.
        axis_name (str): The axis to slice along ('x', 'y', or 'z').
        slice_index (int): The index for the slice.

    Returns:
        tuple: (sliced_data, (xlabel, ylabel))
               sliced_data (np.ndarray): The 2D sliced array.
               (xlabel, ylabel) (tuple): Labels for the axes of the slice.
    
    Raises:
        IndexError: If slice_index is out of bounds for the given axis.
        ValueError: If axis_name is invalid.
    """
    if axis_name == 'x':
        if not 0 <= slice_index < data_3d.shape[0]:
            raise IndexError(f"X-axis slice_index {slice_index} is out of bounds for shape {data_3d.shape[0]}.")
        sliced_data = data_3d[slice_index, :, :]
        axis_labels = ("Y-axis", "Z-axis")
    elif axis_name == 'y': # Kept for completeness, though not used in this version
        if not 0 <= slice_index < data_3d.shape[1]:
            raise IndexError(f"Y-axis slice_index {slice_index} is out of bounds for shape {data_3d.shape[1]}.")
        sliced_data = data_3d[:, slice_index, :]
        axis_labels = ("X-axis", "Z-axis")
    elif axis_name == 'z': # Kept for completeness, though not used in this version
        if not 0 <= slice_index < data_3d.shape[2]:
            raise IndexError(f"Z-axis slice_index {slice_index} is out of bounds for shape {data_3d.shape[2]}.")
        sliced_data = data_3d[:, :, slice_index]
        axis_labels = ("X-axis", "Y-axis")
    else:
        raise ValueError(f"Invalid axis_name '{axis_name}'. Must be 'x', 'y', or 'z'.")
    return sliced_data, axis_labels

if __name__ == '__main__':
    pkl_base_directory = 'data/processed_segs/pytorch/'
    output_directory = 'sliced_images_output/' # Directory to save images
    os.makedirs(output_directory, exist_ok=True) # Create output directory if it doesn't exist

    # --- SPECIFY THE FILES YOU WANT TO PROCESS HERE ---
    specific_files_to_process = [
        'pulse_01823_artery.pkl',
        'pulse_01923_artery.pkl'
        # Add more filenames to this list if needed
    ]
    # ----------------------------------------------------

    if not os.path.isdir(pkl_base_directory):
        print(f"Error: Base directory not found: {pkl_base_directory}")
    else:
        for filename in specific_files_to_process:
            file_path = os.path.join(pkl_base_directory, filename)
            
            if not os.path.isfile(file_path):
                print(f"Warning: File not found at {file_path}. Skipping.")
                continue

            print(f"Processing file: {file_path} for multiple X-slices")
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                original_shape_for_title = data.shape 

                if not isinstance(data, np.ndarray):
                    print(f"Skipping {filename}: Data is not a NumPy array.")
                    continue
                
                if data.ndim == 4:
                    print(f"Info: Detected 4D array in {filename} (shape: {data.shape}). Taking first element to get a 3D volume.")
                    if data.shape[0] > 0:
                        data = data[0] 
                    else:
                        print(f"Skipping {filename}: First dimension of 4D array is 0, cannot extract 3D volume.")
                        continue

                if data.ndim != 3:
                    print(f"Skipping {filename}: NumPy array is not 3-dimensional (found {data.ndim} dims after potential 4D adjustment). Dimensions must be 3.")
                    continue
                if any(s == 0 for s in data.shape):
                    print(f"Skipping {filename}: One or more dimensions are zero {data.shape}. Cannot generate slices.")
                    continue

                x_dim_size = data.shape[0]
                if x_dim_size < 1: # Should be caught by any(s==0) but good to be explicit for X
                    print(f"Skipping {filename}: X-dimension is zero {data.shape}. Cannot generate X-slices.")
                    continue

                # Define X-slice indices: 25%, 50%, 75% of X-dimension depth
                # Ensure at least 1 slice if x_dim_size is very small, avoid duplicate indices if possible
                x_slice_indices = sorted(list(set([
                    max(0, x_dim_size // 4),
                    max(0, x_dim_size // 2),
                    max(0, (x_dim_size * 3) // 4 - (1 if x_dim_size > 0 else 0)) # Adjust to avoid going over if x_dim is small
                ])))
                # Ensure indices are within bounds (0 to x_dim_size - 1)
                x_slice_indices = [min(idx, x_dim_size -1) for idx in x_slice_indices if idx < x_dim_size]
                if not x_slice_indices: # If X dimension was 0 or 1 and list became empty
                     x_slice_indices = [0] if x_dim_size > 0 else []
                
                if not x_slice_indices:
                    print(f"Skipping {filename}: Could not determine valid X-slice indices for shape {data.shape}.")
                    continue

                num_plots = len(x_slice_indices)
                fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
                if num_plots == 1: # If only one slice, axs might not be an array
                    axs = [axs]

                for i, x_idx in enumerate(x_slice_indices):
                    try:
                        current_slice, (xlabel, ylabel) = get_axis_slice(data, 'x', x_idx)
                        im = axs[i].imshow(current_slice, cmap='gray')
                        axs[i].set_title(f"X-Slice at index {x_idx}")
                        axs[i].set_xlabel(xlabel)
                        axs[i].set_ylabel(ylabel)
                        fig.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)
                    except (IndexError, ValueError) as e_slice:
                        print(f"Error creating X-slice at index {x_idx} for {filename}: {e_slice}")
                        axs[i].text(0.5, 0.5, f"Error X-slicing:\n{e_slice}", ha='center', va='center', color='red', wrap=True)
                        axs[i].set_title(f"X-Slice {x_idx} - Error")
                        axs[i].set_xticks([])
                        axs[i].set_yticks([])
                
                fig.suptitle(f"Multiple X-Slices for {os.path.basename(filename)}\n(Original Shape: {original_shape_for_title}, Processed 3D Shape: {data.shape})", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout for suptitle (may need y_top adjustment)
                
                output_filename_base = os.path.splitext(os.path.basename(filename))[0]
                output_image_path = os.path.join(output_directory, f"{output_filename_base}_multiple_x_slices.png")
                plt.savefig(output_image_path)
                print(f"Saved multiple X-slices to: {output_image_path}")
                
                print("Attempting to show plot window...")
                plt.show()
                plt.close(fig)

            except FileNotFoundError:
                print(f"Error: File not found at {file_path}.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}") 