import numpy as np

# The full path to your .npz file
# Ensure this path correctly points to your specific NPZ file on the server.
file_path = "/home/alexandra/test-folder/SDF4CHD-doubleM/data/gdrive_downloads/pulse_00002_artery.npz"

try:
    # Load the .npz file
    print(f"Attempting to load: {file_path}")
    data = np.load(file_path)

    # Print the keys (names of the arrays) in the .npz file
    print("Keys found in the .npz file:", list(data.keys()))

    # To get more details, uncomment the following lines:
    # print("\nDetails for each array:")
    # for key in data.keys():
    #     print(f"  Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}")

    # Close the file
    data.close()
    print("\nSuccessfully loaded and inspected the file.")

except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}")
    print("Please double-check the path in the script and that the file exists on the server.")
except Exception as e:
    print(f"An error occurred: {e}")