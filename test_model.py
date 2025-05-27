import torch
import os
import sys
from pprint import pprint

def inspect_checkpoint(checkpoint_path):
    print("\n" + "="*50)
    print("CHECKPOINT INSPECTION")
    print("="*50)
    
    print(f"\nCheckpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found!")
        sys.exit(1)
    
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\nCheckpoint contents:")
    print("-" * 30)
    
    # Print top-level keys
    print("Top-level keys:")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    # If state dict is present, analyze its structure
    state_dict_key = None
    if 'state_dict' in checkpoint:
        state_dict_key = 'state_dict'
    elif 'model_state_dict' in checkpoint:
        state_dict_key = 'model_state_dict'
        
    if state_dict_key:
        state_dict = checkpoint[state_dict_key]
        print(f"\nModel state dict structure (key: {state_dict_key}):")
        print("-" * 30)
        
        # Group parameters by modules
        modules = {}
        for key in state_dict.keys():
            module_name = key.split('.')[0]
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append(key)
        
        # Print module structure
        for module_name, params in modules.items():
            print(f"\nModule: {module_name}")
            print("Parameters:")
            for param in params:
                shape = tuple(state_dict[param].shape)
                print(f"  - {param}: {shape}")
    
    # Print other contents if present
    for key in checkpoint.keys():
        if key != state_dict_key:
            print(f"\nContents of '{key}':")
            print("-" * 30)
            if isinstance(checkpoint[key], (dict, list, tuple, str, int, float)):
                #pprint(checkpoint[key])
                print(key)
            else:
                print(f"Type: {type(checkpoint[key])}")
                if hasattr(checkpoint[key], 'shape'):
                    print(f"Shape: {checkpoint[key].shape}")

if __name__ == "__main__":
    checkpoint_path = "/home/alexandra/test-folder/SDF4CHD-doubleM/pretrained/sdf4chd_final/net_200.pt"
    inspect_checkpoint(checkpoint_path) 