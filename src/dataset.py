import os
import torch
from torch.utils.data import Dataset
import io_utils
import torch.nn.functional as F
import pickle
import numpy as np
import glob
import pandas as pd
import h5py
import random
import SimpleITK as sitk
import re

def read_excel(fn, sheet_name="Sheet1"):
    df = pd.read_excel(fn, sheet_name=sheet_name, header=0, index_col=1, engine='openpyxl')
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df = df[df.index.notnull()]
    df = df.fillna(0)
    return df

def pad_types(arr, patient_ids, type_ids, type_names, pad_num=20):
    # find normal
    normal_ids = np.where(np.sum(arr[:, type_ids], axis=-1)==0)[0]
    all_add_ids = np.array([]).astype(int)
    if len(normal_ids) < pad_num:
        print("Padding normal: ", pad_num-len(normal_ids))
        add_ids =  np.random.choice(patient_ids[normal_ids], pad_num-len(normal_ids), replace=True)
        all_add_ids = np.concatenate((all_add_ids, add_ids))
    # Find other types
    for i, t in enumerate(type_ids):
        ids = np.where(arr[:, t]==1)[0]
        if len(ids) < pad_num:
            print("Padding {}: ".format(type_names[i]), pad_num-len(ids))
            add_ids =  np.random.choice(patient_ids[ids], pad_num-len(ids), replace=True)
            all_add_ids = np.concatenate((all_add_ids, add_ids))
    return np.concatenate((patient_ids, all_add_ids))


def parse_data_by_chd_type(fns, df, type_names, exclude_type_names, mode=['train'], use_aug=True, pad_num=0, ext='.pkl'):
    arr = df.to_numpy()
    patient_ids = df.index.tolist()
    all_types = df.columns.tolist()
    # Append training, validation or testing mode
    # initialize as false
    mask = arr[:, 0] < -1
    for m in mode:
        mode_id = all_types.index(m)
        mask2 = arr[:, mode_id] > 0
        mask = np.logical_or(mask, mask2)
    ids_to_keep = np.array(patient_ids)[mask]
    
    type_ids = [all_types.index(t) for t in type_names]
    # if a patient has a diagnosis outside of type_names, remove
    exclude_type_ids = [all_types.index(t) if t != 'Normal' else -1 for t in exclude_type_names]
    for i in exclude_type_ids:
        # Handle normal here. 
        if i == -1 :
            mask[np.sum(arr[:, type_ids], axis=-1)==0] = False
        else:
            mask[arr[:, i] > 0.] = False
    ids_to_keep = np.array(patient_ids)[mask]
    if pad_num>0:
        masked_arr = arr[mask, :]
        ids_to_keep = pad_types(masked_arr, ids_to_keep, type_ids, type_names, pad_num=pad_num)
    fns_to_keep = []
    type_data = []
    # need to use the same idx for padded points, store in dict
    idx_dict = {}
    for fn in fns:
        # Accept both old and new file patterns
        basename = os.path.basename(fn)
        original_copy = bool(re.match("ct_[a-z]+_\d+"+ext,basename)) or \
                       bool(re.match("ct_\d+_image"+ext, basename)) or \
                       bool(re.match(".*_mesh_.*"+ext, basename)) or \
                       bool(re.match(".*_vsdVar.*.pkl", basename)) or \
                       bool(re.match("pulse_\d+_artery"+ext, basename))  # Added new pattern
        aug_copy = bool(re.match("ct_[a-z]+_\d+_[0-6]"+ext,basename)) or \
                  bool(re.match("ct_\d+_image_[0-6]"+ext, basename)) or \
                  bool(re.match("pulse_\d+_artery_[0-6]"+ext, basename))  # Added new pattern
        
        if original_copy or (use_aug and aug_copy):
            # If no diagnosis info is provided, accept all files
            if not df.empty and patient_ids:
                for p_id in ids_to_keep:
                    if type(p_id)==float or type(p_id)==np.float64:
                        p_id = int(p_id)
                    if str(p_id) in basename:
                        fns_to_keep.append(fn)
                        type_data.append(arr[patient_ids.index(p_id), type_ids])
                        if not fn in idx_dict:
                            idx_dict[fn] = len(idx_dict)
            else:
                # If no diagnosis info, accept all files and assign default type
                fns_to_keep.append(fn)
                type_data.append(np.zeros(len(type_names)))  # Default to all zeros for type data
                if not fn in idx_dict:
                    idx_dict[fn] = len(idx_dict)
                    
    return fns_to_keep, np.array(type_data), idx_dict

def sample_points_from_sdf(tmplt, n_pt, factor=5, chunk_coord=None, total_size=None, interior=False,binary=True):
    # print("\n=== Starting sample_points_from_sdf ===")
    # Input validation
    if tmplt is None:
        raise ValueError("Input template is None")
    
    if not isinstance(tmplt, np.ndarray):
        raise ValueError(f"Template must be a numpy array, got {type(tmplt)}")
        
    if len(tmplt.shape) != 4:
        raise ValueError(f"Template must be 4D array (C,H,W,D), got shape {tmplt.shape}")
        
    # print(f"Processing template with shape {tmplt.shape}")
    
    _, m, l, n = tmplt.shape
    print(f"Dimensions: m={m}, l={l}, n={n}")
    
    # Additive probability over all classes. Higher prob if close to the surfaces of more classes
    prob_total = np.zeros((m, l, n))
    # print("Computing probability distribution...")
    for i in range(tmplt.shape[0]):
        prob = np.where(tmplt[i]<0., tmplt[i]/np.max(tmplt[i])*np.min(tmplt[i])*-1., tmplt[i])
        prob = (np.max(np.abs(prob)) - np.abs(prob)) # zero has the highest probability
        #import sys
        #print('prob:', prob)
        #sys.exit()
        if np.mean(prob) == 0.:
            print(f"Warning: Mean probability is 0 for sdf_file {i}")
            prob = np.zeros_like(prob)
        else:
            prob = np.exp(prob/np.mean(prob)*factor)
            prob /= np.sum(prob)
        prob_total += prob
    

    # print("Finding points based on criteria...")
    # half of the points are on the boundary, the rest sampled from the prob distribution
    if interior:
        include = tmplt<0.
        # print("Using interior points")
    else:
        include = np.abs(tmplt)<1e-3
        # print("Using boundary points")
    x, y, z = np.where(np.any(include, axis=0)) # x,y,z = HWD координаты точек в 3хм простр-ве 
    
    if len(x) == 0:
        raise ValueError("No points found satisfying the inclusion criteria")
    
    # print(f"Found {len(x)} candidate points")
        
    prob = np.sum(include, axis=0).astype(np.float32)
    prob = prob[np.any(include, axis=0)]
    prob /= np.sum(prob)
    #####ТУТ ОСТАНОВИЛИСЬ !!
    print("Sampling points...", x.shape)
    if interior:
        select = np.random.choice(np.arange(len(x), dtype=np.int64), n_pt, p=prob.flatten(), replace=True)
    else:
        select = np.random.choice(np.arange(len(x), dtype=np.int64), (n_pt * 3) // 4, p=prob.flatten(), replace=True)
    x, y, z = x[select], y[select], z[select] 
    prob_out = prob.flatten()[select]
    print('interiror:', interior)
  
    
    if not interior:
        # print("Sampling additional points from probability distribution...")
        # select points based on sampling probability
        x_c, y_c, z_c = np.where(prob_total>np.percentile(prob_total, 0.97))
        if len(x_c) == 0:
            print("Warning: No points found above 97th percentile, using all points")
            x_c, y_c, z_c = np.where(prob_total > np.min(prob_total))
        prob_selected = prob_total[x_c, y_c, z_c]
        prob_selected /= np.sum(prob_selected)
        remaining_points = n_pt - (n_pt * 3) // 4
        if len(x_c) < remaining_points:
            print(f"Warning: Not enough points for selection ({len(x_c)} < {remaining_points}), using replacement")
            replace = True
        else:
            replace = False
        select = np.random.choice(np.arange(len(x_c), dtype=np.int64), remaining_points, p=prob_selected, replace=replace)
        x = np.concatenate([x, x_c[select]])
        y = np.concatenate([y, y_c[select]])
        z = np.concatenate([z, z_c[select]])
    
    # print("Converting to tensors...")
    x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    x += np.random.normal(0., 1., n_pt).astype(np.float32) #TODO: добавление случайного шума - поэкспериментировать с распределнием шума и убрать шум 
    y += np.random.normal(0., 1., n_pt).astype(np.float32)
    z += np.random.normal(0., 1., n_pt).astype(np.float32)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    z = torch.from_numpy(z)
    
    # print("Normalizing coordinates...")
    # normalize
    x_nrm = 2.*(x.float() / float(tmplt.shape[1]) - 0.5)
    y_nrm = 2.*(y.float() / float(tmplt.shape[2]) - 0.5)
    z_nrm = 2.*(z.float() / float(tmplt.shape[3]) - 0.5)
    
    # print("Creating points tensor...")
    points = torch.stack([z_nrm, y_nrm, x_nrm], dim=-1)
    print('points:', points.shape)
    if points is None:
        raise ValueError("Failed to create points tensor")
    # print(f"Points tensor shape: {points.shape}")
        
    points_gs = points.unsqueeze(0).unsqueeze(0).unsqueeze(0) #(1, 1, 1, N, 3)
    # print(f"Points_gs tensor shape: {points_gs.shape}")
    
    # print("Computing point values...")
    if binary:
        img_binary_py = (tmplt<0.000005).astype(np.float32) # C H W D
        img_binary = torch.from_numpy(img_binary_py)
        point_values_binary = F.grid_sample(img_binary.unsqueeze(0), points_gs, padding_mode='border', align_corners=True)  # (C, 1, 1, N)
        point_values_binary = point_values_binary.squeeze(2).squeeze(2).squeeze(0)
        # print(f"Binary point values shape: {point_values_binary.shape}")

    img_sdv = torch.from_numpy(tmplt.astype(np.float32))
    point_values_sdv = F.grid_sample(img_sdv.unsqueeze(0), points_gs, padding_mode='border', align_corners=True)  # (C, 1, 1, N)
    point_values_sdv = point_values_sdv.squeeze(2).squeeze(2).squeeze(0)
    # print(f"SDV point values shape: {point_values_sdv.shape}")
    
    # print("=== Completed sample_points_from_sdf ===\n")
    if binary:
        return img_binary, points, point_values_binary, point_values_sdv
    else:
        return img_sdv, points, torch.clamp(point_values_sdv, min=-0.001, max=0.001), point_values_sdv

class SDFDataset(Dataset):
    def __init__(self, root_dir, n_pts, type_dir, factor=20, chd_info=None, mode=['train'], use_aug=True, use_cf=False, use_error=True, select_fn_list=None, train=False, pad_num=0, binary=True):
        self.fns = sorted(glob.glob(os.path.join(root_dir, 'pytorch', '*.pkl')))
        self.valid_fns = []  # New list for valid files
        self.root_dir = root_dir
        self.n_pts = n_pts
        self.factor = factor
        self.mode = mode
        self.type_dir = type_dir
        self.use_cf = use_cf
        self.use_error = use_error
        self.train = train
        self.binary = binary
        self.idx_dict = {}

        # Validate files during initialization
        # print("Validating pickle files...")
        for fn in self.fns:
            try:
                with open(fn, 'rb') as f:
                    pickle.load(f)
                self.valid_fns.append(fn)
            except Exception as e:
                print(f"Warning: Skipping corrupted file {fn}: {str(e)}")
        
        self.fns = self.valid_fns  # Use only valid files
        # print(f"Found {len(self.fns)} valid files out of {len(self.valid_fns)} total files")

        # print('select_fn_list', select_fn_list)
        if select_fn_list is not None:
            select_fns = []
            for k in select_fn_list:
                for fn in self.fns:
                    if k in fn:
                        select_fns.append(fn)
            self.fns = select_fns
            # print("Selected files are: ", self.fns)

        # Initialize idx_dict with sequential indices
        for i, fn in enumerate(self.fns):
            self.idx_dict[fn] = i

        self.diag_data = None
        if chd_info is not None and all(key in chd_info for key in ['diag_fn', 'diag_sn', 'types']) and chd_info['diag_fn'] and chd_info['diag_sn']:
            # Only process CHD info if all required fields are present and non-empty
            df = read_excel(chd_info['diag_fn'], sheet_name=chd_info['diag_sn'])
            exclude_types = chd_info.get('exclude_types', [])
            self.fns, self.diag_data, self.idx_dict = parse_data_by_chd_type(self.fns, df, chd_info['types'], exclude_types, mode=mode, use_aug=use_aug, pad_num=pad_num)
            # print(self.fns)
        else:
            # If no CHD info or incomplete info, create dummy diagnosis data
            print('no CHD info or incomplete info, creating dummy diagnosis data')
            self.diag_data = np.zeros((len(self.fns), 1), dtype=np.float32)  # Single type for all samples

    def __len__(self):
        return len(self.fns)

    def get_file_name(self, item):
        return self.fns[item]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
            
        # print(f"\n=== Loading file {item}: {self.fns[item]} ===")
            
        # Add error checking for file existence
        if not os.path.exists(self.fns[item]):
            raise FileNotFoundError(f"File not found: {self.fns[item]}")
            
        try:
            print("Opening pickle file...")
            with open(self.fns[item], 'rb') as f:
                sdf_py_total = pickle.load(f)
                
            # Add shape validation
            if sdf_py_total is None:
                raise ValueError(f"Loaded data is None for file: {self.fns[item]}")
                
            if not isinstance(sdf_py_total, np.ndarray):
                raise ValueError(f"Loaded data is not a numpy array. Got type: {type(sdf_py_total)}")
                
            # print(f"Successfully loaded data with shape: {sdf_py_total.shape}")
            # print(f"Data range: min={np.min(sdf_py_total)}, max={np.max(sdf_py_total)}")
            
            seg_py = np.argmin(sdf_py_total, axis=0)+1
            # print(f"Created seg_py with shape: {seg_py.shape}")
            print('self.binary:', self.binary)
            if self.binary:
                # print("Sampling points with binary=True")
                _, points, point_values, point_values_sdv = sample_points_from_sdf(sdf_py_total, self.n_pts, self.factor, binary=True)#
            
            sample = {'points': points, 'point_values': point_values, 'point_values_sdv': point_values_sdv, \
                    'filename': os.path.basename(self.fns[item]).split('.')[0], 'idx': torch.tensor(self.idx_dict[self.fns[item]])}
            
            if self.diag_data is not None:
                sample['chd_type'] = torch.from_numpy(self.diag_data[item].astype(np.float32))
                print('sample:', sample['chd_type'])
                import sys
                sys.exit()
            
            # print("=== Successfully created sample ===\n")
            return sample
            
        except Exception as e:
            print(f"\nERROR loading file {self.fns[item]}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            raise

