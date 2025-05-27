import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from vtk_utils.vtk_utils import *
import io_utils
import SimpleITK
import metrics
import dataset
import surface_to_segmentation as ss
import random
import argparse

type_dict = {\
        'ASD': {'include': ['ASD'], 'exclude': ['TGA', 'DORV', 'ToF', 'TGA', 'PuA']},\
        'VSD': {'include': ['VSD'], 'exclude': ['TGA', 'DORV', 'ToF', 'TGA', 'PuA']},\
        'TGA': {'include': ['TGA'], 'exclude': ['DORV', 'ToF', 'PuA']},\
        'DORV': {'include': ['DORV'], 'exclude': ['TGA', 'ToF', 'TGA', 'PuA']}, \
        'ToF': {'include': ['ToF'], 'exclude': ['TGA', 'DORV', 'TGA', 'PuA']}, \
        'PuA': {'include': ['PuA'], 'exclude': ['TGA', 'DORV', 'ToF', 'TGA']} \
        }


def specificity_for_type(syn_fns, gt_fns, chd_info, type_name, mode='dice'):
    random.shuffle(syn_fns)

    out_dir = os.path.dirname(syn_fns[0])
    df = dataset.read_excel(chd_info['diag_fn'], sheet_name=chd_info['diag_sn'])
    gt_fns, _, _ = dataset.parse_data_by_chd_type(gt_fns, df, chd_info['types'], chd_info['exclude_types'], mode=['train'], use_aug=False, pad_num=0, ext='.vtp')
    
    ref = ss.create_ref_template((512, 512, 221))
    ref, _ = exportSitk2VTK(ref)

    gt_seg_dict = {}
    for gt_f in gt_fns:
        print(gt_f)
        gt_mesh = load_vtk_mesh(gt_f)
        #gt_vtk = multiclass_convert_polydata_to_imagedata(gt_mesh, ref, connectivity=False)
        gt_vtk = multiclass_convert_polydata_to_imagedata(gt_mesh, ref)
        gt_py = vtk_to_numpy(gt_vtk.GetPointData().GetScalars())
        gt_seg_dict[gt_f] = gt_py

    max_dice_list = []
    for fn in syn_fns[:30]:
        print(fn)
        syn_mesh = load_vtk_mesh(fn)
        #syn_vtk = multiclass_convert_polydata_to_imagedata(syn_mesh, ref, connectivity=False)
        syn_vtk = multiclass_convert_polydata_to_imagedata(syn_mesh, ref)
        syn_py = vtk_to_numpy(syn_vtk.GetPointData().GetScalars())
        if mode == 'dice':
            dice_list = []
            wh_dice = np.array([])
            for k, gt_py in gt_seg_dict.items():
                dice = metrics.dice_score(syn_py, gt_py)
                dice_list.append(dice)
                print(dice)
                wh_dice = np.append(wh_dice, dice[0])
            max_dice = dice_list[np.argmax(wh_dice)]
            max_dice_list.append(max_dice)
            print(max_dice_list)
        elif mode == 'vol':
            vol_list = []
            wh_vol = np.array([])
            for k, gt_py in gt_seg_dict.items():
                vol = metrics.volume_score(syn_py, gt_py)
                vol_list.append(vol)
                print(vol)
                wh_vol = np.append(wh_vol, vol[0])
            min_vol = vol_list[np.argmin(wh_vol)]
            max_dice_list.append(min_vol)

    if mode == 'dice':
        io_utils.write_scores(os.path.join(out_dir, '{}_specificity_dice_whole.csv'.format(type_name)), max_dice_list)
    elif mode == 'vol':
        io_utils.write_scores(os.path.join(out_dir, '{}_specificity_vol_whole.csv'.format(type_name)), max_dice_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    type_name = args.type
    dir_n = '/scratch/users/fwkong/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/train_2200/shape_gen/mesh/*{}*_mesh_spstd0.5_r*.vtp'.format(type_name)
    dir_n = '/scratch/users/fwkong/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0_Pad0_GradMag0_smplfac20_only{}/train_2000/{}_mesh_spstd0.5_*.vtp'.format(type_name, type_name)
    syn_fns = glob.glob(dir_n)
    print(syn_fns)

    gt_dir = '/scratch/users/fwkong/CHD/imageCHDCleanedOriginal_aligned_all/whole_heart_processed_topology_fixed_aligned/vtk/*.vtp'

    gt_fns = glob.glob(gt_dir)
    
    config = {\
            'diag_fn': '/scratch/users/fwkong/CHD/imageCHDCleanedOriginal_aligned_all/whole_heart_processed_topology_fixed_aligned/imageCHD_dataset_WH_diagnosis_info-apr5updates.xlsx', \
            'diag_sn': 'Sheet1', \
            'types': [type_name], \
            'exclude_types': []\
            }
    specificity_for_type(syn_fns, gt_fns, config, type_name, mode=args.mode)
