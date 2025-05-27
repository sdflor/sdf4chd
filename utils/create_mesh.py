import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from vtk_utils.vtk_utils import *
from scipy import stats
import SimpleITK as sitk

def extract_mesh_from_nifty(seg):
    mesh_list, py_sdf = [], []
    region_ids = np.array([])
    seg_py = sitk.GetArrayFromImage(seg)
    r_ids = np.unique(seg_py)
    for i in r_ids: # Myo only
        print("R_id: ", i)
        if i >0:
            seg_i = sitk.GetImageFromArray((seg_py==i).astype(np.uint8))
            seg_i.CopyInformation(seg)
            seg_i_vtk = exportSitk2VTK(seg_i, seg_i.GetSpacing())[0]
            mesh_i = decimation(smooth_polydata(vtk_marching_cube(seg_i_vtk, 0, 1), 100, smoothingFactor=0.5), 0.9)
            region_ids = np.append(region_ids, np.ones(mesh_i.GetNumberOfPoints()) * (i-1))
            mesh_list.append(mesh_i)
    mesh = appendPolyData(mesh_list)
    region_ids_arr = numpy_to_vtk(region_ids)
    region_ids_arr.SetName('RegionId')
    mesh.GetPointData().AddArray(region_ids_arr)
    return mesh
    
if __name__ == '__main__':
    dir_n = '/Users/fanweikong/Documents/ImageData/ImageCHD_dataset/All/ct_train_fixedsp_masks_cube_cleaned2_slicer_aligned/fixed_topology/aligned/seg'
    fns = glob.glob(os.path.join(dir_n, '*.nii.gz'))

    for fn in fns:
        out_fn = fn[:-6]+'vtp'
        seg = sitk.ReadImage(fn)
        mesh = extract_mesh_from_nifty(seg)
        write_vtk_polydata(mesh, out_fn)



