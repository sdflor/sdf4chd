import os
import glob
import vtk
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from vtk_utils.vtk_utils import *
import SimpleITK as sitk
import argparse
try:
    from mpi4py import MPI
except Exception as e: print(e)

def convert_to_segmentation(fn, out_fn, ref_fn=None, if_vtk=False):
    assert ref_fn is not None, "Now assume there's a ref segmentation"
    if if_vtk:
        seg_vtk = load_vtk_image(ref_fn)
        # need this for synthetic data
        pt_arry = seg_vtk.GetPointData().GetArray(0)
        seg_vtk.GetPointData().SetScalars(pt_arry)
    else:
        seg = sitk.ReadImage(ref_fn)
        seg_vtk, M = exportSitk2VTK(seg)
    mesh = load_vtk_mesh(fn)
    new_seg = multiclass_convert_polydata_to_imagedata(mesh, seg_vtk)
    new_seg = exportVTK2Sitk(new_seg)
    #spacing = new_seg.GetSpacing()
    #spacing = 1./spacing[0] * np.array(spacing)
    sitk.WriteImage(new_seg, out_fn)

def create_ref_template(size, origin=(0., 0., 0.), extend=(1., 1., 1.)):
    arr = np.zeros(size).astype(np.uint8)
    ref_seg = sitk.GetImageFromArray(arr.transpose(2, 1, 0))
    spacing = np.array(extend)/np.array(size)
    ref_seg.SetSpacing(spacing)
    ref_seg.SetOrigin(np.array(origin))
    return ref_seg

if __name__ == '__main__':
    
    dir_n = '/scratch/users/fwkong/CHD/output/wh_raw_tests_cleanedall/GenLargeWHSep3/Ours_final_Apr11/alter5Joint1000_latent4_lipx100_NOuseDiag_Init1017_DivMag0.01_Pad0_GradMag0_smplfac20_twophase/train_2200/vsd_shape_gen'

    seg_fn = os.path.join(dir_n, 'ref.nii.gz')
    ref =  create_ref_template((512, 512, 221))
    sitk.WriteImage(ref, seg_fn)

    out_dir = os.path.join(dir_n, 'seg')
    
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
    except Exception as e:
        print(e)
        comm = None
        comm_size = 1
        rank = 0
    
    if rank ==0:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fns = sorted(glob.glob(os.path.join(dir_n, 'mesh', '*.vtp')))
        fns_scatter = [None] * comm_size
        chunck_size = len(fns) // comm_size
        for i in range(comm_size-1):
            fns_scatter[i] = fns[i*chunck_size:(i+1)*chunck_size]
        fns_scatter[-1] = fns[(comm_size-1)*chunck_size:]
    else:
        fns_scatter = None
    if comm is not None:
        fns_scatter = comm.scatter(fns_scatter, root=0)
    else:
        fns_scatter = fns
    
    for mesh_fn in fns_scatter:
        basename = os.path.basename(mesh_fn)[:-4]
        out_fn = os.path.join(out_dir, basename+'.nii.gz')
        convert_to_segmentation(mesh_fn, out_fn, seg_fn, if_vtk=False)
