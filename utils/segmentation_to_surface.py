import os
import glob
import vtk
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from vtk_utils.vtk_utils import *
import SimpleITK as sitk
import argparse


def convert_to_surfaces(fn, ids=[], ref='', rate=0., closing=False, median=False):
    im = sitk.ReadImage(fn)
    im_scalars = sitk.GetArrayFromImage(im)
    print(np.unique(im_scalars))
    for i in np.unique(im_scalars):
        im_scalars[im_scalars==i] = 1. if i in ids else 0.
    im_tmp = sitk.GetImageFromArray(im_scalars)
    im_tmp.CopyInformation(im)

    if closing:
        filt = sitk.BinaryMorphologicalClosingImageFilter() 
        filt.SetKernelRadius(10)
        im_tmp = filt.Execute(im_tmp)
    if median:
        filt = sitk.BinaryMedianImageFilter() 
        filt.SetRadius(3)
        im_tmp = filt.Execute(im_tmp)
    im_vtk = exportSitk2VTK(im_tmp, spacing=im_tmp.GetSpacing())[0]
    mesh = vtk_marching_cube_multi(im_vtk, 0)
    mesh = smooth_polydata(mesh, 20, feature=False)
    mesh = decimation(mesh, rate)
    mesh = smooth_polydata(mesh, 20, smoothingFactor=0.5)
    return mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_fn', help='File name of the segmentation')
    parser.add_argument('--ids', nargs='+', type=int, help='Ids of the segmentation to combine and generate surface')
    parser.add_argument('--output_fn', help='File name of the surface')
    parser.add_argument('--im_ref_fn', default='', help='Reference image file of the segmentation')
    parser.add_argument('--decimate', type=float,default=0., help='Decimation rate')
    args = parser.parse_args()
    dir_n = os.path.dirname(args.output_fn)
    try:
        os.makedirs(dir_n)
    except Exception as e: print(e)
    mesh = convert_to_surfaces(args.im_fn, args.ids, args.im_ref_fn, args.decimate)
    write_vtk_polydata(mesh, args.output_fn)
