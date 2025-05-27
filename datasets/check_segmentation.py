import os
import numpy
import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from pre_process import resample_spacing
import copy

def overlay_seg_on_img(img, seg):
    img = resample_spacing(img, template_size=(256, 256, 256), order=1)[0]
    seg = resample_spacing(seg, template_size=(256, 256, 256), order=0)[0]

    py_img = sitk.GetArrayFromImage(img)
    py_seg = sitk.GetArrayFromImage(seg)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(py_img[128, :, :], cmap=plt.cm.gray, interpolation='bilinear')
    axes[0, 1].imshow(py_img[:, 128, :], cmap=plt.cm.gray, interpolation='bilinear')
    axes[0, 2].imshow(py_img[:, :, 128], cmap=plt.cm.gray, interpolation='bilinear')
    axes[1, 0].imshow(py_img[128, :, :], cmap=plt.cm.gray, interpolation='bilinear')
    axes[1, 1].imshow(py_img[:, 128, :], cmap=plt.cm.gray, interpolation='bilinear')
    axes[1, 2].imshow(py_img[:, :, 128], cmap=plt.cm.gray, interpolation='bilinear')

    my_cmap = copy.copy(plt.cm.get_cmap('jet'))
    my_cmap.set_under(alpha=0)

    axes[0, 0].imshow(py_seg[128, :, :], vmin=1, vmax=7, cmap=my_cmap, alpha=0.2, interpolation='nearest')
    axes[0, 1].imshow(py_seg[:, 128, :], vmin=1, vmax=7, cmap=my_cmap, alpha=0.2, interpolation='nearest')
    axes[0, 2].imshow(py_seg[:, :, 128], vmin=1, vmax=7, cmap=my_cmap, alpha=0.2, interpolation='nearest')

    fig.subplots_adjust(hspace=0.)
    return fig

if __name__ == '__main__':
    img_dir = '/Users/fanweikong/Documents/ImageData/ImageCHD_dataset/ct_train_fixedsp_cube'
    seg_dir = '/Users/fanweikong/Documents/ImageData/ImageCHD_dataset/ct_train_fixedsp_masks_cube_cleaned2_slicer'
    out_dir = os.path.join(seg_dir, 'pngs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    seg_fns = glob.glob(os.path.join(seg_dir, '*.nii.gz'))

    for seg_fn in seg_fns:
        img_fn = os.path.join(img_dir, os.path.basename(seg_fn))
        fig = overlay_seg_on_img(sitk.ReadImage(img_fn), sitk.ReadImage(seg_fn))
        plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)
        plt.savefig(os.path.join(out_dir, os.path.basename(seg_fn).split('.')[0]+'.png'), transparent=True)
        plt.close()

    

