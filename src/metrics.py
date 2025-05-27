import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import SimpleITK as sitk
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../vtk_utils'))
from vtk_utils import *
import csv

def extract_surface(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.ColorRegionsOn()
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def surface_distance(p_surf, g_surf):
    dist_fltr = vtk.vtkDistancePolyDataFilter()
    dist_fltr.SetInputData(1, p_surf)
    dist_fltr.SetInputData(0, g_surf)
    dist_fltr.SignedDistanceOff()
    dist_fltr.Update()
    distance = vtk_to_numpy(dist_fltr.GetOutput().GetPointData().GetArray('Distance'))
    return distance, dist_fltr.GetOutput()

def evaluate_poly_distances(poly, gt, NUM):
    # compute assd and hausdorff distances
    assd_list, haus_list, poly_list = [], [], []
    poly =extract_surface(poly)
    for i in range(NUM):
        poly_i = thresholdPolyData(poly, 'Scalars_', (i+1, i+1),'cell')
        if poly_i.GetNumberOfPoints() == 0:
            print("Mesh based methods.")
            poly_i = thresholdPolyData(poly, 'RegionId', (i, i), 'point')
        gt_i = thresholdPolyData(gt, 'Scalars_', (i+1, i+1),'cell')
        pred2gt_dist, pred2gt = surface_distance(gt_i, poly_i)
        gt2pred_dist, gt2pred = surface_distance(poly_i, gt_i)
        assd = (np.mean(pred2gt_dist)+np.mean(gt2pred_dist))/2
        haus = max(np.max(pred2gt_dist), np.max(gt2pred_dist))
        assd_list.append(assd)
        haus_list.append(haus)
        poly_list.append(pred2gt)

    poly_dist = appendPolyData(poly_list)
    # whole heart
    pred2gt_dist, pred2gt = surface_distance(gt, poly)
    gt2pred_dist, gt2pred = surface_distance(poly, gt)

    assd = (np.mean(pred2gt_dist)+np.mean(gt2pred_dist))/2
    haus = max(np.max(pred2gt_dist), np.max(gt2pred_dist))

    assd_list.insert(0, assd)
    haus_list.insert(0, haus)
    return assd_list, haus_list, poly_dist

def dice_score(pred, true, k=1):
    intersection = np.sum(pred[true==k]==k)
    union = np.sum((pred==k).astype(np.int32)) + np.sum((true==k).astype(np.int32))
    if union == 0:
        return 1
    return 2.*float(intersection)/float(union)

def dice_score_list(pred, true, k=1):
    pred = pred.astype(np.int32)
    true = true.astype(np.int32)
    return [dice_score(p, t, k) for p,t in zip(pred, true)]

def iou_score(pred, true, k=1):
    intersection = np.sum(pred[true==k]==k)
    union = np.sum((pred==k).astype(np.int32)) + np.sum((true==k).astype(np.int32)) - intersection
    if union == 0:
        return 1
    return float(intersection)/float(union)

def iou_score_list(pred, true, k=1):
    pred = pred.astype(np.int32)
    true = true.astype(np.int32)
    return [iou_score(p, t, k) for p,t in zip(pred, true)]

def hausdorff_score(pred, true, k=1):
    if np.sum(pred==k)==0 and np.sum(true==k)==0:
        return 0
    elif np.sum(pred==k)==0 or np.sum(true==k)==0:
        return 1000
    pred = pred.astype(np.int32)
    true = true.astype(np.int32)
    return max(directed_hausdorff((pred==k).astype(float), (true==k).astype(float))[0], 
            directed_hausdorff((true==k).astype(float), (pred==k).astype(float))[0])

def volume_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)
    num_class = np.unique(true)
    verr_out = [None]*len(num_class)
    
    true_v_total = np.sum(true>0)
    total_vol_err = 0.
    for i in range(1, len(num_class)):
        true_v = np.sum(true == num_class[i])
        pred_v = np.sum(pred == num_class[i])
        verr_out[i] = np.abs(pred_v-true_v)/true_v

        total_vol_err += verr_out[i] * float(true_v)/float(true_v_total)
    verr_out[0] = total_vol_err
    return verr_out

def jaccard_score(pred, true):
    pred = pred.astype(np.int)
    true = true.astype(np.int)
    num_class = np.unique(true)

    #change to one hot
    jac_out = [None]*len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        jac_out[i] = np.sum(pred_c*true_c) / (np.sum(pred_c) + np.sum(true_c)-np.sum(pred_c*true_c))

    mask =( pred > 0 )+ (true > 0)
    jac_out[0] = np.sum((pred==true)[mask]) / (np.sum(pred>0) + np.sum(true>0)-np.sum((pred==true)[mask]))
    return jac_out 

def dice_score_from_sdf(pred_sdf, true, thresh=0.5):
    pred_seg = np.argmax(pred_sdf, axis=-1) + 1
    pred_seg[np.all(pred_sdf<thresh, axis=-1)] = 0
    return dice_score(pred_seg, true)

def jaccard_score_from_sdf(pred_sdf, true, thresh=0.5):
    pred_seg = np.argmax(pred_sdf, axis=-1) + 1
    pred_seg[np.all(pred_sdf<thresh, axis=-1)] = 0
    return jaccard_score(pred_seg, true)

