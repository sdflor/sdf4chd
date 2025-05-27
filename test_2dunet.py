import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import vtk
from vtk_utils.vtk_utils import *
import munet_dataset
from torch.utils.data import DataLoader
import yaml
import functools
import pkbar
import matplotlib.pyplot as plt
import io_utils
import metrics
import SimpleITK as sitk
from munet import Modified3DUNet
from unet_2d import UNet2D
from train_2dunet import convert_shape, convert_shape_back
import argparse

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    config_fn = args.config
    
    MODE = 'test'
    NUM_INTERP = 1
    THRESH = 0.
    with open(config_fn, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])

    # create dataloader
    test = munet_dataset.ImgSDFDataset(cfg['data']['test_dir'], chd_info=cfg['data']['chd_info'], mode=[MODE], use_aug=False)
    dataloader_test = DataLoader(test, batch_size=1, shuffle=False, pin_memory=True)

    net = UNet2D(in_channels=1, out_channels=cfg['net']['n_classes'])
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'net_{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['state_dict'])

    dice_score_list = []
    cfg['data']['output_dir'] = os.path.join(cfg['data']['output_dir'], '{}_{}'.format(MODE, args.epoch))
    
    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])
    
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(dataloader_test):
            print(i)
            img = convert_shape(data['image'].to(device))
            sdf  = net(img)
            sdf = convert_shape_back(sdf, 1)
            
            sdf = sdf.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)[0]
            seg = np.argmax(sdf, axis=-1)
            io_utils.write_seg_to_vtk_mesh(seg, os.path.join(cfg['data']['output_dir'], '{}_unet.vtp'.format(data['filename'][0]))) 
            sitk.WriteImage(sitk.GetImageFromArray(seg.transpose(2, 1, 0)), os.path.join(cfg['data']['output_dir'], '{}_seg.nii.gz'.format(data['filename'][0])))
            if "y" in data.keys():
                print(sdf.shape, data['y'].shape)
                io_utils.write_seg_to_vtk_mesh(data['y'].numpy()[0], os.path.join(cfg['data']['output_dir'], '{}_gt.vtp'.format(data['filename'][0]))) 
                dice_score_list.append(metrics.dice_score(seg, data['y'].numpy()[0]))
    io_utils.write_scores(os.path.join(cfg['data']['output_dir'], 'dice.csv'), dice_score_list)
