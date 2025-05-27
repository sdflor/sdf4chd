import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import vtk
from vtk_utils.vtk_utils import *
import dataset
from torch.utils.data import DataLoader
import yaml
import functools
from gen_network import SDF4CHD, SDF4CHDTester
import pkbar
import matplotlib.pyplot as plt
import io_utils
import metrics
import SimpleITK as sitk
import argparse
import glob
import time
import itertools
import re
from collections import OrderedDict

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def sample_shape_space(net, type_mesh, z_s, cfg, filename, mesh_name):
    points = np.flip(vtk_to_numpy(type_mesh.GetPoints().GetData()), axis=-1)
    points = torch.from_numpy(points.copy()).unsqueeze(0).to(device) * 2. - 1.

    new_mesh = vtk.vtkPolyData()
    new_mesh.DeepCopy(type_mesh)
    new_points, _ = net.decoder.flow(points, None, z_s, inverse=False)
    new_points = (np.flip(new_points.detach().cpu().numpy(), -1) +1.)/2.
    new_mesh.GetPoints().SetData(numpy_to_vtk(np.squeeze(new_points)))
    write_vtk_polydata(new_mesh, os.path.join(cfg['data']['test_mesh_output_dir'], '{}_{}.vtp'.format(filename, mesh_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--config')
    parser.add_argument('--grid_size', type=int, default=2)
    args = parser.parse_args()

    mesh_fn = '/home/users/fwkong/CHD/template/VSD_PUA.vtp'
    mesh_name = 'pua'
    type_mesh = load_vtk_mesh(mesh_fn)

    MODE = ['test']
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])
    
    tester = SDF4CHDTester(device, cell_grid_size=args.grid_size, out_dim=cfg['net']['out_dim'])
    dice_score_list, dice_noCorr_score_list, time_list = [], [], []
    z_vector_list = {}

    # create network and latent codes
    net = SDF4CHD(in_dim=0, \
            out_dim=cfg['net']['out_dim'], \
            num_types=len(cfg['data']['chd_info']['types']), \
            z_t_dim=cfg['net']['z_t_dim'], \
            z_s_dim=cfg['net']['z_s_dim'], \
            type_mlp_num=cfg['net']['type_mlp_num'],\
            ds_mlp_num=cfg['net']['ds_mlp_num'],\
            dx_mlp_num=cfg['net']['dx_mlp_num'], \
            latent_dim=cfg['net']['latent_dim'], \
            ins_norm=cfg['net']['ins_norm'], \
            type_bias=False, \
            lip_reg=cfg['net']['lip_reg'], \
            step_size=cfg['net']['step_size'])
        # initialize Z_s
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'net_{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['state_dict'])


    cfg['data']['test_output_dir'] = os.path.join(cfg['data']['output_dir'], MODE[0]+ '_both_lat{}_fixType_opt_{}'.format(cfg['net']['test_l_dim'], args.epoch))
    cfg['data']['test_mesh_output_dir'] = os.path.join(cfg['data']['output_dir'], MODE[0]+ '_both_lat{}_fixType_opt_{}'.format(cfg['net']['test_l_dim'], args.epoch), 'templates')
    if not os.path.exists(cfg['data']['test_output_dir']):
        os.makedirs(cfg['data']['test_output_dir'])
    if not os.path.exists(cfg['data']['test_mesh_output_dir']):
        os.makedirs(cfg['data']['test_mesh_output_dir'])
   
    train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['output_dir'], chd_info=cfg['data']['chd_info'], mode=MODE, use_aug=False)
    dataloader_test = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            original_copy = bool(re.match("ct_[a-z]+_\d+",data['filename'][0])) or bool(re.match("ct_\d+_image", data['filename'][0]))
            print(i, data['filename'][0], original_copy)
            if original_copy:
                filename = data['filename'][0]
                lat_vec_fn = os.path.join(cfg['data']['test_output_dir'], filename +'_feat.npy')
                z_s = torch.from_numpy(np.load(lat_vec_fn)).to(device).view(1, cfg['net']['z_s_dim'], cfg['net']['test_l_dim'], cfg['net']['test_l_dim'], cfg['net']['test_l_dim'])
                sample_shape_space(net, type_mesh, z_s, cfg, filename, mesh_name) 
    
