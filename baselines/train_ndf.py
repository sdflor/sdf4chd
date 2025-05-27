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
from ndf import NDF, NDFTester, LipLinearLayer
import pkbar
import matplotlib.pyplot as plt
from io_utils import plot_loss_curves, save_ckp, write_sampled_point, load_ckp
import io_utils
import argparse
import h5py
import random
import math
from torchinfo import summary
from network import act
from dataset import sample_points_from_sdf
from pytorch3d.loss import chamfer_distance
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

def loss_func(sampled_gt_sdv, type_s, outputs, kbar, i):
    recons_noDs_loss = torch.mean(((outputs['recons'][0].squeeze(1) - sampled_gt_sdv)**2))
    gaussian_s_loss = torch.mean(type_s**2)

    total_loss =  4. * recons_noDs_loss + \
           0.01*gaussian_s_loss

    kbar.update(i, values=[("loss", total_loss), ("recons_noDs", recons_noDs_loss), ("gaussian_s_loss", gaussian_s_loss)])
    return total_loss, recons_noDs_loss

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def initialize_type_network(cfg, net, optimizer_nodecay):
    # initilize to fit a heart regardless of the type first
    sdf_py_tmplt = pickle.load(open(cfg['data']['tmplt_sdf'], 'rb'))
    for i in range(500):
        _, points, point_values, _ = sample_points_from_sdf(sdf_py_tmplt, cfg['train']['n_smpl_pts'], cfg['data']['point_sampling_factor'])
        points = points.unsqueeze(0).to(device)
        point_values = point_values.unsqueeze(0).to(device)
        out = act(net.module.decoder.decoder(points))
        recons_loss = torch.mean(((out.permute(0, 2, 1) - point_values)**2)*(point_values+1))
        recons_loss.backward()
        print("ITER {}: Recons loss: {}.".format(i, recons_loss.item()))
        optimizer_nodecay.step()
    return net

def regularize_lip_bound(net):
    prod_c_0 = 1.
    for layer in net.module.decoder.decoder.children():
        if isinstance(layer, LipLinearLayer):
            prod_c_0 = prod_c_0 * F.softplus(layer.c)
    return prod_c_0

def update_prediction(dataloader, lat_vecs, net, cfg, epoch):
    tester = NDFTester(device, cell_grid_size=1, out_dim=cfg['net']['out_dim'])
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if data['filename'][0] in ['ct_1001_image', 'ct_1007_image', 'ct_1010_image', 'ct_1015_image', 'ct_1025_image', 'ct_1042_image', 'ct_1052_image']:
                z_s = lat_vecs(data['idx'].to(device))
                z_s = z_s.view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
                sdf_noCorr = tester.z2voxel(z_s, net.module, num_blocks=1, out_block=0, get_tmplt_coords=False)
                type_sdf = tester.z2voxel(None, net.module, num_blocks=1, out_type=True)
                io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['output_dir'], 'noCorr_{}_epoch{}.vtp'.format(data['filename'][0], epoch)), 0.5) 
                io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['output_dir'], 'type_{}_epoch{}.vtp'.format(data['filename'][0], epoch)), 0.5) 
    net.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config') 
    args = parser.parse_args()
    start_epoch = 0
    mode = ['train']
    use_aug = False
    use_error = False
    wt_dcy = 0.
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])

    # create dataloader
    train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['output_dir'], cfg['data']['point_sampling_factor'], \
        cfg['data']['chd_info'], mode=mode, use_aug=use_aug, use_error=use_error, train=True, pad_num=cfg['train']['pad_num'], \
        binary=cfg['train']['binary'])
    dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=True, \
            worker_init_fn = worker_init_fn, num_workers=cfg['train']['batch_size'] * 2)
    #dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=True, \
    #        worker_init_fn = worker_init_fn, num_workers=0)
    dataloader_val = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=0)

    # create network and latent codes
    net = NDF(in_dim=0, \
            out_dim=cfg['net']['out_dim'], \
            num_types=len(cfg['data']['chd_info']['types']), \
            z_s_dim=cfg['net']['z_s_dim'], \
            type_mlp_num=cfg['net']['type_mlp_num'],\
            dx_mlp_num=cfg['net']['dx_mlp_num'], \
            latent_dim=cfg['net']['latent_dim'], \
            ins_norm=cfg['net']['ins_norm'], \
            type_bias=False, \
            lip_reg=cfg['net']['lip_reg'])
    # initialize Z_s
    lat_vecs = torch.nn.Embedding(len(train), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
    torch.nn.init.kaiming_normal_(lat_vecs.weight.data, a=0.02, nonlinearity='leaky_relu')
    net = nn.DataParallel(net)
    net.to(device)
    
    # no weight decay for type prediction
    subnets_type = ['decoder.decoder']
    params_type, params_else = set(), set()
    for n in subnets_type:
        params_type |= set(nm for nm, p in net.named_parameters() if n in nm)
    all_names = set(nm for nm, p in net.named_parameters())
    params_else = all_names.difference(params_type)
    
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = params_type & params_else
    union_params = params_type | params_else 
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    
    optimizer_nodecay = torch.optim.Adam((params_dict[n] for n in sorted(list(params_type))), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_decay = torch.optim.Adam((params_dict[n] for n in sorted(list(params_else))), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=wt_dcy)
    optimizer_zs = torch.optim.Adam(lat_vecs.parameters(), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)  

    scheduler_nodecay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nodecay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    scheduler_zs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_zs, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    optimizers = [optimizer_nodecay, optimizer_decay, optimizer_zs]
    schedulers = [scheduler_nodecay, scheduler_decay, scheduler_zs]
    
    if os.path.exists(os.path.join(cfg['data']['output_dir'], 'net.pt')):
        print("LOADING LASTEST CHECKPOINT")
        net, optimizers, schedulers, start_epoch = io_utils.load_ckp(os.path.join(cfg['data']['output_dir'], 'net.pt'), \
                net, optimizers, schedulers)
        lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code.pt'))['latent_codes'])
    else:
        # initialize type network - helps with convergence
        # Only train type network (encoder and decoder) and freeze the rest
        if cfg['train']['init']: 
            for n in params_else:
                params_dict[n].requires_grad = False
                params_dict[n].grad = None
            net = initialize_type_network(cfg, net, optimizer_nodecay)
            for n in params_else:
                params_dict[n].requires_grad = True
    
    fix_type = False
    # start training
    for epoch in range(start_epoch, cfg['train']['epoch']):
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        if epoch >= cfg['train']['alter']['joint_num'] and epoch < cfg['train']['alter']['end_num']:
            if epoch % cfg['train']['alter']['alter_num'] == 0:
                print("RESETING Optimizer")
                optimizers[0] = torch.optim.Adam((params_dict[n] for n in sorted(list(params_type))), lr=optimizers[0].param_groups[0]['lr'], betas=(0.5, 0.999), weight_decay=0.0)
                optimizers[1] = torch.optim.Adam((params_dict[n] for n in sorted(list(params_else))), lr=optimizers[1].param_groups[0]['lr'], betas=(0.5, 0.999), weight_decay=wt_dcy)
                optimizers[2] = torch.optim.Adam(lat_vecs.parameters(), lr=optimizers[2].param_groups[0]['lr'], betas=(0.5, 0.999), weight_decay=0.0)
            if (((epoch-cfg['train']['alter']['joint_num']) // cfg['train']['alter']['alter_num']) % 2 == 1):
                print("TRAIN TYPE ONLY")
                fix_type = False
                for n in params_type:
                    params_dict[n].requires_grad = True
                for n in params_else:
                    params_dict[n].requires_grad = False
                    params_dict[n].grad = None
                for p in lat_vecs.parameters():
                    p.requires_grad = False
                    p.grad = None
            else:
                fix_type = True
                print("TRAIN SHAPE ONLY, Updating TYPE BOUNDARY POINTS")
                for n in params_else:
                    params_dict[n].requires_grad = True
                for n in params_type:
                    params_dict[n].requires_grad = False
                    params_dict[n].grad = None
                for p in lat_vecs.parameters():
                    p.requires_grad = True
        elif epoch >= cfg['train']['alter']['end_num']:
            print("TRAIN TYPE AND SHAPE")
            fix_type = False
            for n in params_else:
                params_dict[n].requires_grad = True
            for n in params_type:
                params_dict[n].requires_grad = True
            for p in lat_vecs.parameters():
                p.requires_grad = True
        total_recons_noDs_loss = 0.
        for i, data in enumerate(dataloader_train):
            z_s = lat_vecs(data['idx'].to(device))
            z_s = z_s.view(cfg['train']['batch_size'], cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
            points = data['points'].to(device)
            point_values = data['pt_sdv'].to(device)
            if epoch == start_epoch and i ==0:
                summary(net, [tuple(z_s.shape), tuple(points.shape)])
                print(points.shape, point_values.shape)
                io_utils.write_sampled_point(points[0], point_values[0], os.path.join(cfg['data']['output_dir'], 'sample_{}_epoch{}.vtp'.format(i, epoch)))
            net.zero_grad()
            outputs = net(z_s, points) 
            loss, recons_noDs_loss = loss_func(point_values, z_s, outputs, kbar, i)
            total_recons_noDs_loss += recons_noDs_loss.item()
            if cfg['net']['lip_reg']:
                loss = loss + 1e-10*regularize_lip_bound(net)
            loss.backward()
            for o in optimizers:
                o.step()
        with torch.no_grad():
            for s in schedulers:
                s.step(total_recons_noDs_loss)
            if (epoch+1) % 500 ==0:
                update_prediction(dataloader_val, lat_vecs, net, cfg, epoch)
                all_latents = lat_vecs.state_dict()
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(epoch+1)))
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code.pt'))
                save_ckp(net, optimizers, schedulers, epoch, cfg['data']['output_dir']) 
