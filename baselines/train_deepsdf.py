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
from deepsdf import DeepSDF,DeepSDFTester
import pkbar
from io_utils import plot_loss_curves, save_ckp, write_sampled_point, load_ckp
import io_utils
import argparse
import h5py
import random
import math
from torchinfo import summary
from network import act
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

def loss_func(sampled_gt_sdv, type_s, outputs, kbar, i):
    recons_loss = torch.mean(((outputs.squeeze(1) - sampled_gt_sdv.permute(0, 2, 1))**2))
    gaussian_s_loss = torch.mean(type_s**2)

    total_loss = 4 * (recons_loss) + 0.01*gaussian_s_loss

    kbar.update(i, values=[("loss", total_loss), ("recons", recons_loss), \
        ("gaussian_s_loss", gaussian_s_loss)])
    return total_loss, recons_loss

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def update_prediction(dataloader, lat_vecs, net, cfg, epoch):
    tester = DeepSDFTester(device, cell_grid_size=1, out_dim=cfg['net']['out_dim'])
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if data['filename'][0] in ['ct_1001_image', 'ct_1007_image', 'ct_1010_image', 'ct_1015_image', 'ct_1042_image', 'ct_1052_image']:
                chd_type = data['chd_type'].to(device)
                z_s = lat_vecs(data['idx'].to(device))
                z_s = z_s.view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
                sdf = tester.z2voxel(z_s, net.module)
                io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['output_dir'], 'pred_{}_epoch{}.vtp'.format(data['filename'][0], epoch)), 0.5) 
    net.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config') 
    args = parser.parse_args()
    start_epoch = 0
    mode = ['train']
    use_aug = False
    use_error = False
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])

    # create dataloader
    train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['output_dir'], cfg['data']['point_sampling_factor'], \
        cfg['data']['chd_info'], mode=mode, use_aug=use_aug, use_error=use_error, train=True, pad_num=cfg['train']['pad_num'], \
        binary=cfg['train']['binary'])
    dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=cfg['train']['batch_size'] * 2)
    dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=0)
    dataloader_val = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=0)

    # create network and latent codes
    net = DeepSDF(cfg['net']['z_s_dim'], \
            out_dim=cfg['net']['out_dim'], \
            z_s_dim=cfg['net']['z_s_dim'], \
            ins_norm=cfg['net']['ins_norm'])
    # initialize Z_s
    lat_vecs = torch.nn.Embedding(len(train), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
    torch.nn.init.kaiming_normal_(lat_vecs.weight.data, a=0.02, nonlinearity='leaky_relu')
    net = nn.DataParallel(net)
    net.to(device)
    
    optimizer_nodecay = torch.optim.Adam(net.parameters(), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_zs = torch.optim.Adam(lat_vecs.parameters(), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)  
    scheduler_nodecay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nodecay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    scheduler_zs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_zs, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    optimizers = [optimizer_nodecay, optimizer_zs]
    schedulers = [scheduler_nodecay, scheduler_zs]
    
    if os.path.exists(os.path.join(cfg['data']['output_dir'], 'net.pt')):
        print("LOADING LASTEST CHECKPOINT")
        net, optimizers, schedulers, start_epoch = io_utils.load_ckp(os.path.join(cfg['data']['output_dir'], 'net.pt'), \
                net, optimizers, schedulers)
        lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code.pt'))['latent_codes'])
    
    # start training
    for epoch in range(start_epoch, cfg['train']['epoch']):
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        total_recons_noDs_loss = 0.
        for i, data in enumerate(dataloader_train):
            z_s = lat_vecs(data['idx'].to(device))
            z_s = z_s.view(cfg['train']['batch_size'], cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
            points = data['points'].to(device)
            point_values = data['pt_sdv'].to(device)
            if epoch == start_epoch and i ==0:
                summary(net, [tuple(z_s.shape), tuple(points.shape)])
                io_utils.write_sampled_point(points[0], point_values[0], os.path.join(cfg['data']['output_dir'], 'sample_{}_epoch{}.vtp'.format(i, epoch)))
            net.zero_grad()
            outputs = net(z_s, points) 
            loss, recons_noDs_loss = loss_func(point_values, z_s, outputs, kbar, i)
            total_recons_noDs_loss += recons_noDs_loss.item()
            loss.backward()
            for o in optimizers:
                o.step()
        with torch.no_grad():
            for s in schedulers:
                s.step(total_recons_noDs_loss)
            if (epoch+1) % 100 ==0:
                update_prediction(dataloader_val, lat_vecs, net, cfg, epoch)
                all_latents = lat_vecs.state_dict()
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(epoch+1)))
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code.pt'))
                save_ckp(net, optimizers, schedulers, epoch, cfg['data']['output_dir']) 
