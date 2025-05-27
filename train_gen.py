import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.getcwd())  # Add current directory to Python path
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
from gen_network import SDF4CHD, SDF4CHDTester, LipLinearLayer
import pkbar
import matplotlib.pyplot as plt
from io_utils import plot_loss_curves, save_ckp, write_sampled_point, load_ckp
import io_utils
import argparse
import h5py
import random
import math
import net_utils
from torchinfo import summary
from network import act
from dataset import sample_points_from_sdf
# from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter
import re
import signal
import sys
import time
import json

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

# Global variables to store training state
global_training_state = {
    'net': None,
    'lat_vecs': None,
    'lat_vecs_ds': None,
    'optimizers': None,
    'schedulers': None,
    'epoch': None,
    'cfg': None,
    'output_dir': None,
    'two_shape_codes': None
}

def save_checkpoint(forced=False):
    """Save checkpoint with current training state"""
    if global_training_state['net'] is None:
        return
        
    print("\nSaving checkpoint...")
    all_latents = global_training_state['lat_vecs'].state_dict()
    torch.save({'epoch': global_training_state['epoch'] + 1, 'latent_codes': all_latents}, 
               os.path.join(global_training_state['output_dir'], 'code.pt'))
    
    if global_training_state['two_shape_codes']:
        all_latents_ds = global_training_state['lat_vecs_ds'].state_dict()
        torch.save({'epoch': global_training_state['epoch'] + 1, 'latent_codes': all_latents_ds}, 
                  os.path.join(global_training_state['output_dir'], 'code_ds.pt'))
    
    save_ckp(global_training_state['net'], 
             global_training_state['optimizers'], 
             global_training_state['schedulers'], 
             global_training_state['epoch'], 
             global_training_state['output_dir'])
    print("Checkpoint saved successfully")

def signal_handler(signum, frame):
    """Handle interruption signals by saving checkpoint before exit"""
    print('\nSignal received. Saving checkpoint before exit...')
    save_checkpoint(forced=True)
    print('Checkpoint saved. Exiting...')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

def loss_func(weights, sampled_gt_sdv, sampled_gt_distance, chd_type, type_z, type_s, outputs, kbar, i, writer=None):
    # Debug print to understand the structure
    # print("Outputs type:", type(outputs))
    
    # Handle outputs from the model
    if isinstance(outputs, tuple):
        # If outputs is a tuple, take the first element
        outputs = outputs[0]
    
    if isinstance(outputs, dict):
        # When outputs is a dictionary (training mode)
        if 'recons' in outputs:
            recons_list = outputs['recons']  # This should be a list of tensors
            # Take the first tensor from each reconstruction
            recons_noDs = recons_list[0]
            recons = recons_list[1]
            div_integral = outputs.get('div_integral', torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device))
            grad_mag = outputs.get('grad_mag', torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device))
    else:
        # When outputs is a list (evaluation mode)
        recons_noDs = outputs[0]
        recons = outputs[1]
        div_integral = torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device)
        grad_mag = torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device)

    # print("Debug shapes before processing:")
    # print("recons_noDs shape:", recons_noDs.shape)
    # print("recons shape:", recons.shape)
    # print("sampled_gt_sdv shape:", sampled_gt_sdv.shape)

    # Make sure tensors are in the right shape and have gradients enabled
    if len(recons_noDs.shape) == 3:
        recons_noDs = recons_noDs.squeeze(1)
    if len(recons.shape) == 3:
        recons = recons.squeeze(1)

    # Handle shape mismatches by reshaping to match batch size and points
    batch_size = sampled_gt_sdv.shape[0]
    num_points = sampled_gt_sdv.shape[-1]

    # Reshape tensors to [batch_size, num_points]
    if recons_noDs.shape != sampled_gt_sdv.shape:
        recons_noDs = recons_noDs.reshape(batch_size, num_points)
    if recons.shape != sampled_gt_sdv.shape:
        recons = recons.reshape(batch_size, num_points)

    # If sampled_gt_sdv has an extra dimension, take the appropriate slice
    if len(sampled_gt_sdv.shape) == 3:
        sampled_gt_sdv = sampled_gt_sdv[:, 0, :]  # Take first slice of middle dimension

    # print("Debug shapes after processing:")
    # print("recons_noDs shape:", recons_noDs.shape)
    # print("recons shape:", recons.shape)
    # print("sampled_gt_sdv shape:", sampled_gt_sdv.shape)

    # Ensure tensors require gradients
    recons_noDs = recons_noDs.requires_grad_(True)
    recons = recons.requires_grad_(True)

    recons_noDs_loss = torch.mean(((recons_noDs - sampled_gt_sdv)**2))
    recons_loss = torch.mean(((recons - sampled_gt_sdv)**2))
    
    # Only compute type losses if type_z is not None
    if type_z is not None:
        gaussian_t_loss = torch.mean(type_z**2)
    else:
        gaussian_t_loss = torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device)
        
    gaussian_s_loss = torch.mean(type_s**2)
    
    if weights['div_integral'] > 0.:
        div_integral = torch.mean(div_integral)
    else:
        div_integral = torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device)
    
    if weights['grad_mag'] > 0.:
        grad_mag = torch.mean(grad_mag * torch.clamp(torch.min(sampled_gt_distance, dim=1)[0], min=0.)) 
    else:
        grad_mag = torch.tensor(0., dtype=torch.float32, requires_grad=True).to(sampled_gt_sdv.device)

    total_loss = weights['recons_loss'] * (recons_loss) + \
            weights['recons_noDs_loss'] * recons_noDs_loss + \
            weights['gaussian_t_loss'] * gaussian_t_loss + \
            weights['gaussian_s_loss'] * gaussian_s_loss + \
            weights['div_integral'] * div_integral + \
            weights['grad_mag'] * grad_mag

    kbar.update(i, values=[("loss", total_loss), ("recons", recons_loss), ("recons_noDs", recons_noDs_loss),  
        ("gaussian_s_loss", gaussian_s_loss), ("gaussian_t_loss", gaussian_t_loss), ("div_integral", div_integral), ("grad_mag", grad_mag)])
    
    if writer is not None:
        writer.add_scalar("Loss/total_loss", total_loss)
        writer.add_scalar("Loss/recons_loss", recons_loss)
        writer.add_scalar("Loss/recons_noDs_loss", recons_noDs_loss)
        writer.add_scalar("Loss/div_integral", div_integral)
    
    return total_loss, recons_noDs_loss

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def initialize_type_network(cfg, net, optimizer_nodecay):
    if not cfg['net']['use_type']:
        return net
        
    # initilize to fit a heart regardless of the type first
    sdf_py_tmplt = pickle.load(open(cfg['data']['tmplt_sdf'], 'rb'))
    for i in range(500):
        _, points, point_values, _ = sample_points_from_sdf(sdf_py_tmplt, cfg['train']['n_smpl_pts'], cfg['data']['point_sampling_factor'])
        points = points.unsqueeze(0).to(device)
        point_values = point_values.unsqueeze(0).to(device)
        chd_type = torch.zeros((1, len(cfg['data']['chd_info']['types']))).float().to(device)  # Changed to zeros since we don't use types
        if net.module.use_diag:
            z_t = chd_type
        else:
            z_t = net.module.type_encoder(chd_type)
        out = act(net.module.decoder.decoder(z_t, points))
        recons_loss = torch.mean(((out.permute(0, 2, 1) - point_values)**2)*(point_values+1))
        recons_loss.backward()
        # print("ITER {}: Recons loss: {}.".format(i, recons_loss.item()))
        optimizer_nodecay.step()
    return net

def regularize_lip_bound(net):
    prod_c_0 = 1.
    for layer in net.module.decoder.decoder.children():
        if isinstance(layer, LipLinearLayer):
            prod_c_0 = prod_c_0 * F.softplus(layer.c)
    return prod_c_0

def update_prediction(dataloader, lat_vecs, lat_vecs_ds, net, cfg, epoch, start_epoch=0, optimizers=None, schedulers=None):
    tester = SDF4CHDTester(device, cell_grid_size=1, out_dim=cfg['net']['out_dim'])
    kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
    total_recons_noDs_loss = 0.
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if data['filename'][0] in ['pulse_00001_artery', 'pulse_00002_artery', 'pulse_00003_artery', 'pulse_00004_artery', 'pulse_00005_artery']:
                # Get points from data
                points = data['points'].to(device)
                point_values = data['point_values'].to(device)
                point_values_sdv = data['point_values_sdv'].to(device)
                
                # Initialize chd_type
                chd_type = None
                if cfg['net']['use_type']:
                    chd_type = data['chd_type'].to(device)
                    if net.module.use_diag:
                        z_t = chd_type
                    else:
                        z_t = net.module.type_encoder(chd_type)
                else:
                    z_t = None

                z_s = lat_vecs(data['idx'].to(device))
                # Calculate the proper dimensions
                total_elements = z_s.numel()  # Get total number of elements
                l_dim = cfg['net']['l_dim']
                batch_size = z_s.shape[0]  # Get batch size from the tensor
                
                # Calculate z_s_dim to make total elements match
                z_s_dim = total_elements // (batch_size * l_dim * l_dim * l_dim)
                
                # First reshape to combine all dimensions except batch
                z_s = z_s.reshape(batch_size, -1)
                
                # Then reshape to final dimensions, ensuring total elements match
                z_s = z_s.reshape(batch_size, z_s_dim, l_dim, l_dim, l_dim)
                
                if cfg['net']['two_shape_codes']:
                    z_s_ds = lat_vecs_ds(data['idx'].to(device))
                    # Apply the same reshaping to z_s_ds
                    z_s_ds = z_s_ds.reshape(batch_size, -1)
                    z_s_ds = z_s_ds.reshape(batch_size, z_s_dim, l_dim, l_dim, l_dim)
                else:
                    z_s_ds = z_s

                if cfg['net']['use_type']:
                    outputs, z_t = net(z_s, z_s_ds, points, chd_type)
                else:
                    outputs = net(z_s, z_s_ds, points)

                if epoch == start_epoch and i == 0:
                    # Print model summary
                    if cfg['net']['use_type']:
                        summary(net, [tuple(z_s.shape), tuple(z_s.shape), tuple(points.shape), tuple(chd_type.shape)])
                    else:
                        summary(net, [tuple(z_s.shape), tuple(z_s.shape), tuple(points.shape)])
                    print(points.shape, point_values.shape)
                    io_utils.write_sampled_point(points[0], point_values[0], os.path.join(cfg['data']['output_dir'], 'sample_{}_epoch{}.vtp'.format(i, epoch)))

                if cfg['net']['lip_reg']:
                    if cfg['net']['use_type']:
                        loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, z_t, torch.cat([z_s, z_s_ds], dim=-1), outputs, kbar, i)
                    else:
                        loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, None, torch.cat([z_s, z_s_ds], dim=-1), outputs, kbar, i)
                else:
                    if cfg['net']['use_type']:
                        loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, z_t, z_s, outputs, kbar, i)
                    else:
                        loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, None, z_s, outputs, kbar, i)
                
                total_recons_noDs_loss += loss[1].item()

        # Move checkpoint saving outside the inner loop
        if (epoch+1) % cfg['train']['save_every'] == 0:
            all_latents = lat_vecs.state_dict()
            torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(epoch+1)))
            torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code.pt'))
            if cfg['net']['two_shape_codes']:
                all_latents_ds = lat_vecs_ds.state_dict()
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents_ds}, os.path.join(cfg['data']['output_dir'], 'code_ds{}.pt'.format(epoch+1)))
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents_ds}, os.path.join(cfg['data']['output_dir'], 'code_ds.pt'))
            save_ckp(net, optimizers, schedulers, epoch, cfg['data']['output_dir'])
    return loss[1].item()

def train_model():
    train_losses = []
    val_losses = []
    output_dir = cfg['data']['output_dir'] 
    loss_json_path = os.path.join(output_dir, 'training_losses.json')
    os.makedirs(output_dir, exist_ok=True)
    
    # ... existing code ...
    for epoch in range(start_epoch, cfg['train']['epoch']):
        print(f"Epoch {epoch} of {cfg['train']['epoch']}")
        global_training_state['epoch'] = epoch  # Update epoch in global state at each iteration
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        total_recons_noDs_loss = 0.
        for i, data in enumerate(dataloader_train):
            points = data['points'].to(device)
            point_values = data['point_values'].to(device)
            point_values_sdv = data['point_values_sdv'].to(device)
            
            # Debug prints for gradient tracking
            # # print("\nGradient tracking status:")
            # print("points.requires_grad:", points.requires_grad)
            # print("point_values.requires_grad:", point_values.requires_grad)
            # print("point_values_sdv.requires_grad:", point_values_sdv.requires_grad)
            
            if cfg['net']['use_type']:
                chd_type = data['chd_type'].to(device)
                # print("chd_type.requires_grad:", chd_type.requires_grad)
            else:
                chd_type = None

            z_s = lat_vecs(data['idx'].to(device))
            # print("z_s.requires_grad:", z_s.requires_grad)
            
            # Calculate the proper dimensions
            total_elements = z_s.numel()
            l_dim = cfg['net']['l_dim']
            batch_size = z_s.shape[0]
            z_s_dim = total_elements // (batch_size * l_dim * l_dim * l_dim)
            
            z_s = z_s.reshape(batch_size, -1)
            z_s = z_s.reshape(batch_size, z_s_dim, l_dim, l_dim, l_dim)
            
            if cfg['net']['two_shape_codes']:
                z_s_ds = lat_vecs_ds(data['idx'].to(device))
                z_s_ds = z_s_ds.reshape(batch_size, -1)
                z_s_ds = z_s_ds.reshape(batch_size, z_s_dim, l_dim, l_dim, l_dim)
                # print("z_s_ds.requires_grad:", z_s_ds.requires_grad)
            else:
                z_s_ds = z_s

            if cfg['net']['use_type']:
                outputs, z_t = net(z_s, z_s_ds, points, chd_type)
                # print("z_t.requires_grad:", z_t.requires_grad)
            else:
                outputs = net(z_s, z_s_ds, points)

            # # Debug print for outputs
            # if isinstance(outputs, dict):
            #     print("outputs['recons'][0].requires_grad:", outputs['recons'][0].requires_grad)
            #     print("outputs['recons'][1].requires_grad:", outputs['recons'][1].requires_grad)

            net.zero_grad()
            if cfg['net']['lip_reg']:
                if cfg['net']['use_type']:
                    loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, z_t, torch.cat([z_s, z_s_ds], dim=-1), outputs, kbar, i)
                else:
                    loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, None, torch.cat([z_s, z_s_ds], dim=-1), outputs, kbar, i)
            else:
                if cfg['net']['use_type']:
                    loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, z_t, z_s, outputs, kbar, i)
                else:
                    loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, None, z_s, outputs, kbar, i)
            
            # print("total_loss.requires_grad:", loss[0].requires_grad)
            total_recons_noDs_loss += loss[1].item()
            train_losses.append(loss[1].item())
            
            try:
                loss[0].backward()
                # print("Backward pass successful")
            except Exception as e:
                # print("Error in backward pass:", str(e))
                for name, param in net.named_parameters():
                    if param.grad is None and param.requires_grad:
                        print(f"Parameter {name} has no gradient")
            
            for o in optimizers:
                o.step()
            
        with torch.no_grad():
            for s in schedulers:
                s.step(total_recons_noDs_loss)
            if (epoch+1) % cfg['train']['save_every'] == 0:
                save_checkpoint()
                if two_shape_codes:
                    val_losses.append(update_prediction(dataloader_val, lat_vecs, lat_vecs_ds, net, cfg, epoch))
                else:
                    val_losses.append(update_prediction(dataloader_val, lat_vecs, lat_vecs, net, cfg, epoch))
    
    losses = {
    "train": train_losses,
    "val": val_losses}

    with open(loss_json_path, "w") as f:
        json.dump(losses, f)
    # Final save at end of training
    save_checkpoint()
    writer.flush()
    writer.close()

def check_network_parameters(net):
    print("\nChecking network parameters:")
    for name, param in net.named_parameters():
        print(f"{name}:")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  shape: {param.shape}")
        if not param.requires_grad:
            print("  WARNING: Parameter does not require gradients!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/gen_test_wh.yml')
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
    
    writer = SummaryWriter(cfg['data']['output_dir'])
    two_shape_codes = cfg['net']['two_shape_codes']

    # Update global state
    global_training_state['output_dir'] = cfg['data']['output_dir']
    global_training_state['two_shape_codes'] = two_shape_codes
    global_training_state['cfg'] = cfg

    # Ensure chd_info exists with at least one type
    if 'chd_info' not in cfg['data']:
        cfg['data']['chd_info'] = {'types': ['normal']}

    # create dataloader
    train_dataset = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['chd_info'].get('type_dir', ''), cfg['data']['point_sampling_factor'], \
        cfg['data']['chd_info'], mode=mode, use_aug=use_aug, use_error=use_error, train=True, pad_num=cfg['train']['pad_num'], \
        binary=cfg['train']['binary'])
    dataloader_train = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=False, worker_init_fn = worker_init_fn, num_workers=0)
    dataloader_val = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=True, worker_init_fn = worker_init_fn, num_workers=0)
   
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
            step_size=cfg['net']['step_size'], \
            use_diag=cfg['net']['use_diag'], \
            use_type=cfg['net']['use_type'], \
            div_loss=True if cfg['train']['weights']['div_integral']> 0. else False, \
            act_func=net_utils.act if cfg['train']['binary'] else lambda x: x)

    # Check network parameters before training
    # check_network_parameters(net)

    # initialize Z_s
    lat_vecs = torch.nn.Embedding(len(train_dataset.idx_dict), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
    torch.nn.init.kaiming_normal_(lat_vecs.weight.data, a=0.02, nonlinearity='leaky_relu')
    if two_shape_codes:
        lat_vecs_ds = torch.nn.Embedding(len(train_dataset.idx_dict), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
        torch.nn.init.kaiming_normal_(lat_vecs_ds.weight.data, a=0.02, nonlinearity='leaky_relu')
        zs_params = list(lat_vecs.parameters())+list(lat_vecs_ds.parameters())
    else:
        zs_params = list(lat_vecs.parameters())
    net = nn.DataParallel(net)
    net.to(device)
   
    # no weight decay for type prediction
    subnets_type = ['decoder.decoder', 'type_encoder']
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
    optimizer_zs = torch.optim.Adam(zs_params, lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)  

    scheduler_nodecay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nodecay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], min_lr=1e-6)
    scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], min_lr=1e-6)
    scheduler_zs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_zs, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], min_lr=1e-6)
    optimizers = [optimizer_nodecay, optimizer_decay, optimizer_zs]
    schedulers = [scheduler_nodecay, scheduler_decay, scheduler_zs]
    
    if os.path.exists(os.path.join(cfg['data']['output_dir'], 'net.pt')) and cfg['resume']:
        print("LOADING LASTEST CHECKPOINT")
        net, optimizers, schedulers, start_epoch = io_utils.load_ckp(os.path.join(cfg['data']['output_dir'], 'net.pt'), \
                net, optimizers, schedulers)
        lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code.pt'))['latent_codes'])
        if two_shape_codes:
            lat_vecs_ds.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code_ds.pt'))['latent_codes'])
    else:
        ## initialize type network - helps with convergence
        if cfg['train']['init']:
            for n in params_else:
                params_dict[n].requires_grad = False
                params_dict[n].grad = None
            net = initialize_type_network(cfg, net, optimizer_nodecay)
            for n in params_else:
                params_dict[n].requires_grad = True
        else:
            pass
    
    fix_type = False
    # Update global state with model components
    global_training_state['net'] = net
    global_training_state['lat_vecs'] = lat_vecs
    global_training_state['lat_vecs_ds'] = lat_vecs_ds if two_shape_codes else None
    global_training_state['optimizers'] = optimizers
    global_training_state['schedulers'] = schedulers
    global_training_state['epoch'] = start_epoch  # Initialize epoch in global state

    # Check network parameters after initialization
    # check_network_parameters(net)

    # start training
    train_model()
