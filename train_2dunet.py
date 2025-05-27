import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import vtk
from vtk_utils.vtk_utils import *
import unet2d_dataset
from torch.utils.data import DataLoader
import yaml
import functools
import pkbar
import matplotlib.pyplot as plt
import io_utils
import argparse
import h5py
import random
from torchinfo import summary
from unet_2d import UNet2D

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def convert_shape(data):
    b, c, h, w, d = data.shape
    data = torch.movedim(data, -1, 0)
    return data.reshape(-1, c, h, w)

def convert_shape_back(data, batch_size):
    n, c, h, w = data.shape
    data = data.reshape(-1, batch_size, c, h, w)
    data = torch.movedim(data, 0, -1)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config') 
    args = parser.parse_args()
    start_epoch = 0

    config_fn = args.config
    with open(config_fn, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])

    # create dataloader
    train = unet2d_dataset.ImgSDFDataset(cfg['data']['train_dir'], cfg['data']['chd_info'], mode=['train'], use_aug=True)
    dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=0)
   
    net = UNet2D(in_channels=1, out_channels=cfg['net']['n_classes'])
    net = nn.DataParallel(net)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['train']['lr'], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg['train']['scheduler']['factor'],
                                                           patience=cfg['train']['scheduler']['patience'], min_lr=1e-6)
    if os.path.exists(os.path.join(cfg['data']['output_dir'], 'net.pt')):
        print("LOADING LASTEST CHECKPOINT")
        net, optimizer, scheduler, start_epoch = io_utils.load_ckp_single(os.path.join(cfg['data']['output_dir'], 'net.pt'), net, optimizer, scheduler)

    best_val_loss = float('inf')
    if cfg['net']['n_classes']>1:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.BCELoss()
    for epoch in range(start_epoch, cfg['train']['epoch']):
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        for i, data in enumerate(dataloader_train):
            img = convert_shape(data['image'].to(device))
            #plt.imshow(img.detach().cpu().numpy()[100, 0, :, :])
            #plt.savefig('test.png')
            if epoch == 0 and i ==0:
                summary(net, tuple(img.shape)) 
            net.zero_grad()
            gt  = data['y'].long().to(device)
            logits = net(img)
            loss = loss_func(convert_shape_back(logits, cfg['train']['batch_size']), gt)
            
            loss.backward()
            optimizer.step()
            kbar.update(i, values=[("loss", loss)])
        with torch.no_grad():
            io_utils.save_ckp_single(net, optimizer, scheduler, epoch, os.path.join(cfg['data']['output_dir']))
            
