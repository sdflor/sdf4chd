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
from cdeepsdf import cDeepSDF,cDeepSDFTester
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
import random
from collections import OrderedDict
from test_gen import get_shape_distribution, get_type_dict

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def fit_testdata(net, cfg, mode, iter_num=200, num_blocks=1, thresh=0.5, output_mode='sdf'):
    STD = 0.01
    PT_NUM = 32768 //2
    fns = sorted(glob.glob(os.path.join(cfg['data']['test_dir'], 'pytorch', '*.pkl')))
    df = dataset.read_excel(cfg['data']['chd_info']['diag_fn'], sheet_name=cfg['data']['chd_info']['diag_sn'])
    fns, chd_data, _ = dataset.parse_data_by_chd_type(fns, df, cfg['data']['chd_info']['types'], cfg['data']['chd_info']['exclude_types'], mode=mode, use_aug=False) 
    print("Test data: ", fns, os.path.join(cfg['data']['test_dir'], 'pytorch', '*.pkl'))
    dice_list = []
    for i, fn in enumerate(fns):
        # freeze for each sample just in case
        net.train()
        for param in net.parameters():
            param.requires_grad = False
        z_s = torch.normal(torch.zeros((1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])), std=STD).to(device)
        z_s.requires_grad=True
       
        # get data
        input_data = pickle.load(open(fn, 'rb'))
        seg_py = np.argmin(input_data, axis=0)+1
        seg_py[np.all(input_data>0.000005, axis=0)] = 0
        filename = os.path.basename(fn).split('.')[0]
        chd_type = torch.from_numpy(chd_data[i, :].astype(np.float32)).unsqueeze(0).to(device)
        z_t = net.type_encoder(chd_type) 
        
        # optimizer
        optimizer = torch.optim.Adam([z_s], lr=0.01, betas=(0.5, 0.999), weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True, min_lr=1e-6)
    
        # optimize
        for k in range(iter_num):
            optimizer.zero_grad()
            _, points, point_values, _ = dataset.sample_points_from_sdf(input_data, PT_NUM, 20)
            points = points.unsqueeze(0).to(device)
            point_values = point_values.unsqueeze(0).to(device)
            outputs = net.decoder(z_t, points, z_s)

            recons_loss = torch.mean(((outputs.permute(0, 2, 1).squeeze(-1) - point_values)**2))
            gaussian_s_loss = torch.mean(z_s**2)
            total_loss = 1.*recons_loss + 0.0001 * gaussian_s_loss
            total_loss.backward()
            print(k, total_loss.item(), recons_loss.item(), gaussian_s_loss.item())
            optimizer.step()
            scheduler.step(total_loss.item())
        with torch.no_grad():
            sdf = tester.z2voxel(z_s, z_t, net)
            io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['test_output_dir'], '{}_test_pred.vtp'.format(filename)), thresh) 
            dice = metrics.dice_score_from_sdf(sdf[1:-1, 1:-1, 1:-1], seg_py, thresh=thresh)
            print("Unseen shape dice: ", dice)
            dice_list.append(dice)
    if len(dice_list)>0:
        io_utils.write_scores(os.path.join(cfg['data']['test_output_dir'], 'dice_test.csv'), dice_list)


def sample_shape_space(net, data, cfg, lat_vecs=None, stats=None, num_copies=10, thresh=0.5):
    import random
    chd_type = data['chd_type'].to(device)
    filename = data['filename']
    z_t = net.type_encoder(chd_type)
    z_s = lat_vecs(data['idx'].to(device)).view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
    dir_n = os.path.join(os.path.dirname(cfg['data']['train_output_dir']), 'aligned_train')
    for i in range(num_copies):
        dx_z_s = torch.normal(torch.from_numpy(stats[0].astype(np.float32)), std=torch.from_numpy(stats[1].astype(np.float32))).to(chd_type.device)
        sdf = tester.z2voxel(dx_z_s, z_t, net)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred_r{}.vtp'.format(filename[0], i)), thresh) 
        io_utils.write_sdf_to_seg(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred_r{}.nii.gz'.format(filename[0], i)), thresh)

def interpolate_type_and_shape(net, data_curr, data_prev, cfg, lat_vecs=None, interval=5,thresh=0.5):
    chd_type_curr = data_curr['chd_type'].to(device)
    filename_curr = data_curr['filename'][0]
    chd_type_prev = data_prev['chd_type'].to(device)
    filename_prev = data_prev['filename'][0]
    type_curr = net.type_encoder(chd_type_curr)
    type_prev = net.type_encoder(chd_type_prev)
    shape_curr = lat_vecs(data_curr['idx'].to(device)).view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
    shape_prev = lat_vecs(data_prev['idx'].to(device)).view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])

    factor = np.linspace(-0.5, 1.5, interval)
    # interpolate type
    for i, f in enumerate(factor):
        z_t = type_prev + (type_curr - type_prev) * f
        z_s = shape_prev 
        sdf = tester.z2voxel(z_s, z_t, net)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_pred_tInterp{}.vtp'.format(filename_prev, filename_curr, i)), thresh) 

    # interpolate shape
    for i, f in enumerate(factor):
        z_t = type_prev
        z_s = shape_prev + (shape_curr - shape_prev) * f
        sdf = tester.z2voxel(z_s, z_t, net)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_to_{}_pred_sInterp{}.vtp'.format(filename_prev, filename_curr, i)), thresh) 


def get_original_prediction(net, z_s, data, thresh=0.5):
    chd_type = data['chd_type'].to(device)
    z_t = net.type_encoder(chd_type)

    z_s_np = np.squeeze(z_s.detach().cpu().numpy().astype(np.float32))
    np.save(os.path.join(cfg['data']['train_output_dir'], '{}_feat.npy'.format(data['filename'][0])), z_s_np)
    
    curr_time = time.time()
    total_time = time.time() - curr_time
    
    sdf = tester.z2voxel(z_s, z_t, net)
    curr_time = time.time()
    
    total_time += time.time() - curr_time
    print("TIME: ", total_time)

    io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(cfg['data']['train_output_dir'], '{}_pred.vtp'.format(data['filename'][0])), thresh) 
    dice = metrics.dice_score_from_sdf(sdf[1:-1, 1:-1, 1:-1], data['y'].numpy()[0], thresh=thresh)
    print("Seen shape dice:", data['filename'][0], dice)
    z_v = z_t.cpu().detach().numpy()
    return z_v, dice, total_time

def test_type_prediction(net, excel_info, thresh=0.5):
    # get the type of all combinations in the dataset
    df = dataset.read_excel(excel_info['diag_fn'], excel_info['diag_sn'])
    p_ids = df.index.tolist()
    all_types = df.columns.tolist()  
    type_ids = [all_types.index(t) for t in excel_info['types']]  
    all_types = [all_types[t] for t in type_ids]
    print("Allowed types:  ", excel_info['types'])
    arr = df.to_numpy()[:, type_ids]
    type_dict = {}
    for r in arr:
        type_name = '_'.join(itertools.compress(all_types, r==1))
        type_name = 'Normal' if type_name == '' else type_name
        if type_name not in type_dict.keys():
            type_dict[type_name] = r
    print("Existing types: ", type_dict.keys())
    # get new combinations of types not in the dataset
    z_s = torch.zeros((1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])).to(device)
    new_type_dict = {}
    for i in range(len(excel_info['types']), 4):
        ts_combs = list(itertools.combinations(excel_info['types'], i+1))
        for ts in ts_combs: 
            type_name = '_'.join(ts)
            type_name = 'Normal' if type_name == '' else type_name
            if (type_name not in type_dict.keys()) and (type_name not in new_type_dict.keys()):
                type_values = np.zeros(len(excel_info['types']))
                for t in ts:
                    type_values[ts.index(t)] = 1.
                new_type_dict[type_name] = type_values
    print("New types: ", new_type_dict.keys())
    for k in type_dict.keys():
        z_t = net.type_encoder(torch.from_numpy(np.array([type_dict[k]]).astype(np.float32)).to(device))
        type_sdf = tester.z2voxel(z_s, z_t, net)
        io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['train_output_dir'], '{}_exist_type.vtp'.format(k)), thresh)
    '''
    for k in new_type_dict.keys():
        z_t = net.type_encoder(torch.from_numpy(np.array([new_type_dict[k]]).astype(np.float32)).to(device))
        type_sdf = tester.z2voxel(None, z_t, net, num_blocks=num_block, out_type=True)
        io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['train_output_dir'], '{}_noexist_type.vtp'.format(k)), 0.5)
    '''
    # try adding in the latent space
    # normal + vsd
    z_t = net.type_encoder(torch.from_numpy(np.array([type_dict['Normal']]).astype(np.float32)).to(device)) + \
            net.type_encoder(torch.from_numpy(np.array([type_dict['VSD']]).astype(np.float32)).to(device))
    type_sdf = tester.z2voxel(z_s, z_t, net)
    io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['train_output_dir'], 'Normal+VSD_type.vtp'), 0.5)
    # vsd - normal 
    z_t = net.type_encoder(torch.from_numpy(np.array([type_dict['VSD']]).astype(np.float32)).to(device)) - \
            net.type_encoder(torch.from_numpy(np.array([type_dict['Normal']]).astype(np.float32)).to(device))
    type_sdf = tester.z2voxel(z_s, z_t, net)
    io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['train_output_dir'], 'VSD-Normal_type.vtp'), 0.5)

def random_type_generation(cfg, net, num_gen, thresh=0.5):
    out_dir = os.path.join(cfg['data']['train_output_dir'], 'random_type_gen')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    type_dict = get_type_dict(cfg['data']['chd_info'], get_new_type=False)
    stats = get_shape_distribution(cfg)
    keys = list(type_dict.keys())
    type_list = []
    for i in range(num_gen):
        k_id = random.choice(keys)
        type_i = type_dict[k_id]
        type_list.append(type_i)
        z_t = net.type_encoder(torch.from_numpy(np.array([type_i]).astype(np.float32)).to(device))
        z_s = torch.normal(torch.from_numpy(stats[0].astype(np.float32)), std=torch.from_numpy(stats[1].astype(np.float32))).to(z_t.device)
        sdf = tester.z2voxel(z_s, z_t, net)
        io_utils.write_sdf_to_vtk_mesh(sdf, os.path.join(out_dir, 'gen_{}.vtp'.format(i)), thresh)
    io_utils.write_scores(os.path.join(out_dir, 'type_gt.csv'), type_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--config')
    parser.add_argument('--grid_size', type=int, default=2)
    args = parser.parse_args()
    test_ops = {
            'type_pred': False,
            'type_interp': False,
            'train_dice': False,
            'test_dice': False,
            'shape_type_interp': False,
            'shape_gen': False, 
            'rand_type_gen': True
            }                 

    THRESH = 0.5
    MODE = ['train']
    num_block=1
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])
    
    tester = cDeepSDFTester(device, cell_grid_size=args.grid_size, out_dim=cfg['net']['out_dim'])
    dice_score_list, time_list = [], []
    z_vector_list = {}

    # TRAINING ACCURACY
    train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['output_dir'], chd_info=cfg['data']['chd_info'], mode=MODE, use_aug=False)
    dataloader_test = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True)
    # create network and latent codes
    net = cDeepSDF(out_dim=cfg['net']['out_dim'], \
            num_types=len(cfg['data']['chd_info']['types']), \
            z_t_dim=cfg['net']['z_t_dim'], \
            z_s_dim=cfg['net']['z_s_dim'], \
            type_mlp_num=cfg['net']['type_mlp_num'],\
            latent_dim=cfg['net']['latent_dim'], \
            ins_norm=cfg['net']['ins_norm'], \
            type_bias=False, \
            lip_reg=cfg['net']['lip_reg'])
        # initialize Z_s
    lat_vecs = torch.nn.Embedding(len(train), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
    #lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['latent_codes'])
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'net_{}.pt'.format(args.epoch)), map_location=torch.device('cpu'))['state_dict'])

    cfg['data']['train_output_dir'] = os.path.join(cfg['data']['output_dir'], MODE[0] + '_{}'.format(args.epoch))
    if not os.path.exists(cfg['data']['train_output_dir']):
        os.makedirs(cfg['data']['train_output_dir'])
    # TEST 1: TYPE PREDICTION
    if test_ops['type_pred']:
        test_type_prediction(net, cfg['data']['chd_info'], thresh=0.5)
    # TEST 2: WH DICE ACCURACY ON SEEN SHAPES
    if test_ops['train_dice']:
        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                original_copy = bool(re.match("ct_[a-z]+_\d+",data['filename'][0])) or bool(re.match("ct_\d+_image", data['filename'][0]))
                print(i, data['filename'][0], original_copy)
                if original_copy:
                    z_s = lat_vecs(data['idx'].to(device)).view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
                    z_vector, dice, total_time = get_original_prediction(net, z_s, data, thresh=THRESH)
                    dice_score_list.append(dice)
                    time_list.append(total_time)
            print("TIME AVG: ", np.mean(np.array(total_time)))
        io_utils.write_scores(os.path.join(cfg['data']['train_output_dir'], 'dice.csv'), dice_score_list)
        print(dice_score_list)
    # TEST 3: WH DICE ACCURACY ON UNSEEN SHAPES
    if test_ops['test_dice']:
        MODE = ['test']
        cfg['data']['test_output_dir'] = os.path.join(cfg['data']['output_dir'], MODE[0]+ '_lat{}_opt_{}'.format(cfg['net']['l_dim'], args.epoch))
        if not os.path.exists(cfg['data']['test_output_dir']):
            os.makedirs(cfg['data']['test_output_dir'])
        fit_testdata(net, cfg, mode=MODE, iter_num=200)
    # TEST 4: TYPE AND SHAPE INTERPOLATION
    if test_ops['shape_type_interp']:
        MODE = ['train']
        #selected_list = ['ct_1017_image', 'ct_1077_image', 'ct_1037_image', 'ct_1098_image', 'ct_1043_image', 'ct_1042_image']
        selected_list = ['ct_1101_image', 'ct_1042_image']
        selected_list = ['ct_1098_image', 'ct_1037_image', 'ct_1017_image']
        selected_list = ['ct_1101_image', 'ct_1112_image']
        selected_list = ['ct_1101_image', 'ct_1138_image']
        selected_list = ['ct_1017_image', 'ct_1012_image']

        data_dict = OrderedDict()
        for i, data in enumerate(dataloader_test):
            original_copy = bool(re.match("ct_[a-z]+_\d+",data['filename'][0])) or bool(re.match("ct_\d+_image", data['filename'][0]))
                #if original_copy and data['filename'][0] in selected_list:
            if original_copy and data['filename'][0] in selected_list:
                data_dict[data['filename'][0]] = data
        with torch.no_grad():
            data_prev = None
            for key in selected_list:
                data = data_dict[key]
                if data_prev is not None:
                    interpolate_type_and_shape(net, data, data_prev, cfg, lat_vecs=lat_vecs,interval=21,thresh=THRESH) 
                data_prev = data
    
    # TEST 5: SHAPE GENERATION
    if test_ops['shape_gen']:
        stats = get_shape_distribution(cfg)
        with torch.no_grad():
            for i, (key, data) in enumerate(data_dict.items()):
                sample_shape_space(net, data, cfg, lat_vecs=lat_vecs, stats=stats, num_copies=20, thresh=THRESH)
    if test_ops['rand_type_gen']:
        random_type_generation(cfg, net, num_gen=50)

