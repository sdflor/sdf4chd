# From https://github.com/qiminchen/UNIST/blob/main/modelAE.py
import os
import math
import numpy as np
import mcubes

import torch
import torch.nn as nn
import torch.nn.functional as F
from io_utils import *

def act(x):
    x = torch.max(torch.min(x, x * 0.05 + 0.99), x * 0.05)
    return x

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

class TYPEDecoder2(torch.nn.Module):
    def __init__(self, z_dim, df_dim, res_block_num=4, out_dim=1, point_dim=3):
        super(TYPEDecoder2, self).__init__()
        self.res_block_num = res_block_num
        self.linear_1 = nn.Linear(z_dim+point_dim, df_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        for i in range(res_block_num):
            lin = nn.Linear(df_dim, df_dim, bias=True)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)
            setattr(self, 'lin_{}'.format(i), lin)
        self.linear_out = nn.Linear(df_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)

    def forward(self, inputs):
        x = self.linear_1(inputs)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=True)

        for i in range(self.res_block_num):
            x = getattr(self, 'lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True) 
            #print("X2: ", torch.max(x))
        x = self.linear_out(x)
        return x

class SHAPEDecoder(nn.Module):
    def __init__(self, im_dim_dx, type_dim_dx, df_dim, num_layers, type_decoder):
        super(SHAPEDecoder, self).__init__()
        self.fc1_dx = nn.Linear(im_dim_dx+130, df_dim, bias=True)
        for i in range(num_layers):
            setattr(self, 'shape_mlp_{}'.format(i), nn.Linear(df_dim, df_dim, bias=True))
        self.fc_out_dx = nn.Linear(df_dim, 3, bias=True)
        self.num_layers = num_layers
        self.type_decoder = type_decoder

    def deform(self, points, points_type, grid_list):
        points_s = (points).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
        points_z_list = []
        for grid in grid_list:
            points_z = F.grid_sample(grid, points_s, padding_mode='border', align_corners=True)  # (B, 32, 1, 1, 4096)
            points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)
            points_z_list.append(points_z)
        
        points_z_total = torch.cat(points_z_list, dim=-1)
        points_sdf = self.type_decoder(points_type, points)
        points_f = torch.cat([positional_encoding(torch.cat([points, points_sdf], dim=-1)), points_z_total], dim=-1)
        points_f = F.leaky_relu(self.fc1_dx(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, 'shape_mlp_{}'.format(i))
            points_f = F.leaky_relu(lin(points_f), 0.2)
        dx = self.fc_out_dx(points_f)
        return dx

    def flow(self, V, x, x_type, solver='euler', step_size=0.2, T=1.0, block=0, inverse=False):
            
        h = step_size
        N = int(T/h)
        #print("N: ", N, solver)
        
        fac = -1. if inverse else 1.
        #print("FAC: ", fac)
        magnitude = 0.
        if solver == 'euler':
            # forward Euler method
            for n in range(N):
                dx = self.deform(x, x_type, V)
                x = x + fac * h * dx
                magnitude += torch.mean(dx**2) 
        if solver == 'midpoint':
            # midpoint method
            for n in range(N):
                dx1 = self.deform(x, x_type, V)
                dx2 = self.deform(x + fac*h*dx1/2, x_type, V)
                x = x + fac * h * dx2
                magnitude += torch.mean(dx2**2)
        if solver == 'heun':
            # Heun's method
            for n in range(N):
                dx1 = self.deform(x, x_type, V)
                dx2 = self.deform(x + fac*h*dx1, x_type, V)
                x = x + fac * h * (dx1 + dx2) / 2
                magnitude += (torch.mean(dx1**2) + torch.mean(dx2**2))/2.
        if solver == 'rk4':
            # fourth-order RK method
            for n in range(N):
                dx1 = self.deform(x, x_type, V)
                dx2 = self.deform(x + fac*h*dx1/2, x_type, V)
                dx3 = self.deform(x + fac*h*dx2/2, x_type, V)
                dx4 = self.deform(x + fac*h*dx3, x_type, V)
                x = x + fac * h * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6
                magnitude += (torch.mean(dx1**2)+2.*torch.mean(dx3**2)+2.*torch.mean(dx3**2)+ torch.mean(dx1**2))/6.
        return x, magnitude 

    def forward(self, points, type_vec, shape_grid, inverse=False, step_size=0.2):
        # Flow from ground truth locations to topology space
        points_o, magnitude = self.flow(shape_grid, points, type_vec, step_size=step_size, inverse=inverse)
        return points_o, magnitude

class CorrectionDecoder(nn.Module):
    def __init__(self, z_dim_ds, df_dim, num_layers):
        super(CorrectionDecoder, self).__init__()
        self.fc1_ds = nn.Linear(z_dim_ds+130, df_dim, bias=True)
        #print("CorrectionDecoder: ", z_dim_ds+52)
        for i in range(num_layers):
            setattr(self, 'ds_mlp_{}'.format(i), nn.Linear(df_dim, df_dim, bias=True))
        self.fc_out_ds = nn.Linear(df_dim, 1, bias=True)
        self.num_layers = num_layers

    def forward(self, points, grid_list):
        points_s = (points[:, :, :3]).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
        points_z_list = []
        for grid in grid_list[-3:]:
            points_z = F.grid_sample(grid, points_s, padding_mode='border', align_corners=True)  # (B, 32, 1, 1, 4096)
            points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)
            points_z_list.append(points_z)
        points_z_total = torch.cat(points_z_list, dim=-1)

        points_f = torch.cat([positional_encoding(points), points_z_total], dim=-1)
        #print("CorrectionDecoder FORWARD: ", points_f.shape, points.shape, positional_encoding(points).shape, points_z_total.shape)
        points_f = F.leaky_relu(self.fc1_ds(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, 'ds_mlp_{}'.format(i))
            points_f = F.leaky_relu(lin(points_f), 0.2)
        ds = self.fc_out_ds(points_f)
        return ds

class TypeEncoder(nn.Module):
    def __init__(self, in_dim, df_dim, res_block_num=4, out_dim=1):
        super(TypeEncoder, self).__init__()
        self.res_block_num = res_block_num
        self.linear_1 = nn.Linear(in_dim, df_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        for i in range(res_block_num):
            lin = nn.Linear(df_dim, df_dim, bias=True)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)
            setattr(self, 'lin_{}'.format(i), lin)
        self.linear_out = nn.Linear(df_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)

    def forward(self, z):
        # input z - B x 4096 x z_dim
        x = self.linear_1(z)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

        for i in range(self.res_block_num):
            x = getattr(self, 'lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True) 
            #print("X2: ", torch.max(x))
        x = self.linear_out(x)
        return x

class IsenseeContextModule(nn.Module):
    def __init__(self, in_channels, num_filters, stride=1, dropout_rate=0.3):

        super(IsenseeContextModule, self).__init__()

        self.conv3d_0 = self.conv_norm_lrelu(in_channels, num_filters, stride=stride)
        self.conv3d_1 = self.conv_norm_lrelu(num_filters, num_filters, stride=1)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)
        self.conv3d_2 = self.conv_norm_lrelu(num_filters, num_filters, stride=1)

    def conv_norm_lrelu(self, num_feat_in, num_feat_out, stride):
        return nn.Sequential(
            nn.Conv3d(num_feat_in, num_feat_out, kernel_size=3, stride=stride, padding='same' if stride==1 else (1,1,1)),
            nn.InstanceNorm3d(num_feat_out),
            nn.LeakyReLU())

    def forward(self, input):
        conv0 = self.conv3d_0(input)
        conv1 = self.dropout3d(self.conv3d_1(conv0))
        conv2 = (self.conv3d_2(conv1) + conv0)/2.
        return conv2

class Isensee3DUNetEncoder(nn.Module):
    def __init__(self, in_channels, base_n_filter = 16, z_dim=32, n_conv_blocks=4):
        super(Isensee3DUNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.base_n_filter = base_n_filter
        self.z_dim = z_dim
        self.n_conv_blocks = n_conv_blocks

        in_filter = self.in_channels
        for i in range(n_conv_blocks):
            out_filter = self.base_n_filter * 2**(i)
            setattr(self, 'conv_block{}'.format(i), IsenseeContextModule(in_filter, out_filter, stride=1 if i==0 else 2))
            setattr(self, 'conv1_block{}'.format(i), nn.Conv3d(out_filter, z_dim, 3, stride=1, padding=1))
            setattr(self, 'conv2_block{}'.format(i), nn.Conv3d(z_dim, z_dim, 3, stride=1, padding=1))
            in_filter = out_filter

        self.conv0 = nn.Conv3d(out_filter, z_dim, 3, stride=1, padding=1)  # (B, 32, 2, 2, 2)
        self.conv1 = nn.Conv3d(z_dim, z_dim, 3, stride=1, padding=1)

    def forward(self, input):
        feat_list = []
        x = input
        for i in range(self.n_conv_blocks):
            x = getattr(self, 'conv_block{}'.format(i))(x)
            #x_out = F.leaky_relu(getattr(self, 'conv1_block{}'.format(i))(x), negative_slope=0.02, inplace=True)
            #x_out = getattr(self, 'conv2_block{}'.format(i))(x_out)
            feat_list.append(x)
        feat_list.reverse()
        return feat_list
        #return feat_list[-3:]


class GridDecoder3D(nn.Module):
    def __init__(self, z_dim=32, df_dim=128, res_block_num=4, out_dim=1):
        super(GridDecoder3D, self).__init__()
        self.decoder = TYPEDecoder2(z_dim=z_dim, df_dim=df_dim, res_block_num=res_block_num, out_dim=out_dim, point_dim=39)

    def forward(self, z, points):
        point_z = torch.cat([positional_encoding(points), torch.tile(z.unsqueeze(1), (1, points.shape[1], 1))], axis=-1)
        sdf = self.decoder(point_z)

        return sdf

class DisentangledGridDecoder3D(nn.Module):
    def __init__(self, z_dim=32, df_dim=128, res_block_num=4, out_dim=1, n_conv_blocks=5, n_flow_blocks=2, n_smpl_pts_f=32768):
        super(DisentangledGridDecoder3D, self).__init__()
        self.decoder = GridDecoder3D(z_dim=z_dim, df_dim=df_dim, res_block_num=res_block_num, out_dim=out_dim)
        self.block_ids = [[0], [1,2,3]]
        self.im_dim = [256, 128+64+32]
        for i in range(n_flow_blocks):
            setattr(self, "shape_decoder_{}".format(i), SHAPEDecoder(self.im_dim[i], z_dim, df_dim, 3, type_decoder=self.decoder))
        self.ds_decoder = CorrectionDecoder(64+32+16, df_dim, 3) 
        self.n_flow_blocks = n_flow_blocks
        self.n_smpl_pts_f = n_smpl_pts_f
    
    def sample_points_in_type_space(self, z_t, thresh=0.5, size=48):
        batch = z_t.shape[0]
        x, y, z = torch.meshgrid([torch.arange(size), torch.arange(size), torch.arange(size)]) 
        x = 1.5 * (torch.normal(x.flatten().float(), 0.3)/float(size) - 0.5)
        y = 1.5 * (torch.normal(y.flatten().float(), 0.3)/float(size) - 0.5)
        z = 1.5 * (torch.normal(z.flatten().float(), 0.3)/float(size) - 0.5)
        points = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
        points = torch.tile(points.unsqueeze(0), (batch, 1, 1)).to(z_t.device)
        points_sdv = self.decoder(z_t, points)
        #print("points_sdv: ", points_sdv.shape)
        hit_indices = torch.any(torch.abs(points_sdv - thresh) < 0.3, dim=-1)
        #print("HIT_INDICES: ", hit_indices.shape)
        points_hit = points[hit_indices, :]
        if points_hit.shape[0] > 1.1 * self.n_smpl_pts_f:
            indices = torch.randperm(points_hit.shape[0])
            points_hit = points_hit[indices[:self.n_smpl_pts_f], :]
        elif points_hit.shape[0] < self.n_smpl_pts_f:
            indices = torch.randperm(size*size*size)
            points_hit = torch.cat([points[0, indices[:(self.n_smpl_pts_f-points_hit.shape[0])], :], points_hit], dim=0)
        else:
            points_hit = points_hit[self.n_smpl_pts_f, :]
        points_hit = torch.tile(points_hit.unsqueeze(0), (batch, 1, 1))
        points_hit_sdv = self.decoder(z_t, points_hit)
        #print("SHAPE CHECK: ", points_hit.shape, points_hit_sdv.shape)
        return points_hit, points_hit_sdv

    def forward_prediction(self, z_s, z_t, gt_binary=None, get_tmplt_coords=False):

        points_hit, points_hit_sdv = self.sample_points_in_type_space(z_t)
        points_s = points_hit
        gt_sdv_list = []
        if get_tmplt_coords:
            saved_point_list = [points_s]

        for i in range(self.n_flow_blocks):
            flow_layer = getattr(self, "shape_decoder_{}".format(i))
            im_feat_i = [z_s[q] for q in self.block_ids[i]]
            points_s, magnitude = flow_layer(points_s, z_t, im_feat_i, inverse=False)
            if self.training:
                points_s_gt_sdv = F.grid_sample(gt_binary, points_s.unsqueeze(1).unsqueeze(1), padding_mode='border', align_corners=True)
                points_s_gt_sdv = points_s_gt_sdv.squeeze(2).squeeze(2).permute(0, 2, 1)
                gt_sdv_list.append(points_s_gt_sdv)
            if get_tmplt_coords:
                saved_point_list.append(points_s)
        
        if get_tmplt_coords and not self.training:
            return points_s, saved_point_list
        else:
            return points_s, [points_hit_sdv], gt_sdv_list

    def backward_prediction(self, z_s, z_t, points, get_tmplt_coords=False):
        output_sdv, output_flow_mag = [], []
        points_t = points
        if get_tmplt_coords:
            saved_point_list = []

        for i in range(self.n_flow_blocks-1, -1, -1):
            flow_layer = getattr(self, "shape_decoder_{}".format(i))
            im_feat_i = [z_s[q] for q in self.block_ids[i]]
            points_t, magnitude = flow_layer(points_t, z_t, im_feat_i, inverse=True)
            if get_tmplt_coords:
                saved_point_list.append(points_t)
            points_t_sdv = self.decoder(z_t, points_t)
            output_sdv.append(points_t_sdv)
            output_flow_mag.append(torch.mean(magnitude**2))
            if self.training and i==0:
                # THIS IS BASICALLY SAYING THAT POINTS_T + 0.01 = FLOW(POINTS_S + 0.01)
                grad_t_approx = (self.decoder(z_t, points_t+0.01) - points_t_sdv)/0.01
                grad_s_approx = (self.decoder(z_t, flow_layer(points+0.01, z_t, im_feat_i, inverse=True)[0]) - points_t_sdv)/0.01
        if self.training:
            return output_sdv, (grad_t_approx, grad_s_approx), output_flow_mag
        else:
            if get_tmplt_coords:
                return output_sdv, saved_point_list
            else:
                return (output_sdv,) 

    def forward(self, z_s, z_t, points, gt_binary=None, get_tmplt_coords=False, add_correction=True):
        
        # backward pass first
        # the first sdv should be really close to gt
        backward_output = self.backward_prediction(z_s, z_t, points, get_tmplt_coords)
        # then forward pass  #points_s, points_hit_sdv, gt_sdv_list
        forward_output = self.forward_prediction(z_s, z_t, gt_binary=gt_binary, get_tmplt_coords=get_tmplt_coords)

        # get the sdv at the point locations
        if add_correction:
            ds_b = self.ds_decoder(torch.cat([points, backward_output[0][-1]], dim=-1), z_s)
            points_t_sdv_ds_b = backward_output[0][-1] + ds_b
            backward_output[0].append(points_t_sdv_ds_b)

            if self.training:
                ds_f = self.ds_decoder(torch.cat([forward_output[0], forward_output[1][0]], dim=-1), z_s)
                points_t_sdv_ds_f = forward_output[1][0] + ds_f
                forward_output[1].append(points_t_sdv_ds_f)

        if self.training:
            return {'back_recons': [act(o).permute(0, 2, 1) for o in backward_output[0][-2:]],\
                    'forward_recons_w_gt': ([act(o) for o in forward_output[1]], forward_output[-1]), \
                    'flow_mag': backward_output[-1], \
                    'grad': backward_output[1]
                    }
        else:
            if get_tmplt_coords:
                return [act(o) for o in backward_output[0]], forward_output[-1] 
            else:
                return [act(o) for o in backward_output[0]]


class DisentangledConditionalGridAE3D(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, num_types=1, ef_dim=16, z_dim=32, df_dim=128, res_block_num=6, n_conv_blocks=5, im_size=128, n_smpl_pts_f=32768):
        super(DisentangledConditionalGridAE3D, self).__init__()
        self.encoder = Isensee3DUNetEncoder(in_channels=in_dim, base_n_filter=ef_dim, z_dim=z_dim, n_conv_blocks=n_conv_blocks)
        self.decoder = DisentangledGridDecoder3D(z_dim=z_dim, df_dim=df_dim, res_block_num=res_block_num, out_dim=out_dim, n_conv_blocks=n_conv_blocks, n_smpl_pts_f=n_smpl_pts_f)
        self.type_encoder = TypeEncoder(in_dim=num_types, df_dim=df_dim, res_block_num=res_block_num, out_dim=z_dim)
        self.n_smpl_pts_f = n_smpl_pts_f

    def forward(self,img, points, chd_type, gt_binary=None, get_tmplt_coords=False):
        z_s_list = self.encoder(img)
        z_t = self.type_encoder(chd_type)
        if self.training:
            outputs = self.decoder(z_s_list, z_t, points, gt_binary)
            return outputs, z_t
        else:
            if get_tmplt_coords:
                points_t_sdv_list, points_t_list = self.decoder(z_s_list, z_t, points, gt_binary)
                return points_t_sdv_list, points_t_list
            else:
                points_t_sdv_list = self.decoder(z_s_list, z_t, points, gt_binary)
                return points_t_sdv_list

class Tester:
    def __init__(self, device, cell_grid_size=4, frame_grid_size=64, out_dim=1):
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
        self.out_dim=out_dim
        self.cell_grid_size = cell_grid_size
        self.frame_grid_size = frame_grid_size
        self.real_size = self.cell_grid_size * self.frame_grid_size  # =256, output point-value voxel grid size in testing
        self.test_point_batch_size = self.test_size * self.test_size * self.test_size  # 32 x 32 x 32, do not change
        self.sampling_threshold = 0.5
        self.device = device

        self.get_test_coord_for_training()  # initialize self.coords
        self.get_test_coord_for_testing()  # initialize self.frame_coords

    def get_test_coord_for_training(self):
        dima = self.test_size  # 32
        dim = self.frame_grid_size  # 64
        multiplier = int(dim / dima)  # 2
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        ranges = np.arange(0, dim, multiplier, np.uint8)
        self.aux_x = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(1, 2))
        self.aux_y = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(0, 2))
        self.aux_z = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(0, 1))
        self.coords = np.zeros([multiplier ** 3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = self.aux_x + i
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = self.aux_y + j
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = self.aux_z + k
        self.coords = 2.*((self.coords.astype(np.float32) + 0.5) / dim - 0.5)
        self.coords = np.reshape(self.coords, [multiplier3, self.test_point_batch_size, 3])
        self.coords = torch.from_numpy(self.coords).to(self.device)

    def get_test_coord_for_testing(self):
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_coords = np.zeros([dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
        self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
        self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i, j, k] = i
                    self.cell_y[i, j, k] = j
                    self.cell_z[i, j, k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x + i * dimc
                    self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y + j * dimc
                    self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z + k * dimc
                    self.frame_coords[i, j, k, 0] = i
                    self.frame_coords[i, j, k, 1] = j
                    self.frame_coords[i, j, k, 2] = k
                    self.frame_x[i, j, k] = i
                    self.frame_y[i, j, k] = j
                    self.frame_z[i, j, k] = k

        self.cell_coords = 2.*((self.cell_coords.astype(np.float32) + 0.5) / self.real_size - 0.5)
        self.cell_coords = np.reshape(self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3])
        self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
        self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
        self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
        self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
        self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
        self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
        self.frame_coords = 2.*((self.frame_coords.astype(np.float32) + 0.5) / dimf - 0.5)
        self.frame_coords = np.reshape(self.frame_coords, [dimf * dimf * dimf, 3])
        self.frame_coords = torch.from_numpy(self.frame_coords).to(self.device)
    
    def test_during_train(self, network, batch_voxels, batch_types, name, save_data_id, save_epoch_id):
        network.eval()
        model_float = np.zeros([self.frame_grid_size + 2, self.frame_grid_size + 2, self.frame_grid_size + 2, self.out_dim], np.float32)
        multiplier = int(self.frame_grid_size / self.test_size)
        multiplier2 = multiplier * multiplier
        with torch.no_grad():
            zs_vector = network.encoder(batch_voxels)
            zts_vector = network.type_encoder(batch_types)

        for idx, zt_vector in enumerate(zts_vector):
            z_s_list = [grid[idx].unsqueeze(0) for grid in zs_vector]
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        with torch.no_grad():
                            net_out = network.decoder(z_s_list, zt_vector.unsqueeze(0), point_coord)
                        model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1, :] = np.reshape(net_out.detach().cpu().numpy(),
                                                                                                             [self.test_size,
                                                                                                              self.test_size,
                                                                                                              self.test_size, 
                                                                                                              self.out_dim])
            write_vtk_image(model_float, name + '-sdf{}_{}_epoch{}.vti'.format(idx, save_data_id[idx], save_epoch_id))

    def z2voxel(self, z_s, z_t, network, num_blocks, out_block=-1, out_type=False, get_tmplt_coords=False, add_correction=True, gt_data=None):
        # NEED TO CHANGE FOR TYPE ENCODER
        network.eval()
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2, self.out_dim], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, self.out_dim], np.uint8)
        queue = []
        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        out_point_coords_list = [torch.zeros((0, 3)).to(z_t.device) for i in range(num_blocks+1)]
        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                if out_type:
                    model_out_ = act(network.decoder.decoder(z_t, point_coord))
                else:
                    if get_tmplt_coords:
                        model_out_, points_t_list = network.decoder(z_s, z_t, point_coord, get_tmplt_coords=get_tmplt_coords, add_correction=add_correction)
                        for q, (points_t, sdf) in enumerate(zip(points_t_list, model_out_)):
                            """
                            if gt_data is not None:
                                points_s = point_coord.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
                                points_s_sdf = F.grid_sample(gt_data, points_s, padding_mode='border', align_corners=True)
                                points_s_sdf = points_s_sdf.squeeze(2).squeeze(2).permute(0, 2, 1)
                                out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(points_s_sdf>self.sampling_threshold, dim=-1)]], dim=0)
                            else:
                                out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(model_out_[-2]>self.sampling_threshold, dim=-1)]], dim=0)
                            """
                            out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t.squeeze(0)], dim=0)
                    else:
                        model_out_ = network.decoder(z_s, z_t, point_coord, add_correction=add_correction)
                    model_out_ = model_out_[out_block]
                #print(model_out_.shape, self.test_point_batch_size * self.out_dim)
                model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1, :] = np.reshape((model_out > self.sampling_threshold).astype(np.uint8),
                                                                              [self.test_point_batch_size, self.out_dim])

        # get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = 1.0

        # print("running queue:", len(queue))
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0
        # run queue
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            with torch.no_grad():
                if out_type:
                    model_out_batch_ = act(network.decoder.decoder(z_t, cell_coords))
                else:
                    if get_tmplt_coords:
                        model_out_batch_, points_t_list = network.decoder(z_s, z_t, cell_coords, get_tmplt_coords=get_tmplt_coords, add_correction=add_correction)
                        for q, (points_t, sdf) in enumerate(zip(points_t_list, model_out_batch_)):
                            """
                            if gt_data is not None:
                                points_s = cell_coords.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
                                points_s_sdf = F.grid_sample(gt_data, points_s, padding_mode='border', align_corners=True)
                                points_s_sdf = points_s_sdf.squeeze(2).squeeze(2).permute(0, 2, 1)
                                out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(points_s_sdf>self.sampling_threshold, dim=-1)]], dim=0)
                            else:
                                out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(model_out_batch_[-2]>self.sampling_threshold, dim=-1)]], dim=0)
                            """
                            out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t.squeeze(0)], dim=0)
                    else:
                        model_out_batch_ = network.decoder(z_s, z_t, cell_coords, add_correction=add_correction)
                    model_out_batch_ = model_out_batch_[out_block]
                model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, :]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf:
                            continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf:
                                continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf:
                                    continue
                                if frame_flag[pi, pj, pk].all() == 0:
                                    frame_flag[pi, pj, pk, :] = 1
                                    queue.append((pi, pj, pk))
        if get_tmplt_coords:
            return model_float, out_point_coords_list
        else:
            return model_float
