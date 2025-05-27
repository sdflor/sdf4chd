import os
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from io_utils import *


class TYPEDecoder(torch.nn.Module):
    def __init__(self, df_dim, mlp_num=6, out_dim=1, point_dim=3, bias=True, lip_reg=True):
        super(TYPEDecoder, self).__init__()
        self.mlp_num = mlp_num 
        if lip_reg:
            print("TypeDecoder: using lip reg")
            self.linear_1 = LipLinearLayer(point_dim, df_dim, bias=True)
        else:
            self.linear_1 = nn.Linear(point_dim, df_dim, bias=True)
        for i in range(mlp_num):
            if lip_reg:
                lin = LipLinearLayer(df_dim, df_dim, bias=True)
            else:
                lin = nn.Linear(df_dim, df_dim, bias=True)
            setattr(self, 'lin_{}'.format(i), lin)
        if lip_reg:
            self.linear_out = LipLinearLayer(df_dim, out_dim, bias=True)
        else:
            self.linear_out = nn.Linear(df_dim, out_dim, bias=True)

    def forward(self, points):

        point_z = positional_encoding(points, num_encoding_functions=6)
        
        x = self.linear_1(point_z)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=True)

        for i in range(self.mlp_num):
            x = getattr(self, 'lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True) 
        x = self.linear_out(x)
        return x


class SHAPEDecoder(nn.Module):
    def __init__(self, z_s_dim, df_dim, num_layers, type_decoder, ins_norm=False, out_dim=1):
        super(SHAPEDecoder, self).__init__()
        self.fc1_dx = nn.Linear(z_s_dim, df_dim, bias=True)
        for i in range(num_layers):
            setattr(self, 'shape_mlp_{}'.format(i), nn.Linear(df_dim, df_dim, bias=True))
        self.fc_out_dx = nn.Linear(df_dim, 3, bias=True)
        self.num_layers = num_layers
        if ins_norm:
            self.norm = torch.nn.InstanceNorm3d(num_features=z_s_dim)
        self.ins_norm = ins_norm
        self.type_decoder = type_decoder

    def deform(self, points, grid):
        points_s = (points).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
        points_z = F.grid_sample(grid, points_s, padding_mode='border', align_corners=True)  # (B, 32, 1, 1, 4096)
        points_z = points_z.squeeze(2).squeeze(2).permute(0, 2, 1)

        points_f = positional_encoding(points, num_encoding_functions=4) * points_z
        points_f = F.leaky_relu(self.fc1_dx(points_f), 0.2)
        for i in range(self.num_layers):
            lin = getattr(self, 'shape_mlp_{}'.format(i))
            points_f = F.leaky_relu(lin(points_f), 0.2)
        dx = self.fc_out_dx(points_f)
        return dx

    def flow(self, V, x, solver='euler', step_size=0.25, T=1.0, block=0, inverse=False):
            
        h = step_size
        N = int(T/h)
        
        fac = -1. if inverse else 1.
        magnitude = 0.
        if solver == 'euler':
            # forward Euler method
            for n in range(N):
                dx = self.deform(x, V)
                x = x + fac * h * dx
                if self.training:
                   dx_de = self.deform(x + 0.0075, V)
                   magnitude += torch.mean((dx-dx_de)**2) 
        if solver == 'midpoint':
            # midpoint method
            for n in range(N):
                dx1 = self.deform(x, V)
                dx2 = self.deform(x + fac*h*dx1/2, V)
                x = x + fac * h * dx2
                magnitude += torch.mean(dx2**2)
        if solver == 'heun':
            # Heun's method
            for n in range(N):
                dx1 = self.deform(x, V)
                dx2 = self.deform(x + fac*h*dx1, V)
                x = x + fac * h * (dx1 + dx2) / 2
                magnitude += (torch.mean(dx1**2) + torch.mean(dx2**2))/2.
        if solver == 'rk4':
            # fourth-order RK method
            for n in range(N):
                dx1 = self.deform(x, V)
                dx2 = self.deform(x + fac*h*dx1/2, V)
                dx3 = self.deform(x + fac*h*dx2/2, V)
                dx4 = self.deform(x + fac*h*dx3, V)
                x = x + fac * h * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6
                magnitude += (torch.mean(dx1**2)+2.*torch.mean(dx3**2)+2.*torch.mean(dx3**2)+ torch.mean(dx1**2))/6.
        return x, magnitude 

    def forward(self, points, shape_grid, inverse=False, step_size=0.2):
        # Flow from ground truth locations to topology space
        if self.ins_norm:
            shape_grid = self.norm(shape_grid)
        points_o, magnitude = self.flow(shape_grid, points, step_size=step_size, inverse=inverse)
        return points_o, magnitude

class DisentangledGridDecoder3D(nn.Module):
    def __init__(self,\
            z_s_dim=128, \
            df_dim=512, \
            type_mlp_num=6, \
            dx_mlp_num=6, \
            out_dim=1, \
            ins_norm=False, \
            type_bias=True, \
            lip_reg=True):
        super(DisentangledGridDecoder3D, self).__init__()
        self.decoder = TYPEDecoder(df_dim=df_dim, mlp_num=type_mlp_num, out_dim=out_dim, point_dim=39, bias=type_bias, lip_reg=lip_reg)
        self.flow = SHAPEDecoder(z_s_dim, df_dim, dx_mlp_num, type_decoder=self.decoder, ins_norm=ins_norm, out_dim=out_dim)

    def forward(self, z_s, points, get_tmplt_coords=False, add_correction=True, inverse=True):
        # first flow back to the topology space
        output_sdv, output_flow_mag = [], []
        if get_tmplt_coords:
            points_t_list = []
        points_t, magnitude = self.flow(points, z_s, inverse=inverse)
        points_t = F.tanh(points_t) 
        if get_tmplt_coords:
            points_t_list.append(points_t)
        # get the sdv at the point locations
        points_t_sdv = self.decoder(points_t)
        
        output_sdv.append(points_t_sdv)
        
        if self.training:
            return {'recons': [act(o).permute(0, 2, 1) for o in output_sdv], 'flow_mag': magnitude}
        else:
            if get_tmplt_coords:
                return [act(o) for o in output_sdv], points_t_list
            else:
                return [act(o) for o in output_sdv] 
        

class NDF(nn.Module):
    def __init__(self, in_dim=1, \
            out_dim=1, \
            num_types=1, \
            z_s_dim=128, \
            type_mlp_num=6, \
            dx_mlp_num=6, \
            latent_dim=512, \
            ins_norm=False, \
            type_bias=True, \
            lip_reg=True):
        super(NDF, self).__init__()
        self.decoder = DisentangledGridDecoder3D(z_s_dim=z_s_dim, df_dim=latent_dim, type_mlp_num=type_mlp_num, dx_mlp_num=dx_mlp_num, out_dim=out_dim, ins_norm=ins_norm, lip_reg=lip_reg)
        self.in_dim = in_dim
    
    def forward(self, z_s, points, get_tmplt_coords=False):
        if self.training:
            outputs = self.decoder(z_s, points)
            return outputs
        else:
            if get_tmplt_coords:
                points_t_sdv_list, points_t_list = self.decoder(z_s, points)
                return points_t_sdv_list, points_t_list
            else:
                points_t_sdv_list = self.decoder(z_s, points)
                return points_t_sdv_list

class NDFTester(Tester):
    
    def z2voxel(self, z_s, network, num_blocks=1, out_block=-1, out_type=False, get_tmplt_coords=False, add_correction=True):
        # NEED TO CHANGE FOR TYPE ENCODER
        network.eval()
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2, self.out_dim], np.float32)
        dimc = self.cell_grid_size # 4^3 cube
        dimf = self.frame_grid_size # 64^3 cube

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, self.out_dim], np.uint8)
        queue = []
        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        out_point_coords_list = [torch.zeros((0, 3)).to(self.device) for i in range(num_blocks)]
        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                if out_type:
                    model_out_ = network.decoder.decoder(point_coord.flip(-1))
                    model_out_ = act(model_out_)
                else:
                    if get_tmplt_coords:
                        model_out_, points_t_list = network.decoder(z_s, point_coord.flip(-1), get_tmplt_coords, add_correction=add_correction)
                        for q, (points_t, sdf) in enumerate(zip(points_t_list, model_out_)):
                            out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(sdf>self.sampling_threshold, dim=-1)]], dim=0)
                    else:
                        model_out_ = network.decoder(z_s, point_coord.flip(-1), add_correction=add_correction)
                    model_out_ = model_out_[out_block]
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
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1, :] = 1.

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
                    model_out_batch_ = network.decoder.decoder(cell_coords.flip(-1))
                    print(torch.min(model_out_batch_), torch.max(model_out_batch_))
                    model_out_batch_ = act(model_out_batch_)
                else:
                    if get_tmplt_coords:
                        model_out_batch_, points_t_list = network.decoder(z_s, cell_coords.flip(-1), get_tmplt_coords, add_correction=add_correction)
                        for q, (points_t, sdf) in enumerate(zip(points_t_list, model_out_batch_)):
                            out_point_coords_list[q] = torch.cat([out_point_coords_list[q], points_t[torch.any(sdf>self.sampling_threshold, dim=-1)]], dim=0)
                    else:
                        model_out_batch_ = network.decoder(z_s, cell_coords.flip(-1), add_correction=add_correction)
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
    
