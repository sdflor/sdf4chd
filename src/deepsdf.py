import os
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from io_utils import *
class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims= [ 512, 512, 512, 512, 512, 512, 512, 512 ],
        dropout=[0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob=0.2,
        norm_layers=[],
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=True,
        use_tanh = False,
        latent_dropout=False,
        positional_encoding = True,
        fourier_degree = 6, 
        out_dim=1,
        z_s_dim=32,
        ins_norm=True
        #omega = 30
    ):
        #norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
        super(DeepSDF, self).__init__()
        
        self.ins_norm = ins_norm
        if ins_norm:
            self.norm = torch.nn.InstanceNorm3d(num_features=z_s_dim)

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3+3] + dims + [out_dim]  # currently not used
        else:
            dims = [latent_size + 3] + dims + [out_dim]  # (x, y, z) -> (x, y, z, t)

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all # currently not used
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= (3+2*fourier_degree*3)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()


    # input: N x (L+3)
    def forward(self, grid, xyz):
        if self.ins_norm:
            grid = self.norm(grid)

        xyz_p = positional_encoding(xyz, self.fourier_degree)
        xyz_s= (xyz).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 4096, 3)
        shape_xyz = F.grid_sample(grid, xyz_s, padding_mode='border', align_corners=True)  # (B, 32, 1, 1, 4096)
        shape_xyz = shape_xyz.squeeze(2).squeeze(2).permute(0, 2, 1)
        input = torch.cat([shape_xyz, xyz_p], dim=-1)

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz_p], -1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            # layers getting a latent code input
            if layer in self.latent_in:
                x = torch.cat([x, input], -1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz_p], -1)
            x = lin(x)
            # last layer Tanh (if use_tanh = True)
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            # hidden layers
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = act(x)
        return x

class DeepSDFTester(Tester):
    
    def z2voxel(self, z_s, network):
        # NEED TO CHANGE FOR TYPE ENCODER
        network.eval()
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2, self.out_dim], np.float32)
        dimc = self.cell_grid_size # 4^3 cube
        dimf = self.frame_grid_size # 64^3 cube

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2, self.out_dim], np.uint8)
        queue = []
        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                model_out_ = network(z_s, point_coord.flip(-1))
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
                model_out_batch_ = network(z_s, cell_coords.flip(-1))
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
        return model_float
    
