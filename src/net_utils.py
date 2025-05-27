import os
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
def act(x):
    x = torch.max(torch.min(x, x * 0.05 + 0.99), x * 0.05)
    return x


class LipLinearLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.c = nn.Parameter(torch.Tensor(1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            #nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.bias, 0)
        # Initialize with the infinity norm, max rowsum
        nn.init.constant_(self.c, torch.max(torch.sum(torch.abs(self.weight), dim=1)))

    @staticmethod
    def normalize(weight, c):
        absrowsum = torch.clip(torch.sum(torch.abs(weight), dim=1), min=1e-16)
        scale = torch.minimum(torch.ones_like(absrowsum), F.softplus(c)/absrowsum)
        return weight * scale.unsqueeze(1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #print("inf norm before: ", torch.max(torch.abs(self.weight)), self.c)
        normed_weight = self.normalize(self.weight, self.c)
        output = F.linear(input, normed_weight, self.bias)
        #output = F.linear(input, self.weight, self.bias)
        #print("inf norm after: ", torch.max(torch.abs(normed_weight)), self.c)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def positional_encoding(
    tensor, num_encoding_functions=4, include_input=True, log_sampling=True
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
        #return torch.cat(encoding, dim=-1)
        return torch.cat(encoding, dim=-1) * 2. + 2.01

class TypeEncoder(nn.Module):
    def __init__(self, in_dim, df_dim, mlp_num=4, out_dim=1):
        super(TypeEncoder, self).__init__()
        self.mlp_num = mlp_num 
        self.linear_1 = nn.Linear(in_dim, df_dim, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        for i in range(mlp_num):
            lin = nn.Linear(df_dim, df_dim, bias=True)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0)
            setattr(self, 'lin_{}'.format(i), lin)
        self.linear_out = nn.Linear(df_dim, out_dim * 8, bias=True)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)
        self.out_dim = out_dim

    def forward(self, z):
        x = self.linear_1(z)
        x = F.leaky_relu(x, negative_slope=0.02, inplace=True)
        for i in range(self.mlp_num):
            x = getattr(self, 'lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True) 
        x = self.linear_out(x)
        x = x.view(-1, self.out_dim, 2, 2, 2) 
        return x

class Tester:
    ''' 
    Based on code of UNIST: Unpaired Neural Implicit Shape Translation Network; Chen et al 2022
    '''
    def __init__(self, device, cell_grid_size=4, frame_grid_size=64, out_dim=1, sampling_threshold=0.5, binary=True):
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
        self.out_dim=out_dim
        self.cell_grid_size = cell_grid_size
        self.frame_grid_size = frame_grid_size
        self.real_size = self.cell_grid_size * self.frame_grid_size  # =256, output point-value voxel grid size in testing
        self.test_point_batch_size = self.test_size * self.test_size * self.test_size  # 32 x 32 x 32, do not change
        self.sampling_threshold = sampling_threshold
        self.device = device
        self.binary = binary

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
