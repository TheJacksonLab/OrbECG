#!/usr/bin/env python
import argparse
import math
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import RAdam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.utils import shuffle

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

try:
    from dig.threedgraph.evaluation import ThreeDEvaluator
except Exception:
    ThreeDEvaluator = None

# Globals used inside OrbECG
flag_cg_global = 0
device = torch.device('cpu')
wfn_var_global = None


class BT3DDataset(InMemoryDataset):
    """Bithiophene 3D dataset loader.

    Expects a raw npz named <molecule>_eV.npz in root/<molecule>/raw.
    """

    def __init__(self, root='dataset', molecule='BT', transform=None, pre_transform=None, pre_filter=None):
        self.molecule = molecule
        dataset_root = osp.join(root, molecule)
        super().__init__(dataset_root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.molecule}_eV.npz"]

    @property
    def processed_file_names(self):
        return [f"{self.molecule}_pyg.pt"]

    def download(self):
        raise RuntimeError(
            f"Raw dataset file not found. Expected: {self.raw_dir}/{self.raw_file_names[0]}"
        )

    def process(self):
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing raw dataset file: {raw_path}. Place the npz file there."
            )

        data = np.load(raw_path, allow_pickle=True)
        R = data['R']
        Z = data['Z']
        N = data['N']

        split = np.cumsum(N)[:-1]
        R_split = np.split(R, split)
        Z_split = np.split(Z, split)

        targets = {name: np.expand_dims(data[name], axis=-1) for name in ['eig', 'psi', 'ind']}

        data_list = []
        for i in tqdm(range(len(N)), desc='Processing'):
            R_i = torch.tensor(R_split[i], dtype=torch.float32)
            z_i = torch.tensor(Z_split[i], dtype=torch.int64)
            y_i = [torch.tensor(targets[name][i], dtype=torch.float32) for name in ['eig', 'psi', 'ind']]
            data_list.append(Data(pos=R_i, z=z_i, y=y_i[0], psi=y_i[1], ind=y_i[2]))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        data_size = min(int(data_size), len(self))
        if train_size + valid_size > data_size:
            raise ValueError(
                f"train_size + valid_size must be <= data_size (got {train_size + valid_size} > {data_size})"
            )
        ids = shuffle(range(data_size), random_state=seed)
        train_idx = torch.tensor(ids[:train_size])
        val_idx = torch.tensor(ids[train_size:train_size + valid_size])
        test_idx = torch.tensor(ids[train_size + valid_size:])
        return {'train': train_idx, 'valid': val_idx, 'test': test_idx}


def trim_to_full_batches(indices, batch_size):
    n = (len(indices) // batch_size) * batch_size
    return indices[:n]


def compute_wfn_variance(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dens_list = []
    for data in loader:
        dens = data.psi.view(-1, 8, 16)
        dens = dens / torch.sum(dens, dim=-1, keepdim=True)
        dens_list.append(dens)
    return torch.var(torch.cat(dens_list, dim=0), dim=0)


def r2_per_target(y_true, y_pred, eps=1e-12):
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    return 1.0 - ss_res / (ss_tot + eps)


def generate_correlation_torch(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.
    https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays?rq=3
    """
    mu_x = torch.mean(x,dim=1)
    mu_y = torch.mean(y,dim=1)
    n = x.shape[1]
    if n != y.shape[1]:
        print(x.shape)
        print(y.shape)
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = torch.std(x,dim=1)
    s_y = torch.std(y,dim=1)
    cov = (x - mu_x.view(-1,1)) @ (y.T - mu_y.view(1,-1)) / n
    return torch.diag( cov / ( 1e-3 + s_x.view(-1,1) @ s_y.view(1,-1) ) ) , (1 - s_x/(5*s_y))**2
class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=5):
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0

        # Exact shape match with params
        self.slow_weights = []
        for group in self.param_groups:
            for p in group['params']:
                self.slow_weights.append(p.data.clone().detach().contiguous())

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        idx += 1
                        continue
                    slow = self.slow_weights[idx]
                    # Align shape using .reshape_as
                    new_slow = slow + self.alpha * (p.data - slow)
                    self.slow_weights[idx] = new_slow.clone().detach().contiguous()
                    p.data.copy_(new_slow.reshape_as(p.data))
                    idx += 1
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': [w.clone() for w in self.slow_weights],
            'step_counter': self.step_counter
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        for i, sw in enumerate(state_dict['slow_weights']):
            self.slow_weights[i].copy_(sw)
        self.step_counter = state_dict['step_counter']

from torch_scatter import scatter, scatter_min, scatter_max, scatter_mean
from torch_scatter.composite import scatter_softmax
from torch_geometric.nn.pool import SAGPooling
from torch_cluster import radius_graph
from torch.nn import Embedding
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
import torch.nn.init as init
from torch import nn
from torch_geometric.nn import inits
import torch.nn.functional as F
from math import sqrt
from random import randint
import itertools


def swish(x):
    return x * torch.sigmoid(x)
class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='kaiming_uniform',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=1e0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)

class NLayerLinear(torch.nn.Module):
    def __init__(
            self,
            channels_list,
            bias=True,
            act=False,
    ):
        super(NLayerLinear, self).__init__()
        self.num_layers = len(channels_list) - 1
        self.lins = torch.nn.ModuleList(
            [
                Linear(
                    channels_list[i],
                    channels_list[i+1],
                    bias = bias
                )
                for i in range(self.num_layers)
            ]
        )
        self.act = act
        self.m = nn.ReLU()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i in range(self.num_layers-1):
            x = self.m(self.lins[i](x))
        x = self.lins[-1](x)
        if self.act:
            x = self.m(x)
        return x


class SoftEncoder(nn.Module):
    def __init__(self, num_bins=16, bin_min=-1.0, bin_max=1.0, sigma=0.2, threshold=1e-6):
        """
        Initialize the Gaussian kernel expansion layer with threshold masking.
        
        Args:
            num_bins (int): Number of bins to expand each value into
            bin_min (float): Minimum value for bin centers
            bin_max (float): Maximum value for bin centers
            sigma (float): Standard deviation of the Gaussian kernels
            threshold (float): Values below this threshold will be zeroed out
        """
        super().__init__()
        
        self.sigma = sigma
        self.num_bins = num_bins
        self.threshold = threshold
        
        # Create equally spaced bin centers
        bin_centers = torch.linspace(bin_min, bin_max, num_bins, device=device).unsqueeze(0)
        # self.enc_mu = nn.Parameter(-sigma**-1 * bin_centers.detach().clone())
        # self.enc_sigma = nn.Parameter(sigma**-1 * torch.ones_like(bin_centers))
        self.enc_weight = nn.Parameter(bin_centers.detach().clone()/sigma**2)
        self.enc_bias = nn.Parameter(-.5*(bin_centers.detach().clone()/sigma)**2)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        # assume x has shape (n,1)
        # enc_weight = self.enc_mu / (1e-10 + self.enc_sigma**2)
        # enc_bias = -.5 * (self.enc_mu / (1e-10 + self.enc_sigma))**2
        x = self.enc_weight * x + self.enc_bias
        
        # x = -torch.abs(self.enc_sigma*x+self.enc_mu)
        # output = self.sm()
        
        return x


def nanvar(tensor, dim=None, unbiased=True, keepdim=False):
    """
    Compute variance along the specified dimensions, ignoring NaN values.
    
    Args:
        tensor (Tensor): Input tensor.
        dim (int or tuple of ints, optional): Dimensions to reduce.
        unbiased (bool, optional): Whether to use Bessel's correction. Default: True
        keepdim (bool, optional): Whether the output has dim retained or not. Default: False
    
    Returns:
        Tensor: Variance of input tensor, ignoring NaNs.
    """
    # Create mask for non-NaN values
    mask = ~torch.isnan(tensor)
    
    # Completely separate computation paths for NaN and non-NaN values
    # Extract only valid values to compute on
    valid_tensor = tensor.clone()
    valid_tensor[~mask] = 0.0
    
    # Count number of non-NaN elements
    n = mask.sum(dim=dim, keepdim=True).clamp(min=1)  # Avoid division by zero
    
    # Calculate mean only on valid elements
    sum_valid = torch.sum(valid_tensor, dim=dim, keepdim=True)
    mean = sum_valid / n
    
    # Calculate squared difference but only where values are valid
    diff = valid_tensor - mean
    diff = diff * mask  # Zero out invalid positions
    
    # Square the differences
    sq_diff = diff.pow(2)
    
    # Sum squared differences and divide by count
    if unbiased:
        # Apply Bessel's correction for unbiased estimation
        n_unbiased = n - 1
        n_unbiased = n_unbiased.clamp(min=1)  # Handle case of single-element dimension
        divisor = n_unbiased if keepdim else n_unbiased.squeeze(dim)
    else:
        divisor = n if keepdim else n.squeeze(dim)
        
    var = torch.sum(sq_diff, dim=dim, keepdim=keepdim) / divisor
    
    return var

def masked_var(tensor, mask, dim=None, unbiased=True, keepdim=False):
    """
    Compute variance along the specified dimensions using an explicit mask.
    Functionally equivalent to nanvar but without introducing NaNs in the computation graph.
    
    Args:
        tensor (Tensor): Input tensor.
        mask (Tensor): Boolean mask of valid values.
        dim (int or tuple of ints, optional): Dimensions to reduce.
        unbiased (bool, optional): Whether to use Bessel's correction. Default: True
        keepdim (bool, optional): Whether the output has dim retained or not. Default: False
    
    Returns:
        Tensor: Variance of input tensor, using only masked values.
    """
    # Convert boolean mask to float for calculations
    mask_float = mask.float()
    
    # Count number of valid elements
    n = mask_float.sum(dim=dim, keepdim=True).clamp(min=1)  # Avoid division by zero
    
    # Calculate mean only on valid elements
    valid_tensor = tensor * mask_float  # Zero out invalid positions
    sum_valid = torch.sum(valid_tensor, dim=dim, keepdim=True)
    mean = sum_valid / n
    
    # Calculate squared difference but only where values are valid
    diff = (tensor - mean) * mask_float  # Zero out invalid positions
    
    # Square the differences
    sq_diff = diff.pow(2)
    
    # Sum squared differences and divide by count
    if unbiased:
        # Apply Bessel's correction for unbiased estimation
        n_unbiased = (n - 1).clamp(min=1)  # Handle case of single-element dimension
        divisor = n_unbiased if keepdim else n_unbiased.squeeze(dim)
    else:
        divisor = n if keepdim else n.squeeze(dim)
        
    var = torch.sum(sq_diff, dim=dim, keepdim=keepdim) / divisor
    
    return var

class OrbECG(nn.Module):
    def __init__(
            self,
            hidden_channels=64
    ):
        super().__init__()
        act = swish
        
        self.sigma = 1.
        # self.sigma_param = nn.Parameter(torch.tensor([.3],device=device))
        # self.sigma_param = nn.Parameter(1.*torch.ones((self.num_atoms,1),device=device))

        if flag_cg_global == 0:
            self.num_atoms = 16
            self.edge1 = [0,0,1,2,2,3,4,5,6,8,8,9,10,11,11,13,14]
            self.edge2 = [1,4,2,3,5,4,12,6,7,9,12,10,11,12,13,14,15]
            self.edge_list = [[1,4],[0,2],[1,3,5],[2,4],[0,3,12],[2,6],[5,7],[5,6],[9,12],[8,10],[9,11],[10,12,13],[4,8,11],[11,14],[13,15],[13,14]]
        else:
            self.num_atoms = 8
            self.edge1 = [0,0,1,1,2,4,4,5,5]
            self.edge2 = [1,2,2,3,6,5,6,6,7]
            self.edge_list = [[1,2],[0,2,3],[0,1,6],[1,2],[5,6],[4,6,7],[2,4,5],[5,6]]

        self.cg_proj = torch.tensor([[8./11,3./11,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,.5,.5,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,1./3,1./3,1./3,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,8./11,3./11,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,.5,.5,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1./3,1./3,1./3]],device=device)
        self.assign_aa_to_cg = torch.tensor([[1.,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]],device=device)
        
        # self.cross_scale = [-.5,-.25,.25,.5]
        # self.cross_scale = [-.5,.5,-.25,.25]
        # self.cross_scale = [-.25,.25,-.1,.1]
        self.cross_scale = [-.25,.25]
        # self.cross_scale = [-.25,.25]
        # self.basis_list = [.75,.5,.25]
        self.basis_list = [.5]
        # self.sigma_list = [1.,2.,3.,.3,.6]
        # self.sigma_list = [.2,.4,.8,1.,1.5]
        self.sigma_list = [.4,.8,1.2]
        # self.sigma_param = nn.Parameter(torch.tensor(self.sigma_list,device=device))
        self.sigma_param = nn.Parameter(torch.zeros(len(self.sigma_list),device=device))
        # self.sigma_param = torch.tensor(self.sigma_list,device=device)
        # self.sigma_param = nn.Parameter(torch.tensor([.5,1.,1.5,2],device=device))
        # self.num_basis = (self.num_atoms + len(self.edge1) * len(self.basis_list)) * len(self.sigma_list)
        self.num_p = len(self.cross_scale)*sum([len(x)*(len(x)-1)//2 for x in self.edge_list])
        self.num_basis = len(self.sigma_param)*(self.num_atoms + len(self.edge1) + self.num_p)

        self.edge_scale = 1
        
        self.flag_train = 1
        self.temp = 0
        self.temp2 = 0
        self.pop_thresh = 1e-1
        self.m = nn.ReLU()
        
        self.num_beads = 8
        self.num_orbs = 8
        
        self.cg_ref = nn.Parameter(.01*torch.randn((1,1*self.num_beads,self.num_basis),device=device))
    
        
        # self.proj = nn.Parameter(torch.randn((512,hidden_channels),device=device))

        # self.gke_overlap = GaussianKernelExpansion(num_bins=64, bin_min=-1.1, bin_max=1.1, sigma=0.15, threshold=1e-4)
        num_enc = 128
        # self.gke_energy = GaussianKernelExpansion(num_bins=num_enc, bin_min=-.1, bin_max=3., sigma=0.15, threshold=1e-4)
        self.enc_ovl = SoftEncoder(num_bins=num_enc, bin_min=-1., bin_max=1., sigma=0.2, threshold=1e-10)
        self.enc_in = SoftEncoder(num_bins=num_enc, bin_min=-2.5, bin_max=3., sigma=.75, threshold=1e-10)
        # self.enc_ham = SoftEncoder(num_bins=num_enc, bin_min=-3., bin_max=3., sigma=0.5, threshold=1e-10)
        # self.mu_proj = nn.Parameter(torch.cat((torch.linspace(-6.,-3.5,num_enc//2,device=device),torch.linspace(3.5,6.5,num_enc//2,device=device))).unsqueeze(1))
        self.mu_proj = nn.Parameter(torch.linspace(-2.5,3.,num_enc,device=device).unsqueeze(1))
        self.pred = NLayerLinear([num_enc, 2*hidden_channels,2*hidden_channels,num_enc],act=False)
        
        self.sm = nn.Softmax(dim=1)
        
        self.lr_proj = nn.Parameter(torch.tensor([3.5],device=device))
        self.test = 0

        mulliken_temp = torch.zeros((self.num_atoms,len(self.basis_list)*len(self.edge1)),device=device)
        loop_ct = 0
        for i in range(len(self.basis_list)):
            for j in range(len(self.edge1)):
                mulliken_temp[self.edge1[j],loop_ct] = self.basis_list[i]
                mulliken_temp[self.edge2[j],loop_ct] = 1-self.basis_list[i]
                loop_ct += 1
        self.mulliken = torch.cat((torch.eye(self.num_atoms,device=device),mulliken_temp),dim=-1).T # .repeat(1,len(self.sigma_list)).T

        list_edges = self.edge_list
        # Pre-compute all possible edge pairs and their indices
        edge_pairs = []
        node_indices = []
        mulliken_indices = []
        loop_ct = 0
        for node_idx in range(self.num_atoms):
            connected_nodes = list_edges[node_idx]
            mulliken_node = [0 for i in range(self.num_atoms)]
            mulliken_node[node_idx] = 1
            
            for i in range(len(connected_nodes)):
                for j in range(i + 1, len(connected_nodes)):
                    edge_pairs.append((connected_nodes[i], connected_nodes[j]))
                    node_indices.append(node_idx)


                    for k in range(len(self.cross_scale)):
                        mulliken_indices += [mulliken_node]
                        loop_ct += 1

        # self.num_p = loop_ct
        # Convert to tensors for parallel processing
        self.edge_pairs = torch.tensor(edge_pairs, device=device)
        self.node_indices = torch.tensor(node_indices, device=device)

        self.mulliken = torch.cat((self.mulliken,torch.tensor(mulliken_indices,device=device))).repeat(len(self.sigma_param),1).T

    def make_opposite_blocks(self, tensor_in):
        n1 = self.num_atoms + len(self.edge1)
        n2 = self.num_p
        n_blocks = len(self.sigma_param)
        batch = tensor_in.shape[0]
        block_length = n1 + n2
        
        # Use detach() to break gradient connections for the base tensor
        result = tensor_in.clone().detach()
        
        for i in range(n_blocks):
            block_start = i * block_length
            n2_start = block_start + n1
            n2_block = tensor_in[:, n2_start:n2_start + n2]
            odd_cols = n2_block[:, ::2]
            
            # Maintain gradients only for odd columns
            result[:, n2_start + 0:n2_start + n2:2] = odd_cols
            # Create new tensor for even columns with no gradient connection
            result[:, n2_start + 1:n2_start + n2:2] = -odd_cols.detach()
        
        return result
    
    def generate_reference_vectors(self, pos):
        """
        Generate reference vectors using batch parallelization for better performance.
        
        Args:
            pos (torch.Tensor): Node positions tensor of shape (batch, n_nodes, 3)
            list_edges (list): Nested list where list_edges[i] contains indices of nodes connected to node i
            scale_factors (list): List of scaling factors to apply to each reference vector
            
        Returns:
            torch.Tensor: Reference vectors tensor of shape (batch, n_vectors, 3)
        """
        batch_size, n_nodes, _ = pos.shape
        
        scale_factors = self.cross_scale
        edge_pairs = self.edge_pairs
        node_indices = self.node_indices
        
        # Get all relevant positions in parallel
        edge1_pos = pos[:, edge_pairs[:, 0]]  # Shape: (batch, n_pairs, 3)
        edge2_pos = pos[:, edge_pairs[:, 1]]  # Shape: (batch, n_pairs, 3)
        node_pos = pos[:, node_indices]        # Shape: (batch, n_pairs, 3)
        
        # Calculate edge vectors in parallel
        edge1_vectors = edge1_pos - node_pos   # Shape: (batch, n_pairs, 3)
        edge2_vectors = edge2_pos - node_pos   # Shape: (batch, n_pairs, 3)
        
        # Calculate edge lengths in parallel
        edge1_lengths = torch.norm(edge1_vectors, dim=2)  # Shape: (batch, n_pairs)
        edge2_lengths = torch.norm(edge2_vectors, dim=2)  # Shape: (batch, n_pairs)
        
        # Calculate geometric mean of edge lengths
        geom_mean = torch.sqrt(edge1_lengths * edge2_lengths).unsqueeze(2)  # Shape: (batch, n_pairs, 1)
        
        # Calculate cross products in parallel
        cross_products = torch.cross(edge1_vectors, edge2_vectors, dim=2)  # Shape: (batch, n_pairs, 3)
        
        # Normalize cross products
        cross_norms = torch.norm(cross_products, dim=2, keepdim=True)
        normalized_crosses = cross_products * geom_mean / (cross_norms + 1e-8)  # Shape: (batch, n_pairs, 3)
        
        # Apply scale factors and create positive/negative versions
        scale_factors = torch.tensor(scale_factors, device=device)
        n_pairs = len(node_indices)
        n_scales = len(scale_factors)
        
        # Expand dimensions for broadcasting
        normalized_crosses = normalized_crosses.unsqueeze(2)  # (batch, n_pairs, 1, 3)
        scale_factors = scale_factors.view(1, 1, -1, 1)      # (1, 1, n_scales, 1)
        node_pos = node_pos.unsqueeze(2)                     # (batch, n_pairs, 1, 3)
        
        # Generate all scaled versions in parallel
        scaled_vectors = normalized_crosses * scale_factors   # (batch, n_pairs, n_scales, 3)
        
        # Create positive and negative versions
        pos_vectors = node_pos + scaled_vectors              # (batch, n_pairs, n_scales, 3)
        # neg_vectors = node_pos - scaled_vectors              # (batch, n_pairs, n_scales, 3)
        
        # Combine and reshape
        # all_vectors = torch.cat([pos_vectors, neg_vectors], dim=2)  # (batch, n_pairs, 2*n_scales, 3)
        final_shape = (batch_size, n_pairs * n_scales, 3)
        pos_edges = pos_vectors.reshape(final_shape)
        
        return pos_edges

    def gaussian_graph_overlap_degen(self, pos):
        # sigma_copy = torch.tensor(self.sigma_list, device=device).view(-1,1)
        # sigma_copy = 1e-8+torch.abs(self.sigma_param.reshape(-1,1))
        
        
        pos_expanded = torch.cat((pos.mT,torch.cat([(pos[:,self.edge1]*b + pos[:,self.edge2]*(1-b)).mT for b in self.basis_list],dim=-1)),dim=-1).mT
        pos_expanded = torch.cat((pos_expanded.mT, self.generate_reference_vectors(pos).mT),dim=-1).mT
        
        r_mat_ref = torch.cdist(pos_expanded,pos_expanded,p=2)


        sigma_copy = (torch.tensor(self.sigma_list,device=device)+.25*torch.tanh(self.sigma_param)).reshape(-1,1)
        sigma_A = sigma_copy + torch.zeros_like(sigma_copy.T)
        r_mat = torch.kron(torch.ones_like(sigma_A), r_mat_ref)
        sigma_A = torch.kron(sigma_A,torch.ones_like(r_mat_ref))
        sigma_B = torch.zeros_like(sigma_copy) + sigma_copy.T
        sigma_B = torch.kron(sigma_B,torch.ones_like(r_mat_ref))
        sigma_AB = sigma_A * sigma_B
        sigma_AmB = sigma_A**2 - sigma_B**2
        
        condition_r = r_mat < 1e-6
        condition_a = torch.abs(sigma_AmB) < 1e-6

        r_mat = torch.where(condition_r, 1e-6*torch.ones_like(r_mat), r_mat)
        sigma_AmB = torch.where(condition_a, 1e-6*torch.ones_like(sigma_AmB), sigma_AmB)
        
        overlap_full = 8 * (sigma_AB)**1.5 * sigma_AmB**-3 * ( torch.exp(-r_mat * sigma_A**-1) * (sigma_A * sigma_AmB - 4*sigma_AB**2 * r_mat**-1) 
                                                             + torch.exp(-r_mat * sigma_B**-1) * (sigma_B * sigma_AmB + 4*sigma_AB**2 * r_mat**-1) )
        # overlap_full = overlap_full_unsafe.clone()
        # overlap_full.masked_fill_(torch.isnan(overlap_full),0.0)
        overlap_r0 = 8 * (sigma_AB * (sigma_A + sigma_B)**-2)**1.5
        overlap_AeB = torch.exp(-r_mat * sigma_B**-1) * ( 1 + r_mat / sigma_B + 1./3 * (r_mat / sigma_B)**2 )

        overlap = torch.where(condition_a, overlap_AeB, overlap_full)
        overlap = torch.where(condition_r, overlap_r0, overlap)

        
        overlap = 0.5 * (overlap + overlap.mT)

        diag_mask = torch.eye(overlap.shape[-1], device=overlap.device)
        overlap = overlap * (1 - diag_mask) + diag_mask #* (1 + 1e-8)

        dist_mat = r_mat**2 + 3 * (sigma_A**2+sigma_B**2)
        # dist_mat = dist_mat / torch.kron(sigma_copy @ torch.ones_like(sigma_copy.T), torch.ones_like(r_mat))
        
        return overlap, dist_mat

    def second_derivative_regularization(self, points_batch):
        """
        Compute L2 norm of numerical second derivative using adjacent points.
        
        Args:
            function_placeholder: Function that takes (n,1) tensor and returns (n,1) tensor
            domain_points: (n,1) tensor of evenly spaced points
        
        Returns:
            Scalar tensor containing the regularization loss
        """
        # Compute spacing (h) from domain points
        start=torch.min(points_batch).item()-.25
        end=torch.max(points_batch).item()+.25
        n_points=100
        domain_points = torch.linspace(start, end, n_points, device=device).reshape(-1, 1)
        
        h = (domain_points[1] - domain_points[0]).item()
        
        # Evaluate function at all points at once
        es_enc = self.enc_ovl(domain_points)
        eh_enc = self.sm( es_enc + self.pred(es_enc) )
        f = eh_enc @ self.mu_proj
        
        # Compute second derivative using adjacent points (excluding endpoints)
        second_deriv = (f[2:] - 2*f[1:-1] + f[:-2]) / (h * h)
        
        # Compute L2 norm (scaled by dx for integration)
        # reg_loss = torch.mean((second_deriv-torch.mean(second_deriv))**2) * h
        reg_loss = torch.mean(second_deriv**2) * h
        
        return reg_loss, torch.cat((domain_points,f),1)

    def mulliken_orbital_populations(self, wfn_full, ovl_full, mulliken_weights):
        """
        Calculate Mulliken populations for each orbital
        
        Args:
            wfn_full: Molecular orbital coefficients (batch, num_orbitals, num_basis)
            ovl_full: Overlap matrix (batch, num_basis, num_basis)
            mulliken_weights: Atomic weights for basis functions (num_atoms, num_basis)
        
        Returns:
            Orbital populations per atom (batch, num_atoms, num_orbitals)
        """
        batch, num_orbitals, num_basis = wfn_full.shape
        num_atoms = mulliken_weights.shape[0]
        
        # Density matrix terms for each orbital
        # P_munu = c_mui c_nui for orbital i
        # Shape: (batch, num_orbitals, num_basis, num_basis)
        density_terms = wfn_full.unsqueeze(-1) * wfn_full.unsqueeze(-2)
        
        # Multiply by overlap matrix: P_munu * S_munu
        # Shape: (batch, num_orbitals, num_basis, num_basis)
        weighted_density = density_terms * ovl_full.unsqueeze(1)
        
        # Sum over all nu to get gross populations for each mu
        # Shape: (batch, num_orbitals, num_basis)
        basis_populations = weighted_density.sum(dim=-1)
        
        # Convert to atomic populations using Mulliken weights
        # Shape: (batch, num_orbitals, num_atoms)
        atomic_populations = torch.einsum('bok,ak->boa', basis_populations, mulliken_weights)
        atomic_populations = atomic_populations / torch.sum(atomic_populations,-1).unsqueeze(-1)
        
        return atomic_populations
    
    def lowdin_orbital_populations(self, wfn_full, ovl_full, mulliken_weights):
        """
        Calculate Lowdin populations for each orbital
        
        Args: same as mulliken_orbital_populations
        Returns: Orbital populations per atom (batch, num_atoms, num_orbitals)
        """
        batch, num_orbitals, num_basis = wfn_full.shape
        
        # Calculate S^(-1/2) for each batch
        # First compute eigendecomposition of S
        eigvals, eigenvecs = torch.linalg.eigh(ovl_full)
        
        # Compute S^(-1/2) = U diag(s^(-1/2)) U^T
        # print(eigvals[0])
        eigenvals = torch.where(eigvals > self.pop_thresh, eigvals.pow(-0.5), torch.zeros_like(eigvals))
        # if torch.sum(torch.isnan(eigenvals)) > 0:
        #     print(eigenvals[0])
        # print(eigenvals[0])
        s_inv_sqrt = eigenvecs @ torch.diag_embed(eigenvals) @ eigenvecs.transpose(-2, -1)
        
        # Transform coefficients
        # Shape: (batch, num_orbitals, num_basis)
        wfn_transformed = torch.bmm(wfn_full, s_inv_sqrt)
        
        # Square the coefficients
        # Shape: (batch, num_orbitals, num_basis)
        basis_populations = wfn_transformed.pow(2)
        
        # Convert to atomic populations using Mulliken weights
        # Shape: (batch, num_orbitals, num_atoms)
        atomic_populations = torch.einsum('bok,ak->boa', basis_populations, mulliken_weights)
        atomic_populations = atomic_populations / torch.sum(atomic_populations,-1).unsqueeze(-1)
        
        return atomic_populations

    def vec_entropy(self, x):
        output = -torch.sum(x * torch.log(1e-8 + x),-1)
        return output

    def kl_div(self, x1, x2):
        output = torch.sum(x1 * torch.log(1e-8 + x1 / (1e-8 + x2)),-1)
        return output
    
    def _forward(self, data):
        batch = data.batch
        z = data.z.long()
        # b = data.b.long()
        pos = data.pos
        num_atoms = self.num_atoms

        y_ref = data.y.view(-1,8) + 3.515
        
        # y_nogap = y_ref - self.lr_proj[0] * torch.tensor([[-1.,-1,1,1]],device=device)
        # y_nogap = y_nogap.view(-1,1)

        dens_ref = data.psi.view(-1,8,16)
        dens_ref = dens_ref / torch.sum(dens_ref,-1).unsqueeze(-1)
        
        if self.num_orbs < 8:
            diff = (8-self.num_orbs)//2
            y_ref = y_ref[diff:-diff]
            dens_ref = dens_ref[:,diff:-diff,:]

        pos2 = pos.view(-1,16,3)
        if flag_cg_global == 1:
            pos2 = self.cg_proj @ pos2
            dens_ref = dens_ref @ self.assign_aa_to_cg.T
        # if self.test == 0 and self.temp2 > 0:
        #     pos2 = pos2 + .25 * self.temp2 * torch.randn_like(pos2)

        r_mat = torch.cdist(pos2,pos2,p=2)
        ovl_temp, r_full = self.gaussian_graph_overlap_degen(pos2)
        
        # ovl_full[torch.abs(ovl_full) < 1e-4] = 0
        # ovl_full = torch.where(ovl_temp >= 1e-4, ovl_temp, torch.zeros_like(ovl_temp))
        ovl_full = ovl_temp
        batch_size = torch.max(batch) + 1
        # x_int = self.make_opposite_blocks(self.cg_ref[0]).unsqueeze(0)
        if self.test == 0:
            x_int = self.make_opposite_blocks(self.cg_ref[0]).unsqueeze(0)
        else:
            x_int = self.make_opposite_blocks(self.cg_ref[0]).unsqueeze(0)
        
        
        basis_ovl = x_int**2
        basis_ovl = basis_ovl / torch.sum(basis_ovl,-1).unsqueeze(-1)
        cutoff_psi = 6
        if self.test == 0:
            cutoff_psi = cutoff_psi + np.random.randn()
        x_int_norm = torch.where(basis_ovl >= 10**(-cutoff_psi), x_int, torch.zeros_like(x_int)).repeat((batch_size,1,1))
        x_mask = torch.where(basis_ovl >= 10**(-cutoff_psi), torch.ones_like(x_int), torch.zeros_like(x_int))

        ovl_cg = x_int_norm @ ovl_full @ x_int_norm.mT
        ovl_norm = torch.diagonal(ovl_cg,dim1=-2,dim2=-1).view(-1,1,self.num_beads)
        x_int_norm = x_int_norm.view(-1,self.num_basis) * (1e-12 + ovl_norm.reshape(-1,1))**-.5


        ovl_cg = ovl_cg * (ovl_norm.mT @ ovl_norm)**-.5
        ovl_cg = ovl_cg - torch.eye(self.num_beads,device=device)
        
        energy_raw_full, eigvecs = torch.linalg.eigh(ovl_cg)
        if self.num_orbs < self.num_beads:
            orb_diff = (self.num_beads-self.num_orbs)//2
            energy_raw_0 = energy_raw_full[:,orb_diff:-orb_diff]
            eigvecs = eigvecs[:,:,orb_diff:-orb_diff]
        else:
            energy_raw_0 = energy_raw_full
        
        
        if self.test == 0:
            es_enc = self.enc_ovl(energy_raw_0.reshape(-1,1))
        else:
            es_enc = self.enc_ovl(energy_raw_0.reshape(-1,1))
        
        eh_enc = es_enc + self.pred(es_enc)
        eh_enc = self.sm( eh_enc )
        es_enc = self.sm(es_enc)
        
        e_sign = (torch.tensor([[-1],[1.]],device=device)*torch.ones((2,self.num_orbs//2),device=device)).reshape(1,-1)
        energy_pred_0 = eh_enc @ self.mu_proj + self.lr_proj[0] * e_sign.repeat(batch_size,1).view(-1,1)
        if self.test == 0:
            energy_pred = energy_pred_0 #+ (energy_raw_skip1 + energy_raw_skip2).view(-1,1)
        else:
            energy_pred = energy_pred_0
        
        x_int_norm = x_int_norm.view(-1,self.num_beads,self.num_basis)
        x_square = x_int_norm**2
        dist_reg = torch.diagonal(torch.abs(x_int_norm) @ r_full**2 @ torch.abs(x_int_norm).mT, dim1=-1,dim2=-2)
        dist_reg = dist_reg / torch.sum(x_square,-1)
        dist_reg = torch.mean(dist_reg,0).unsqueeze(-1)
        dist_reg = dist_reg @ dist_reg.T
        
        dist_reg = dist_reg * torch.mean(ovl_cg**2,0) # / (1e-8+torch.var(ovl_cg,0)) * (1-torch.eye(self.num_beads,device=device))
        
        loss_temp = torch.zeros(1,device=device)
        
        ipr_mod = torch.diagonal(x_square @ ovl_full**2 @ x_square.mT, dim1=-1,dim2=-2)
        ipr_mod = ipr_mod / torch.sum(x_square,-1)**2
        
        loss_diff, nn_plot = self.second_derivative_regularization(energy_raw_0)
        wfn_atom = eigvecs.mT @ x_int_norm

        ovl_eig = wfn_atom @ ovl_full @ wfn_atom.mT
        ovl_eig_norm = torch.diagonal(ovl_eig,dim1=-2,dim2=-1).view(-1,1,self.num_orbs)
        wfn_atom = (wfn_atom.view(-1,self.num_basis) * (1e-9 + ovl_eig_norm.reshape(-1,1))**-.5).view(-1,self.num_orbs,self.num_basis)
        # batch, num_beads, num_atoms
        wfn_out = wfn_atom**2 @ self.mulliken.T.unsqueeze(0)
        wfn_out = wfn_out / torch.sum(wfn_out,-1).unsqueeze(-1)

        loss_kl = torch.mean(loss_diff)
        if flag_cg_global == 0:
            loss_wfn = torch.mean((dens_ref-wfn_out)**2 / (1e-5 + torch.mean(wfn_var_global.to(device)))) #/ .0018
        else:
            loss_wfn = torch.mean((dens_ref-wfn_out)**2 / (1e-5 + wfn_var_global.to(device)@self.assign_aa_to_cg.T)) #/ .0018
        if self.test == 0:
            return energy_pred, torch.mean(dist_reg), loss_wfn, torch.mean(torch.zeros(1,device=device)), torch.mean(loss_temp), loss_kl, energy_raw_0, energy_pred_0, 0
        
        
        mulliken_out = self.mulliken_orbital_populations(wfn_atom, ovl_full, self.mulliken)
        lowdin_out = self.lowdin_orbital_populations(wfn_atom, ovl_full, self.mulliken)
        
        return energy_pred, mulliken_out, x_int_norm, nn_plot, lowdin_out, wfn_out, wfn_atom, ovl_cg, energy_raw_0.reshape(-1,1)

    def forward(self, batch_data):
        out, out2, out3, out4, out5, out6, out7, out8, out9 = self._forward(batch_data)
        return out, out2, out3, out4, out5, out6, out7, out8, out9

class OrbECGTrainer:
    """Training loop wrapper for OrbECG."""

    def __init__(self, device):
        self.device = device
        self.flag_temp = 0
        self.score_data = 0
        self.score_update = 0
        self.l2 = 0
        self.l4 = 0
        self.target = 0
        self.deg = 0
        self.n_atoms = 32
        self.m = nn.ReLU()
        self.cap = 5
        self.train_mean = 0
        self.train_std = 1
        self.ref_std = torch.tensor(
            [[0.028614, 0.034424, 0.047732, 0.038061, 0.038520, 0.063247, 0.061445, 0.11079]],
            device=self.device,
        )
        self.ref_mean = torch.tensor(
            [[-0.535, -0.456, -0.349, -0.227, -0.110, 0.262, 0.486, 0.929]],
            device=self.device,
        )
        self.batch_size = 0
        self.batch_size_test = 0
        self.temp2 = 0

    def train_loop(
        self,
        device,
        train_dataset,
        valid_dataset,
        test_dataset,
        model,
        loss_func,
        likelihood,
        trial=0,
        epochs=500,
        batch_size=32,
        vt_batch_size=32,
        lr=0.0005,
        lr_decay_factor=0.5,
        lr_decay_step_size=50,
        weight_decay=0,
        energy_and_force=False,
        p=100,
        save_dir='',
        log_dir='',
    ):
        metric_labels = ['loss_R', 'loss_C', 'loss_S', 'loss_D', 'loss_Sp']

        def format_metrics(split_name, values):
            parts = [f'{label}={value:.6f}' for label, value in zip(metric_labels, values)]
            return f"{split_name}: " + ", ".join(parts)

        self.batch_size = batch_size
        self.batch_size_test = vt_batch_size
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')

        base_optim = RAdam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        optimizer = Lookahead(base_optim, alpha=0.5, k=5)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            _ = SummaryWriter(log_dir=log_dir)

        temp_list = np.maximum(0 * np.ones(epochs), np.minimum(np.ones(epochs), np.linspace(1.25, -0.5, epochs)))
        temp2_list = np.maximum(0 * np.ones(epochs), np.minimum(np.ones(epochs), np.linspace(1.0, -2.0, epochs)))
        model.time = 0
        loss_best = 500000000

        for epoch in range(1, epochs + 1):
            model.temp = (1 - temp_list[epoch - 1]) ** 2
            model.temp2 = temp2_list[epoch - 1]

            print(f"=====Epoch {epoch}", flush=True)

            print('Training...', flush=True)
            train_metrics = self.train(
                model, optimizer, train_loader, energy_and_force, likelihood, loss_func, device
            )

            print('Evaluating...', flush=True)
            valid_metrics = self.val(
                model, valid_loader, energy_and_force, likelihood, loss_func, device
            )

            print('Testing...', flush=True)
            test_metrics = self.val(
                model, test_loader, energy_and_force, likelihood, loss_func, device
            )

            print()
            print(format_metrics('Train', train_metrics))
            print(format_metrics('Validation', valid_metrics))
            print(format_metrics('Test', test_metrics))

            scheduler.step()
            if not math.isnan(train_metrics[0]):
                torch.save(model.state_dict(), 'model_simplest.pt')
                if train_metrics[0] < loss_best:
                    loss_best = train_metrics[0]
                    torch.save(model.state_dict(), 'model_simplest_best.pt')

        self.report_r2(model, train_dataset, valid_dataset, test_dataset, device)

    def _collect_predictions(self, model, data_loader, device):
        preds = []
        targets = []
        model.eval()
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                out, out2, out3, out4, out5, out6, out7, out8, _ = model(batch_data)
                y_ref = batch_data.y.view(-1, 8) + 3.515
                if model.num_orbs < 8:
                    diff = (8 - model.num_orbs) // 2
                    y_ref = y_ref[:, diff:-diff]
                pred = out.reshape(-1, model.num_orbs)
                preds.append(pred.detach().cpu())
                targets.append(y_ref.detach().cpu())
        return torch.cat(targets, dim=0), torch.cat(preds, dim=0)

    def report_r2(self, model, train_dataset, valid_dataset, test_dataset, device):
        loaders = {
            'Train': DataLoader(train_dataset, self.batch_size_test, shuffle=False),
            'Validation': DataLoader(valid_dataset, self.batch_size_test, shuffle=False),
            'Test': DataLoader(test_dataset, self.batch_size_test, shuffle=False),
        }
        for split, loader in loaders.items():
            y_true, y_pred = self._collect_predictions(model, loader, device)
            r2 = r2_per_target(y_true, y_pred)
            r2_list = ', '.join([f"{v:.4f}" for v in r2.tolist()])
            print(f"{split} R2 per target: [{r2_list}]")

    def train(self, model, optimizer, train_loader, energy_and_force, likelihood, loss_func, device):
        model.train()
        loss_accum = 0
        loss_recon_accum = 0
        loss_d2_accum = 0
        loss_dist_accum = 0
        loss_ipr_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            model.time += 1
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out, out2, out3, out4, out5, out6, out7, out8, _ = model(batch_data)

            loss_recon = 0
            loss = 0
            loss_reg = 0
            loss_1 = 0
            loss_2 = 0
            y_ref = batch_data.y.view(-1, 8) + 3.515
            train_mag = torch.tensor([0.2172, 0.0801, 0.1026, 0.2364, 0.2194, 0.2508, 0.1802, 0.1211], device=device)
            if model.num_orbs < 8:
                diff = (8 - model.num_orbs) // 2
                y_ref = y_ref[:, diff:-diff].detach()
                train_mag = train_mag[diff:-diff]

            loss_1, loss_2 = generate_correlation_torch(y_ref.T, out7.T)
            loss_5, loss_6 = generate_correlation_torch(y_ref.T, out.reshape(-1, model.num_orbs).T)
            loss_3, loss_4 = generate_correlation_torch(y_ref.reshape(1, -1), out.T)
            loss_recon = 0 - torch.mean(loss_3) - torch.sum(loss_1) - torch.sum(loss_5)

            loss_mae = (torch.abs(y_ref.reshape(-1, 1) - out).reshape(-1, model.num_orbs) / train_mag)
            loss_mae = torch.mean(loss_mae**2)

            loss_reg = loss_recon

            cg_prob = torch.abs(model.make_opposite_blocks(model.cg_ref[0])) ** 2
            cg_prob = cg_prob / torch.sum(cg_prob, -1, keepdim=True)
            cg_prob = torch.mean(torch.sum((1e-10 / model.num_basis + cg_prob) ** 2, -1) ** -2)

            seig_reg = torch.var(out.reshape(-1, model.num_orbs), 0) / (train_mag.unsqueeze(0) / torch.mean(train_mag)) ** 2
            seig_reg = (1 + seig_reg**1.5) / (1e-8 + seig_reg)
            seig_reg = torch.mean(seig_reg)

            loss = 2 * loss_mae + 1 * loss_recon + (0.1 * out6) + 2 * out3 + 0.25 * out2 / out2.detach() + 0.1 * cg_prob / cg_prob.detach()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            loss_accum += loss_mae.item()
            loss_recon_accum += loss_reg.item()
            loss_d2_accum += out6.item()
            loss_dist_accum += out3.item()
            loss_ipr_accum += out2.item()

        return (
            loss_accum / (step + 1),
            loss_recon_accum / (step + 1),
            loss_d2_accum / (step + 1),
            loss_dist_accum / (step + 1),
            loss_ipr_accum / (step + 1),
        )

    def val(self, model, data_loader, energy_and_force, likelihood, loss_func, device):
        model.eval()

        loss_accum = 0
        loss_recon_accum = 0
        loss_d2_accum = 0
        loss_dist_accum = 0
        loss_ipr_accum = 0
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out, out2, out3, out4, out5, out6, out7, out8, _ = model(batch_data)

            loss_recon = 0
            loss = 0
            loss_reg = 0
            loss_1 = 0
            loss_2 = 0

            y_ref = batch_data.y.view(-1, 8) + 3.515
            train_mag = torch.tensor([0.2172, 0.0801, 0.1026, 0.2364, 0.2194, 0.2508, 0.1802, 0.1211], device=device)
            if model.num_orbs < 8:
                diff = (8 - model.num_orbs) // 2
                y_ref = y_ref[:, diff:-diff].detach()
                train_mag = train_mag[diff:-diff]

            loss_1, loss_2 = generate_correlation_torch(y_ref.T, out7.T)
            loss_5, loss_6 = generate_correlation_torch(y_ref.T, out.reshape(-1, model.num_orbs).T)
            loss_3, loss_4 = generate_correlation_torch(y_ref.reshape(1, -1), out.T)
            loss_recon = 0 - torch.mean(loss_3) - torch.sum(loss_1) - torch.sum(loss_5)

            loss_mae = (torch.abs(y_ref.reshape(-1, 1) - out).reshape(-1, model.num_orbs) / train_mag)
            loss_mae = torch.mean(loss_mae**2)
            loss_reg = loss_recon

            loss_accum += loss_mae.item()
            loss_recon_accum += loss_reg.item()
            loss_d2_accum += out6.item()
            loss_dist_accum += out3.item()
            loss_ipr_accum += out2.item()

        return (
            loss_accum / (step + 1),
            loss_recon_accum / (step + 1),
            loss_d2_accum / (step + 1),
            loss_dist_accum / (step + 1),
            loss_ipr_accum / (step + 1),
        )




def main():
    parser = argparse.ArgumentParser(description='OrbECG training')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vt_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_factor', type=float, default=1.0)
    parser.add_argument('--lr_decay_step_size', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--molecule', type=str, default='BT')
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--train_size', type=int, default=8000)
    parser.add_argument('--valid_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--flag_cg_global', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    global device, flag_cg_global, wfn_var_global
    device = torch.device(args.device)
    flag_cg_global = int(args.flag_cg_global)

    dataset = BT3DDataset(root=args.data_root, molecule=args.molecule)
    data_size = min(args.max_samples, len(dataset))
    split_idx = dataset.get_idx_split(data_size, train_size=args.train_size, valid_size=args.valid_size, seed=args.seed)

    train_ids = trim_to_full_batches(split_idx['train'], args.batch_size)
    valid_ids = trim_to_full_batches(split_idx['valid'], args.vt_batch_size)
    test_ids = trim_to_full_batches(split_idx['test'], args.vt_batch_size)

    train_dataset = dataset[train_ids]
    valid_dataset = dataset[valid_ids]
    test_dataset = dataset[test_ids]

    print('train, validation, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

    model = OrbECG(hidden_channels=args.hidden_channels)
    model = model.to(device)
    model.test = 0

    wfn_var_global = compute_wfn_variance(train_dataset)
    if model.num_orbs < 8:
        diff = (8 - model.num_orbs) // 2
        wfn_var_global = wfn_var_global[diff:-diff]

    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator() if ThreeDEvaluator is not None else None

    train_loader_init = DataLoader(train_dataset, args.batch_size, shuffle=True)
    num_sigma_model = len(model.sigma_param)
    num_p_model = model.num_p
    num_central_model = model.num_atoms + len(model.edge1)
    ovl_avg = torch.zeros((num_p_model * num_sigma_model, num_p_model * num_sigma_model), device=device)
    ct_loader = 0

    with torch.no_grad():
        for _, data in enumerate(train_loader_init):
            data = data.to(device)
            pos_init = data.pos.view(-1, 16, 3)
            if flag_cg_global == 1:
                pos_init = model.cg_proj @ pos_init
            ovl_batch, _ = model.gaussian_graph_overlap_degen(pos_init)
            ovl_avg = ovl_avg + torch.mean(ovl_batch, 0).reshape(
                num_sigma_model,
                num_p_model + num_central_model,
                num_sigma_model,
                num_p_model + num_central_model,
            )[:, num_central_model:, :, num_central_model:].reshape(num_sigma_model * num_p_model, num_sigma_model * num_p_model)
            ct_loader += 1

    ovl_avg = ovl_avg / ct_loader
    val_avg, vec_avg = torch.linalg.eigh(ovl_avg)
    vec_init = torch.zeros((num_sigma_model, model.num_basis // num_sigma_model, model.num_beads), device=device)
    vec_avg = vec_avg[:, -model.num_beads:]
    vec_avg = 0.01 * vec_avg / torch.std(vec_avg, 0).unsqueeze(0)
    vec_init[:, num_central_model:num_central_model + num_p_model, :] = vec_avg.reshape(num_sigma_model, num_p_model, model.num_beads)
    vec_init = vec_init.reshape(model.num_basis, model.num_beads).T.unsqueeze(0)
    model.cg_ref.data = vec_init + 0.005 * torch.randn_like(vec_init)

    trainer = OrbECGTrainer(device)
    trainer.train_loop(
        device,
        train_dataset,
        valid_dataset,
        test_dataset,
        model,
        loss_func,
        evaluation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        vt_batch_size=args.vt_batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=args.weight_decay,
    )


if __name__ == '__main__':
    main()
