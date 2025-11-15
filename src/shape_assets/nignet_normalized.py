from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalize


class NIGnetNorm(nn.Module):
    def __init__(self, layer_count, act_fn, skip_connections = True):
        super(NIGnetNorm, self).__init__()
        
        self.layer_count = layer_count
        self.skip_connections = skip_connections

        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        Linear_class = nn.Linear

        self.linear_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()


        for i in range(layer_count):
            self.linear_layers.append(Linear_class(2, 2))
            self.act_layers.append(act_fn())
        
        self.final_linear = Linear_class(2, 2, bias = False)
    

    def forward(self, T):
        t = T
        X = self.closed_transform(t)

        for i, (linear_layer, act_layer) in enumerate(zip(self.linear_layers, self.act_layers)):
            X = linear_layer(X)

            if self.skip_connections:
                residual = X
            
            X = act_layer(X)

            if self.skip_connections:
                X = (X + residual) / 2.0
        
        X = self.final_linear(X)

        
        # Center and scale
        X = normalize(X)

        return X


class NIGnetNorm_Airfoil(nn.Module):
    def __init__(self, layer_count, act_fn, skip_connections = True):
        super(NIGnetNorm_Airfoil, self).__init__()
        
        self.layer_count = layer_count
        self.skip_connections = skip_connections

        coord_filepath = Path(__file__).resolve().parent / '../cfd_utils/airfoil_utils/naca0010.dat'
        closed_manifold = np.loadtxt(coord_filepath)
        self.closed_manifold = torch.from_numpy(closed_manifold).to(torch.float32)

        self.closed_transform = nn.Identity()

        Linear_class = nn.Linear

        self.linear_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()


        for i in range(layer_count):
            self.linear_layers.append(Linear_class(2, 2))
            self.act_layers.append(act_fn())
        
        self.final_linear = Linear_class(2, 2, bias = False)
    

    def forward(self, T):
        t = self.closed_manifold
        X = self.closed_transform(t)

        for i, (linear_layer, act_layer) in enumerate(zip(self.linear_layers, self.act_layers)):
            X = linear_layer(X)

            if self.skip_connections:
                residual = X
            
            X = act_layer(X)

            if self.skip_connections:
                X = (X + residual) / 2.0
        
        X = self.final_linear(X)

        
        # Center and scale
        X = normalize(X)

        return X