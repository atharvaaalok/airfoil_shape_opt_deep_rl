from pathlib import Path

from geodiff.nice import NICE
from geodiff.template_architectures import ResMLP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preaux_net_airfoil import PreAuxNet_Airfoil
from .utils import normalize


class NiceNorm_Airfoil(nn.Module):
    def __init__(
        self,
        layer_count,
        coupling_net_layer_count,
        coupling_net_hidden_dim,
        use_batchnormalization = False,
        use_residual_connection = False,
        volume_preserving = False
    ):
        super(NiceNorm_Airfoil, self).__init__()

        self.layer_count = layer_count

        self.preaux_net = PreAuxNet_Airfoil()
        self.coupling_net = ResMLP(input_dim = 1, output_dim = 1,
                                   layer_count = coupling_net_layer_count,
                                   hidden_dim = coupling_net_hidden_dim)

        self.nice = NICE(
            geometry_dim = 2,
            layer_count = layer_count,
            preaux_net = self.preaux_net,
            coupling_net = self.coupling_net,
            use_batchnormalization = use_batchnormalization,
            use_residual_connection = use_residual_connection,
            volume_preserving = volume_preserving,
        )
    
    def forward(self, T):
        X = self.nice(T)

        # Center and scale
        X = normalize(X)
        
        return X