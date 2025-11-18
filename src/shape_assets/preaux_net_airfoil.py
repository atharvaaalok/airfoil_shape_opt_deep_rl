from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreAuxNet_Airfoil(nn.Module):
    def __init__(self):
        super().__init__()

        coord_filepath = Path(__file__).resolve().parent / '../cfd_utils/airfoil_utils/naca0010.dat'
        closed_manifold = np.loadtxt(coord_filepath)
        self.closed_manifold = torch.from_numpy(closed_manifold).to(torch.float32)

        self.closed_transform = nn.Identity()

        self.latent_dim = 0

    
    def forward(self, T = None, code = None):
        t = self.closed_manifold
        X = self.closed_transform(t)

        return X