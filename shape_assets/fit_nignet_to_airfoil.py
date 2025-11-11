from pathlib import Path

from nignet_normalized import NIGnetNorm
from geodiff.utils import sample_T
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import automate_training, plot_curves, normalize


# Set random seed for reproducibility
torch.manual_seed(0)


# Get points on target curve
airfoil_file_path = Path('../cfd_utils/airfoil_utils/naca0010.dat')
Xt = np.loadtxt(airfoil_file_path)
Xt = torch.from_numpy(Xt).to(torch.float32)

# Move centroid to origin and set x range to [-1, 1]
Xt = normalize(Xt)


# Get input parameter discretization for t in [0, 1]
T = sample_T(geometry_dim = 2, num_pts = Xt.shape[0])


# Create a NIGnet object
nig_net = NIGnetNorm(layer_count = 2, act_fn = torch.nn.Tanh)


# Fit the NIGnet to the target airfoil using the provided automate training function
automate_training(
    model = nig_net, loss_fn = torch.nn.MSELoss(), X_train = T, Y_train = Xt,
    learning_rate = 0.01, weight_decay = 1e-5, epochs = 100000, print_cost_every = 20000
)


total_params = sum(p.numel() for p in nig_net.parameters())
print(f"Total parameters: {total_params}")


# Visualize the fit
Xc = nig_net(T)
plot_curves(Xc, Xt)


# Save the model
torch.save(nig_net.state_dict(), f'models/nignet_normalized_fit_to_naca0010.pth')