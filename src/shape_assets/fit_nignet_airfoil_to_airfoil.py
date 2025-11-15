from pathlib import Path

from .nignet_normalized import NIGnetNorm_Airfoil
from geodiff.utils import sample_T
import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import automate_training, plot_curves, normalize


# Set random seed for reproducibility
torch.manual_seed(0)


# Get points on target curve
airfoil_file_path = Path(__file__).resolve().parent / '../cfd_utils/airfoil_utils/naca0010.dat'
Xt = np.loadtxt(airfoil_file_path)
Xt = torch.from_numpy(Xt).to(torch.float32)

# Move centroid to origin and set x range to [-1, 1]
Xt = normalize(Xt)


# Get input parameter discretization for t in [0, 1]
T = sample_T(geometry_dim = 2, num_pts = Xt.shape[0])


# Create a NIGnet object
nig_net = NIGnetNorm_Airfoil(layer_count = 3, act_fn = torch.nn.Tanhshrink)


# Fit the NIGnet to the target airfoil using the provided automate training function
automate_training(
    model = nig_net, loss_fn = torch.nn.MSELoss(), X_train = T, Y_train = Xt,
    learning_rate = 0.01, weight_decay = 1e-6, epochs = 100000, print_cost_every = 20000
)


total_params = sum(p.numel() for p in nig_net.parameters())
print(f"Total parameters: {total_params}")


# Print all the parameters to see their scale
for name, param in nig_net.named_parameters():
    print(f"\n{name}:\n{param}")

# Visualize the fit
Xc = nig_net(T)
plot_curves(Xc, Xt)


# Save the model
model_save_path = (Path(__file__).resolve().parent /
                   'models/nignet_airfoil_normalized_fit_to_naca0010.pth')
torch.save(nig_net.state_dict(), model_save_path)