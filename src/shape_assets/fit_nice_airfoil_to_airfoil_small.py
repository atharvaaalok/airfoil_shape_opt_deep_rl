from pathlib import Path

from geodiff.utils import sample_T
import matplotlib.pyplot as plt
import numpy as np
import torch

from .nice_normalized import NiceNorm_Airfoil
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


nice = NiceNorm_Airfoil(
    layer_count = 1,
    coupling_net_layer_count = 1,
    coupling_net_hidden_dim = 5,
    use_batchnormalization = False,
    use_residual_connection = False,
    volume_preserving = False
)


# Fit the NIGnet to the target airfoil using the provided automate training function
automate_training(
    model = nice, loss_fn = torch.nn.MSELoss(), X_train = T, Y_train = Xt,
    learning_rate = 0.01, weight_decay = 1e-6, epochs = 10000, print_cost_every = 2000
)


total_params = sum(p.numel() for p in nice.parameters())
print(f"Total parameters: {total_params}")


# Print all the parameters to see their scale
for name, param in nice.named_parameters():
    print(f"\n{name}:\n{param}")

# Visualize the fit
Xc = nice(T)
plot_curves(Xc, Xt, show_plot = True)


# Save the model
model_save_path = (Path(__file__).resolve().parent /
                   'models/nice_airfoil_normalized_fit_to_naca0010_small.pth')
torch.save(nice.state_dict(), model_save_path)