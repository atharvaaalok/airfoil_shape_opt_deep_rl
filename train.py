import argparse
import os
from pathlib import Path
import random
import sys

from geodiff.utils import sample_T
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.cfd_utils.aerodynamics import compute_L_by_D
from src.gym_envs.nignet_shape_env import NIGnetShapeEnv
from src.shape_assets.nignet_normalized import NIGnetNorm, NIGnetNorm_Airfoil
from src.shape_assets.utils import plot_curves


# Set random seed for reproducibility
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


base_log_dir = Path('logs')


# Specify gym environment properties
airf_to_airf = False
config = {
    'action_sigma': 0.001,
    'max_episode_steps': 16,
    'non_convergence_reward': -50.0,
    'num_env': 16,
    'total_training_timesteps': 2**10,
    'n_steps_per_env': 8,
    'batch_size': 64,
    'Mach_num': 0.1,
    'Re': 1e6,
    'nignet_model': f'nignet_{"airfoil_" if airf_to_airf else ''}normalized_fit_to_naca0010.pth',
    'policy_learning_rate': 0.0003,
    'NIGnet_class': NIGnetNorm if not airf_to_airf else NIGnetNorm_Airfoil
}

NIGnet_class = config['NIGnet_class']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0, help = 'Random seed for reproducibility')
    parser.add_argument('--log-name', type = str, default = 'default_log',
                        help = 'Name of the log directory under logs/.')
    args = parser.parse_args()
    seed = args.seed
    log_name = args.log_name

    # Set seed and log name for the experiment
    set_global_seed(seed)
    log_path = base_log_dir / log_name / f'seed_{seed}'


    print('Configuration:\n' + '-'*50)
    for config_key, config_value in config.items():
        print(f'{config_key} = {config_value}')

    # Import the NIGnet model that we trained to fit the airfoil
    nig_net = NIGnet_class(layer_count = 3, act_fn = torch.nn.Tanhshrink)
    nig_net.load_state_dict(torch.load(
        Path(f'./src/shape_assets/models/{config['nignet_model']}'), weights_only = True
    ))

    # Print the model parameters to get an idea of the scale of values
    print('\nNIGnet Parameters:\n' + '-'*50)
    print(f'nignet_model = {config['nignet_model']}')
    for name, param in nig_net.named_parameters():
        print(f"\n{name}:\n{param}")


    print('\nCreating Environment:\n' + '-'*50)
    # Create a gym environment
    def make_nignet_env(rank: int, seed: int = 0):
        def _init():
            env = NIGnetShapeEnv(
                nig_net = nig_net,
                action_sigma = config['action_sigma'],
                max_episode_steps = config['max_episode_steps'],
                non_convergence_reward = config['non_convergence_reward'],
                Mach_num = config['Mach_num'],
                Re = config['Re']
            )
            env.reset(seed = seed + rank)
            return env

        return _init

    # Create vectorized gym environments
    train_env = SubprocVecEnv(
        [make_nignet_env(rank = i, seed = seed) for i in range(config['num_env'])]
    )
    train_env = VecMonitor(train_env)


    # Setup logging
    tensorboard_log_file_location = log_path
    # Setup logger
    training_logger = configure(str(tensorboard_log_file_location),
                                ['stdout', 'tensorboard', 'log'])


    # Create RL model
    model = PPO(policy = 'MlpPolicy', learning_rate = config['policy_learning_rate'],
                env = train_env, verbose = 1, tensorboard_log = tensorboard_log_file_location,
                n_steps = config['n_steps_per_env'], batch_size = config['batch_size'], seed = seed)
    
    model.set_logger(training_logger)


    print('\nModel Training:\n' + '-'*50)
    # Start training
    model.learn(total_timesteps = config['total_training_timesteps'], progress_bar = True)


    print('\nModel Evaluation:\n' + '-'*50)
    # Policy evaluation
    test_rank = 100
    test_env_init = make_nignet_env(rank = test_rank, seed = seed)
    test_env = test_env_init()
    observation, info = test_env.reset()
    done = False

    print(f'{"Step":>4} {"L_by_D":>10}')
    step = 0
    while not done:
        step += 1
        # Get action from the trained policy
        action, _state = model.predict(observation, deterministic = True)

        # Step the environment
        observation, reward, terminated, truncated, info = test_env.step(action)
        done = truncated
        print(f'{step:4d} {reward:10.4f}')


    # Convert observation to NIGnet parameters
    test_nig_net = NIGnet_class(layer_count = 3, act_fn = nn.Tanhshrink)
    vector_to_parameters(torch.from_numpy(observation), test_nig_net.parameters())

    # Calculate the L-by-D ratio of the final shape and plot the airfoil produced
    num_pts = 251
    T = sample_T(geometry_dim = 2, num_pts = num_pts)
    X_test = test_nig_net(T)
    L_by_D = compute_L_by_D(X = X_test.detach().cpu().numpy(), M = config['Mach_num'],
                            Re = config['Re'])

    X_original = nig_net(T)
    fig_filename = log_path / 'final_shape.svg'
    plot_curves(X_original, X_test, filename = fig_filename, show_plot = False)
    L_by_D = f'{L_by_D:7.3f}' if L_by_D is not None else str(L_by_D)
    print(f'\n\nFinal L_by_D of trained policy: {L_by_D}')
    print(f'Last reward: {reward:7.3f}')