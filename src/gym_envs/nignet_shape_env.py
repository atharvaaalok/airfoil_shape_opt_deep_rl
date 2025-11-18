import copy

from geodiff.utils import sample_T
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..cfd_utils.aerodynamics import compute_L_by_D


class NIGnetShapeEnv(gym.Env):

    metadata = {'render_modes': []}

    def __init__(
        self,
        *,
        nig_net: nn.Module,
        action_sigma: float,
        max_episode_steps: int,
        non_convergence_reward: float,
        Mach_num: float,
        Re: float,
        xfoil_max_iter: float,
        reward_delta: bool = False,
    ) -> None:
        """Initialize the NIGnet shape environment by specifying the observation and action spaces
        and the initial internal state.

        Args:
            nig_net: A NIGnet to be used as the starting state for the network.
            action_sigma: The variance of the action space.
            max_episode_steps: Maximum number of steps to be taken in the environment.
            non_convergence_reward: The reward to be given on Xfoil failure.
        """
        super().__init__()

        self.nig_net = copy.deepcopy(nig_net)
        self.action_sigma = action_sigma
        self.max_episode_steps = max_episode_steps
        self.non_convergence_reward = non_convergence_reward
        self.Mach_num = Mach_num
        self.Re = Re
        self.xfoil_max_iter = xfoil_max_iter
        self.reward_delta = reward_delta

        # Track previous L/D for delta rewards
        self.prev_L_by_D = None


        # Flatten parameter vector once to fix observation and action dimensions
        with torch.no_grad():
            self._param_template = parameters_to_vector(self.nig_net.parameters()).detach().clone()
        self.num_params = self._param_template.numel()


        # Define observation space
        low = -np.inf * np.ones(self.num_params, dtype = np.float32)
        high = np.inf * np.ones(self.num_params, dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

        # Define action space to be between [-1, 1], they will be scaled by action_sigma during
        # step(), this means action sigma will act as our hyperparameter to tune
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (self.num_params,),
                                       dtype = np.float32)
        
        # Internal state
        self._step_count = 0
        self.state = self._param_template.cpu().numpy().astype(np.float32)
    

    def _get_obs(self):
        return self.state.copy()


    def _get_info(self, reward: float | None = None):
        return {'steps': self._step_count, 'reward': reward}
    

    def reset(self, seed: int | None = None):
        super().reset(seed = seed)

        # Reinitialize nig_net to the network parameters that represent starting airfoil
        vector_to_parameters(self._param_template, self.nig_net.parameters())

        self.state = self._param_template.cpu().numpy().copy().astype(np.float32)
        self._step_count = 0

        # (Re)initialize previous L/D for delta rewards
        if self.reward_delta:
            num_pts = 251
            T = sample_T(geometry_dim = 2, num_pts = num_pts)
            X = self.nig_net(T).detach().cpu().numpy()
            L_by_D = compute_L_by_D(X = X, M = self.Mach_num, Re = self.Re,
                                    max_iter = self.xfoil_max_iter)
            self.prev_L_by_D = L_by_D
        else:
            self.prev_L_by_D = None

        observation = self._get_obs()
        info = self._get_info(reward = None)
        return observation, info


    def step(self, action: np.ndarray):
        self._step_count += 1

        perturb = self.action_sigma * torch.from_numpy(action)
        new_params = torch.from_numpy(self.state) + perturb
        # Set nig_net parameters to the new parameter state
        vector_to_parameters(new_params, self.nig_net.parameters())

        self.state = new_params.cpu().numpy().astype(np.float32)

        # Calculate reward
        num_pts = 251
        T = sample_T(geometry_dim = 2, num_pts = num_pts)
        X = self.nig_net(T).detach().cpu().numpy()
        L_by_D = compute_L_by_D(X = X, M = self.Mach_num, Re = self.Re,
                                max_iter = self.xfoil_max_iter)
        if L_by_D is None:
            # Xfoil failure -> fixed penalty, regardless of reward_delta
            reward = self.non_convergence_reward
        else:
            if self.reward_delta:
                reward = L_by_D - self.prev_L_by_D
            else:
                reward = L_by_D
        
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        observation = self._get_obs()
        info = self._get_info(reward)

        return observation, reward, terminated, truncated, info


    def render(self):
        pass


    def close(self):
        pass