import math
import os
import warnings
from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

from src.opt import cbo
from src.utils import sample_pts
# from .scbo_botorch import SCBO

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

SMOKE_TEST = os.environ.get("SMOKE_TEST")

dim=20
# Here we define the example 20D Ackley function
fun = Ackley(dim=dim, negate=True).to(dtype=dtype, device=device)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)
dim = fun.dim
lb, ub = fun.bounds
n_pts = 10000

def c_fun(x):  # Equivalent to enforcing that x[0] >= 0
    return x[0]

n_init = 10
n_iter = 100
train_times = 50
n_repeat = 30
max_cholesky_size = float("inf")  # Always use Cholesky



x_tensor = sample_pts(lb, ub, n_pts, dim=dim).to(dtype=dtype, device=device)
y_tensor = torch.tensor([fun(x) for x in x_tensor], dtype=dtype, device=device).unsqueeze(-1)
c_tensor = torch.tensor([c_fun(x) for x in x_tensor], dtype=dtype, device=device).unsqueeze(-1)
init_feasible_filter = c_tensor[:n_init] >= 0
print(f"initial reward {y_tensor[:n_init][init_feasible_filter]} while global max {y_tensor.max().item()}")

regret = cbo(x_tensor, y_tensor, c_tensor, constraint_threshold=0, constraint_confidence=.5,
            n_init=10, n_repeat=n_repeat, train_times=train_times, regularize=False, low_dim=True,
            spectrum_norm=False, retrain_interval=1, n_iter=n_iter, filter_interval=1, acq="ci", 
            ci_intersection=True, verbose=True, lr=1e-4, name="test", return_result=True, retrain_nn=True,
            plot_result=True, save_result=True, save_path='./res', fix_seed=True,  pretrained=False, ae_loc=None, 
            _minimum_pick = 10, _delta = 0.2, beta=0, filter_beta=0.5, exact_gp=False, constrain_noise=False, local_model=False)

regret = cbo(x_tensor, y_tensor, c_tensor, constraint_threshold=0, constraint_confidence=.5,
            n_init=10, n_repeat=n_repeat, train_times=train_times, regularize=False, low_dim=True,
            spectrum_norm=False, retrain_interval=1, n_iter=n_iter, filter_interval=1, acq="ci", 
            ci_intersection=True, verbose=True, lr=1e-4, name="test", return_result=True, retrain_nn=True,
            plot_result=True, save_result=True, save_path='./res', fix_seed=True,  pretrained=False, ae_loc=None, 
            _minimum_pick = 10, _delta = 0.2, beta=0, filter_beta=0.5, exact_gp=True, constrain_noise=False, local_model=False)

print(f"With constraints, the minimum regret we found is: {regret.min(axis=-1)}")