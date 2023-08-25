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

from src.opt import baseline_cbo_m
from src.utils import sample_pts, feasible_filter_gen
# from .scbo_botorch import SCBO

warnings.filterwarnings("ignore")

device = torch.device('cpu')
dtype = torch.float

SMOKE_TEST = os.environ.get("SMOKE_TEST")

dim=20
# Here we define the example 20D Ackley function
fun = Ackley(dim=dim, negate=True).to(dtype=dtype, device=device)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)
dim = fun.dim
lb, ub = fun.bounds
n_pts = 20000

def c_fun_1(x):  # Equivalent to enforcing that x[0] >= 0
    return x[0]

def c_fun_2(x):  # Equivalent to enforcing that x[1] >= 0
    return x[1]

def c_fun_3(x):  # Equivalent to enforcing that x[2]-x[3] >= 0
    return x[2] - x[3]

c_fun_list = [c_fun_1, c_fun_2, c_fun_3]
c_num = len(c_fun_list)

n_init = 10
# n_iter = 200
n_iter=20
train_times = 50
# n_repeat = 10
n_repeat=2
# acq = "qei"
# acq = 'cmes-ibo'
interpolate=True
acq = 'random'
max_cholesky_size = float("inf")  # Always use Cholesky

print(f"acq {acq} n repeat {n_repeat}")


x_tensor = sample_pts(lb, ub, n_pts, dim=dim).to(dtype=dtype, device=device)
y_tensor = torch.tensor([fun(x) for x in x_tensor], dtype=dtype, device=device).unsqueeze(-1)
c_tensor_list = [torch.tensor([c_fun_list[c_idx](x) for x in x_tensor], dtype=dtype, device=device).unsqueeze(-1) for c_idx in range(c_num)]
constraint_threshold_list = torch.zeros(c_num)
constraint_confidence_list = torch.ones(c_num) * 0.5
feasible_filter = feasible_filter_gen(c_tensor_list, constraint_threshold_list)

print(f"initial reward {y_tensor[:n_init][feasible_filter[:n_init]].squeeze()} while global max {y_tensor[feasible_filter].max().item()}")

regret = baseline_cbo_m(x_tensor, y_tensor, c_tensor_list, 
                        constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
                        n_init=10, n_repeat=n_repeat, train_times=train_times, n_iter=n_iter,
                        regularize=False, low_dim=True,
                        spectrum_norm=False, retrain_interval=1, acq=acq, 
                        verbose=True, lr=1e-4, name="test-baseline", 
                        return_result=True, retrain_nn=True,
                        plot_result=True, save_result=True, save_path=f'./res/baseline/{acq}', 
                        fix_seed=True,  pretrained=False, ae_loc=None, 
                        exact_gp=False, constrain_noise=True, interpolate=interpolate)

# regret = baseline_cbo_m(x_tensor, y_tensor, c_tensor_list, constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
#             n_init=10, n_repeat=n_repeat, train_times=train_times, regularize=False, low_dim=True,
#             spectrum_norm=False, retrain_interval=1, n_iter=n_iter, filter_interval=1, acq="ci", 
#             ci_intersection=False, verbose=True, lr=1e-4, name="test", return_result=True, retrain_nn=True,
#             plot_result=True, save_result=True, save_path='./res', fix_seed=True,  pretrained=False, ae_loc=None, 
#             _minimum_pick = 10, _delta = 0.01, beta=0, filter_beta=0, exact_gp=True, constrain_noise=True, local_model=False)

print(f"With constraints, the minimum regret we found is: {regret.min(axis=-1)}")