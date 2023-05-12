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

import scbo_botorch
# from .scbo_botorch import SCBO

def feasible_filter_gen(c_tensor_list, threshold_list):
    n_pts = c_tensor_list[0].size(0)
    c_num = len(c_tensor_list)
    feasible_filter = torch.tensor([True for _ in range(n_pts)]).squeeze()
    
    for c_idx in range(c_num):
        _tmp_filter = c_tensor_list[c_idx] >= threshold_list[c_idx]
        feasible_filter = feasible_filter.logical_and(_tmp_filter.squeeze())
    return feasible_filter

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Here we define the example 20D Ackley function
fun = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
fun.bounds[0, :].fill_(-5)
fun.bounds[1, :].fill_(10)
dim = fun.dim
lb, ub = fun.bounds

batch_size = 1
# n_init = 2 * dim
n_init = 10
n_pts = 20000
max_cholesky_size = float("inf")  # Always use Cholesky
sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
x_tensor = sobol.draw(n=n_pts).to(device=device, dtype=dtype)
y_tensor = torch.tensor([fun(unnormalize(x, (lb, ub))) for x in x_tensor])



# ### Defining two simple constraint functions
# 
# #### We'll use two constraints functions: c1 and c2 
# We want to find solutions which maximize the above Ackley objective subject to the constraint that 
# c1(x) <= 0 and c2(x) <= 0 
# Note that SCBO expects all constraints to be of the for c(x) <= 0, so any other desired constraints must be modified to fit this form. 
# 
# Note also that while the below constraints are very simple functions, the point of this tutorial is to show how to use SCBO, and this same implementation could be applied in the same way if c1, c2 were actually complex black-box functions. 
# 
def c1(x):  # Equivalent to enforcing that x[0] >= 0
    return -x[0]


def c2(x):  # Equivalent to enforcing that x[1] >= 0
    return -x[1]

# for c_tensor only
def c_fun_1(x):  # Equivalent to enforcing that x[0] >= 0
    return x[0]

def c_fun_2(x):  # Equivalent to enforcing that x[0] >= 0
    return x[1]

c_fun_list = [c_fun_1, c_fun_2]
c_num = len(c_fun_list)
c_tensor_list = [torch.tensor([c_fun_list[c_idx](unnormalize(x, (lb, ub))) for x in x_tensor], dtype=dtype, device=device).unsqueeze(-1) for c_idx in range(c_num)]
constraint_threshold_list = torch.zeros(c_num)
constraint_confidence_list = torch.ones(c_num) * 0.5
feasible_filter = feasible_filter_gen(c_tensor_list, constraint_threshold_list)

init_feasible_reward = y_tensor[:n_init][feasible_filter[:n_init]]
max_reward = init_feasible_reward.max().item()
max_global = y_tensor[feasible_filter].max().item()
print(f"Feasible Y {init_feasible_reward}")
print(f"Before Optimization the best value is: {max_reward:.4f} / global opt {max_global:.4f} := regret {max_global - max_reward:.4f} ")

scbo = scbo_botorch.SCBO(fun, [c1, c2], dim=dim, lower_bound=lb, upper_bound=ub, 
                 batch_size = batch_size, n_init=n_init, train_X = x_tensor[:n_init])

rewards = scbo.optimization(n_iter=100//batch_size, x_tensor=x_tensor)
# Valid samples must have BOTH c1 <= 0 and c2 <= 0

max_reward = rewards.max().item()
print(f"With constraints, the best value we found is: {max_reward:.4f} / global opt {max_global:.4f} := regret {max_global - max_reward:.4f} ")



