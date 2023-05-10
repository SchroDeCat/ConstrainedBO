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

from SCBO.scbo_botorch import SCBO

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

batch_size = 4
n_init = 2 * dim
max_cholesky_size = float("inf")  # Always use Cholesky

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


scbo = SCBO(fun, [c1, c2], dim=dim, lower_bound=lb, upper_bound=ub, 
                 batch_size = batch_size, n_init=n_init,)

rewards = scbo.optimization(n_iter=100)
# Valid samples must have BOTH c1 <= 0 and c2 <= 0


print(f"With constraints, the best value we found is: {rewards.max().item():.4f}")



