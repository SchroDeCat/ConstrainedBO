import os
import sys
import warnings
import tqdm
import random
import numpy as np
from dataclasses import dataclass

import torch
from torch.quasirandom import SobolEngine
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

import scbo_botorch

import matplotlib.pyplot as plt

sys.path.append(f"{os.path.dirname(__file__)}/..")
from utils import save_res
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
n_repeat = 2
fix_seed = True
n_iter = 100
train_iter = 50
lr = 1e-4
exact_gp = False
name = 'ackly-3c'
plot_results = True
save_results = True
save_path = f"{os.path.dirname(__file__)}/../../res/scbo"

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

def c1(x):  # Equivalent to enforcing that x[0] >= 0
    return -x[0]


def c2(x):  # Equivalent to enforcing that x[1] >= 0
    return -x[1]

def c3(x):  # Equivalent to enforcing that x[1] >= 0
    return -x[2] + x[3]

# for c_tensor only
def c_fun_1(x):  # Equivalent to enforcing that x[0] >= 0
    return x[0]

def c_fun_2(x):  # Equivalent to enforcing that x[0] >= 0
    return x[1]

def c_fun_3(x):  # Equivalent to enforcing that x[0] >= 0
    return x[2] - x[3]

c_fun_list = [c_fun_1, c_fun_2, c_fun_3]
c_num = len(c_fun_list)
c_tensor_list = [torch.tensor([c_fun_list[c_idx](unnormalize(x, (lb, ub))) for x in x_tensor], dtype=dtype, device=device).unsqueeze(-1) for c_idx in range(c_num)]
constraint_threshold_list = torch.zeros(c_num)
constraint_confidence_list = torch.ones(c_num) * 0.5
feasible_filter = feasible_filter_gen(c_tensor_list, constraint_threshold_list)

init_feasible_reward = y_tensor[:n_init][feasible_filter[:n_init]]
max_reward = init_feasible_reward.max().item()
max_global = y_tensor[feasible_filter].max().item()
reg_record = np.zeros([n_repeat, n_iter])
print(f"Feasible Y {init_feasible_reward}")
print(f"Before Optimization the best value is: {max_reward:.4f} / global opt {max_global:.4f} := regret {max_global - max_reward:.4f} ")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for rep in tqdm.tqdm(range(n_repeat), desc=f"Experiment Rep"):
        # set seed
        if fix_seed:
            _seed = rep * 20 + n_init
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        scbo = scbo_botorch.SCBO(fun, [c1, c2, c3], dim=dim, lower_bound=lb, upper_bound=ub, train_times=train_iter, lr=lr,
                        batch_size = batch_size, n_init=n_init, train_X = x_tensor[:n_init], dk=not exact_gp)

        rewards = scbo.optimization(n_iter=n_iter//batch_size, x_tensor=x_tensor)
        regrets = max_global - rewards
        reg_record[rep] = regrets[-n_iter:]

reg_output_record = reg_record.mean(axis=0)
_file_prefix = f"Figure_{name}{'-Exact' if exact_gp else ''}-RI{1}"
_file_postfix = f"-{'ts'}-R{n_repeat}-P{1}-T{n_iter}_I{1}_L{int(-np.log10(lr))}-TI{train_iter}"
if plot_results:
    fig = plt.figure()
    plt.plot(reg_output_record)
    plt.ylabel("regret")
    plt.xlabel("Iteration")
    plt.title(f'simple regret for {name}')
    _path = f"{save_path}/Regret-{_file_prefix}{_file_postfix}"
    plt.savefig(f"{_path}.png")

if save_results:
    save_res(save_path=save_path, name=f"Regret-{_file_prefix}-", res=reg_record, n_repeat=n_repeat, num_GP=1, n_iter=n_iter, train_iter=train_iter,
        init_strategy='none', cluster_interval=1, acq='ts', ucb_strategy="exact", ci_intersection=False, verbose=False,)


max_reward = rewards.max().item()
print(f"With constraints, the best value we found is: {max_reward:.4f} / global opt {max_global:.4f} := regret {max_global - max_reward:.4f} ")



