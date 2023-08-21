'''
One step CBO
'''
import warnings
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
import torch
import gpytorch
import random


# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].

from src.opt import cbo_multi_nontest
from src.utils import feasible_filter_gen

warnings.filterwarnings("ignore")

device = torch.device('cpu')
dtype = torch.float

# load data
sheet_url = "https://docs.google.com/spreadsheets/d/1ClwjfI4SJO2y2XfgyweNgaMO42m8wkKH_0xBTCNVKkw/edit#gid=0"
sheet_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
data = pd.read_csv(sheet_url)
data = data.fillna(0)

fe = data['Fe ratio'].to_numpy()
co = data['Co ratio'].to_numpy()
fwhm_002 = data['002 peak FWHM'].to_numpy()
crystal_size = data['crystal size (nm^3)'].to_numpy()
ratio_002 = data["002/(amorphous+002) ratio"].to_numpy()
n_init = fe.shape[0]
dim=2
train_times = 100
output_scale_constraint=gpytorch.constraints.Interval(0.7,1.2)
norm_factor = 1e4
normalizer = lambda x: x / norm_factor
denormalizer = lambda x: x * norm_factor

# TD: add mesh_grid as candidates
def generate_candidates(interval:int=0.5)->np.ndarray:
    '''
    Generate Candidate for the Alloy
    Input:
        @interval: interval of the grid
    Return:
        Fe, Co, Ni ratio respectively
    '''
    fe, co = np.mgrid[0:1.00001:interval, 0:1.00001:interval]
    ni = 1 - fe - co

    return fe, co, ni

fe_grid, co_gird, ni_grid = generate_candidates(interval=0.1)
ratio_grid, crystal_grid = np.zeros(shape=fe_grid.shape), np.zeros(shape=fe_grid.shape)

def fit_mgrid(x:np.ndarray, y:np.ndarray, x_grid:np.ndarray, y_grid:np.ndarray, obj:np.ndarray, obj_grid:np.ndarray, err_bnd:float=0.05) -> List[np.ndarray]:
    '''
    Given existing obs, fit them into test grid. if not found, discard the values
    E.g. When grid interval is 0.1, to avoid discarding 0.33, 0.33, the error bound should be 0.0424
    '''
    observed = np.zeros(x_grid.shape)
    for _x, _y, _obj in zip(x, y, obj):
        error_2norm = np.sqrt((x_grid - _x) ** 2 + (y_grid - _y) ** 2)
        obj_grid[error_2norm < err_bnd] = _obj
        observed[error_2norm < err_bnd] = 1

    return obj_grid, observed

ratio_grid, _ = fit_mgrid(fe, co, fe_grid, co_gird, ratio_002, ratio_grid)
crystal_grid, observed = fit_mgrid(fe, co, fe_grid, co_gird, crystal_size, crystal_grid)

fe, co, crystal_size, ratio_002, observed, ni = fe_grid.ravel(), co_gird.ravel(), crystal_grid.ravel(), ratio_grid.ravel(), observed.ravel(), ni_grid.ravel()
ni_filter = ni > -1e-10
fe, co, crystal_size, ratio_002, observed = fe[ni_filter], co[ni_filter], crystal_size[ni_filter], ratio_002[ni_filter], observed[ni_filter]

# fit GP and Opt
x_tensor = torch.from_numpy(np.vstack([fe, co])).T.to(dtype=dtype, device=device)
y_tensor = torch.from_numpy(normalizer(crystal_size)).to(dtype=dtype, device=device).unsqueeze(-1)
c_tensor_list = [torch.from_numpy(ratio_002).to(dtype=dtype, device=device).unsqueeze(-1)]
constraint_threshold_list = torch.ones(1) * 0.1 # ratio > 0.1
constraint_confidence_list = torch.ones(1) * 0.5
feasible_filter = feasible_filter_gen(c_tensor_list, constraint_threshold_list)

print(f"initial reward {y_tensor[observed==1][feasible_filter[observed==1]].squeeze()}")


_seed = 2 * 20 + n_init
torch.manual_seed(_seed)
np.random.seed(_seed)
random.seed(_seed)
torch.cuda.manual_seed(_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

result = cbo_multi_nontest(x_tensor, y_tensor, c_tensor_list, constraint_threshold_list=constraint_threshold_list, 
                           constraint_confidence_list=constraint_confidence_list,
                            observed=observed, train_times=train_times, regularize=False, low_dim=True,
                            spectrum_norm=False, acq="ci", 
                            ci_intersection=False, verbose=True, lr=1e-2,
                            pretrained=False, ae_loc=None, 
                            _minimum_pick = 10, _delta = 0.2, filter_beta=0, beta=0, 
                            exact_gp=False, constrain_noise=True, local_model=False,
                            output_scale_constraint=output_scale_constraint)

_f_model, _c_model_list, _cbo_m = result
print(f'Next Pick X: {_cbo_m.init_x[-1]} Y:{_cbo_m.init_y[-1]} C:{_cbo_m.init_c_list[0][-1]}')