"""
Utilities to support exps
"""

import gpytorch
import os
import random
import torch
import tqdm
import time
import matplotlib
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import itertools

from sparsemax import Sparsemax
from scipy.stats import ttest_ind
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize
from botorch.models.transforms.outcome import Standardize

from ..models import beta_CI



def _random_seed_gen(size:int=100):
    np.random.seed(0)
    return np.random.choice(10000, size)

def sample_pts(lb, ub, n_pts:int=10, dim:int=2, seed:int=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    x = sobol.draw(n=n_pts)
    return unnormalize(x, (lb, ub))

def model_list_CI(model_list, x_tensor, DEVICE):
    lcb_list, ucb_list = [], []
    for model in model_list:
        lcb_list_tmp, ucb_list_tmp = model.CI(x_tensor.to(DEVICE))
        lcb_list.append(lcb_list_tmp), ucb_list.append(ucb_list_tmp)
    return lcb_list, ucb_list 

def intersecting_ROI_globe(max_all_lcb, min_all_ucb, roi_lcb, roi_ucb, roi_beta, roi_filter, adaptive_scaling=False):
    roi_lcb_scaled, roi_ucb_scaled = beta_CI(roi_lcb, roi_ucb, roi_beta)   
    if adaptive_scaling:
        _lcb_scaling_factor, _ucb_scaling_factor = max_all_lcb[roi_filter].max()/ roi_lcb_scaled[roi_filter].max(), min_all_ucb[roi_filter].max() / roi_lcb_scaled[roi_filter].max()
    else:
        _lcb_scaling_factor, _ucb_scaling_factor = 1, 1

    _max_all_lcb, _min_all_ucb = torch.max(max_all_lcb, roi_lcb_scaled * _lcb_scaling_factor), torch.min(min_all_ucb, roi_ucb_scaled * _ucb_scaling_factor) 
    max_all_lcb[roi_filter], min_all_ucb[roi_filter] = _max_all_lcb[roi_filter], _min_all_ucb[roi_filter]
    return max_all_lcb, min_all_ucb, roi_lcb_scaled, roi_ucb_scaled

def feasible_filter_gen(c_tensor_list, threshold_list):
    n_pts = c_tensor_list[0].size(0)
    c_num = len(c_tensor_list)
    feasible_filter = torch.tensor([True for _ in range(n_pts)]).squeeze()
    
    for c_idx in range(c_num):
        _tmp_filter = c_tensor_list[c_idx] >= threshold_list[c_idx]
        feasible_filter = feasible_filter.logical_and(_tmp_filter.squeeze())
    return feasible_filter

# def beta_CI(lcb, ucb, beta):
#     """Lower then upper"""
#     _ucb_scaled = (ucb - lcb) / 4 * (beta-2) + ucb
#     _lcb_scaled = (ucb - lcb) / 4 * (2-beta) + lcb
#     return _lcb_scaled, _ucb_scaled

def _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, cluster_interval, acq, lr, train_iter, ucb_strategy, ci_intersection=False):
    return f"{save_path}/OL-{name}-{init_strategy}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_iter}-US{ucb_strategy}{'-sec' if ci_intersection else ''}"

def save_res(save_path, name, res, n_repeat=2, num_GP=2, n_iter=40, init_strategy:str="kmeans", cluster_interval:int=1, acq:str='ts', ucb_strategy="exact", lr:float=1e-3, train_iter:int=10, ci_intersection=True, verbose=True):
    file_path = _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, cluster_interval, acq, lr, train_iter, ucb_strategy, ci_intersection)
    np.save(file_path, res)
    if verbose:
        print(f"File stored to {file_path}")

def load_res(save_path, name, n_repeat=2, num_GP=2, n_iter=40, init_strategy:str="kmeans",  cluster_interval:int=1, acq:str='ts', ucb_strategy="exact", lr:float=1e-3,  train_iter:int=10, ci_intersection=True, verbose=True):
    file_path = _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, cluster_interval, acq, lr, train_iter, ucb_strategy)
    file_path = f"{file_path}.npy"
    data = np.load(file_path)
    if verbose:
        print(f"Data {data.shape()} loaded from {file_path}")
    return data


