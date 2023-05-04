'''
Full pipeline for constrained BO
'''

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

from ..models import DKL, AE, beta_CI
from ..utils import save_res, load_res, clustering_methods
from .dkbo_olp import DK_BO_OLP
from .dkbo_ae import DK_BO_AE
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
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# STUDY_PARTITION = True
STUDY_PARTITION = False

def cbo(x_tensor, y_tensor, c_tensor, n_init=10, n_repeat=2, train_times=10, beta=2, regularize=True, low_dim=True, spectrum_norm=False, retrain_interval=1,
            n_iter=40, filter_interval=1, acq="ci", ci_intersection=True, verbose=True, lr=1e-2, name="test", return_result=True, retrain_nn=True,
            plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, study_partition=STUDY_PARTITION, _minimum_pick = 10, 
            _delta = 0.2, filter_beta=.05, exact_gp=False, constrain_noise=False):
    
    ####### configurations
    if constrain_noise:
        global_noise_constraint = gpytorch.constraints.Interval(0.1,.6)
        roi_noise_constraint = gpytorch.constraints.Interval(1e-5,0.1)
        name = f"{name}-noise_c"
    else:
        global_noise_constraint = None
        roi_noise_constraint = None

    name = name if low_dim else name+'-hd'
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])
    ratio_record = np.zeros([n_repeat, n_iter])
    max_LUCB_interval_record = np.zeros([n_repeat, 3, n_iter]) # 0 - Global, 1 - ROI, 2 -- intersection

    ####### init dkl and generate ucb for partition
    data_size = x_tensor.size(0)
    util_array = np.arange(data_size)
    if regularize:
        name += "-reg"

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None

    default_beta = beta <= 0
    default_fbeta = filter_beta < 1e-10

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in tqdm.tqdm(range(n_repeat), desc=f"Experiment Rep"):
            # set seed
            if fix_seed:
                _seed = rep * 20 + n_init
                # _seed = 70
                torch.manual_seed(_seed)
                np.random.seed(_seed)
                random.seed(_seed)
                torch.cuda.manual_seed(_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                
            ####### init in each round
            observed = np.zeros(data_size)
            observed[:n_init] = 1
            init_x = x_tensor[:n_init]
            init_y = y_tensor[:n_init]
            # TBD: fix the ae loading here
            _f_model = DKL(init_x, init_y.squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, noise_constraint=global_noise_constraint)
            if regularize:
                _f_model.train_model_kneighbor_collision()
            else:
                _f_model.train_model(verbose=verbose)
            lcb, ucb = _f_model.CI(x_tensor.to(DEVICE))


            ####### each test instance
            iterator = tqdm.tqdm(range(0, n_iter, filter_interval))
            for iter in iterator:
                if default_beta:
                    beta = (2 * np.log((x_tensor.shape[0] * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
                _lcb, _ucb = beta_CI(lcb, ucb, beta)
                max_test_x_lcb, min_test_x_ucb = _lcb.clone(), _ucb.clone()    # actually not taking all historical intersections             
                if default_fbeta:
                    filter_beta = beta
                
                _filter_lcb, _filter_ucb = beta_CI(lcb, ucb, filter_beta)
                ucb_filter = _filter_ucb >= _filter_lcb.max()
                filter_ratio = ucb_filter.sum()/x_tensor.shape[0]
                _minimum_pick = 10
                if sum(ucb_filter[observed==1]) <= _minimum_pick:
                    _, indices = torch.topk(ucb[observed==1], min(_minimum_pick, data_size))
                    for idx in indices:
                        ucb_filter[util_array[[observed==1]][idx]] = 1
                observed_unfiltered = np.min([observed, ucb_filter.numpy()], axis=0)      # observed and not filtered outs
                init_x = x_tensor[observed_unfiltered==1]
                init_y = y_tensor[observed_unfiltered==1]
                
                sim_dkbo = DK_BO_AE(x_tensor[ucb_filter], y_tensor[ucb_filter], lr=lr, spectrum_norm=spectrum_norm, low_dim=low_dim,
                                    n_init=n_init,  train_iter=train_times, regularize=regularize, dynamic_weight=False,  retrain_nn=True,
                                    max=max_val, pretrained_nn=ae, verbose=verbose, init_x=init_x, init_y=init_y, exact_gp=exact_gp, noise_constraint=roi_noise_constraint)

                _roi_ucb = _ucb
                _roi_lcb, _roi_ucb = sim_dkbo.dkl.CI(x_tensor)
                # if ci_intersection:
                if not (default_beta): # only for plot & intersection
                    _roi_beta = min(1e2, max(1e-2, ucb.max()/_roi_ucb.max()) )
                else:
                    _roi_beta = (2 * np.log((x_tensor[ucb_filter].shape[0] * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
                _roi_lcb_scaled, _roi_ucb_scaled = beta_CI(_roi_lcb, _roi_ucb, _roi_beta)
                
                # intersection of ROI CI
                max_test_x_lcb[ucb_filter], min_test_x_ucb[ucb_filter] = beta_CI(lcb[ucb_filter], ucb[ucb_filter], _roi_beta)
                _lcb_scaling_factor, _ucb_scaling_factor = max_test_x_lcb[ucb_filter].max()/ _roi_lcb_scaled[ucb_filter].max(), min_test_x_ucb[ucb_filter].max() / _roi_ucb_scaled[ucb_filter].max()
                _max_test_x_lcb, _min_test_x_ucb = torch.max(max_test_x_lcb, _roi_lcb_scaled * _lcb_scaling_factor), torch.min(min_test_x_ucb, _roi_ucb_scaled * _ucb_scaling_factor) 
                max_test_x_lcb[ucb_filter], min_test_x_ucb[ucb_filter] = _max_test_x_lcb[ucb_filter], _min_test_x_ucb[ucb_filter]
                
                query_num = min(filter_interval, n_iter-iter)
                if filter_interval >= n_iter - iter:
                    _acq = 'lcb'
                else:
                    _acq = acq
                _roi_beta_passed_in = _roi_beta  if not (default_beta) else 0 # allow it to calculate internal ROI_beta
                sim_dkbo.query(n_iter=query_num, acq=_acq, study_ucb=False, study_interval=10, study_res_path=save_path,  if_tqdm=verbose, retrain_interval=retrain_interval,
                                ci_intersection=ci_intersection, max_test_x_lcb=max_test_x_lcb[ucb_filter], min_test_x_ucb=min_test_x_ucb[ucb_filter], beta=_roi_beta_passed_in)

                # update records
                _step_size = min(iter+filter_interval, n_iter)
                reg_record[rep, iter:_step_size] = sim_dkbo.regret[-_step_size:]
                ratio_record[rep, iter:_step_size] = min(filter_ratio, ratio_record[rep, iter-1]) if iter > 0 else filter_ratio

                # update interval
                max_LUCB_interval_record[rep:, 0, iter:_step_size] = (_ucb.max() - _lcb.max()).numpy()
                max_LUCB_interval_record[rep:, 1, iter:_step_size] = (_roi_ucb_scaled[ucb_filter].max() - _roi_lcb_scaled[ucb_filter].max()).numpy()
                max_LUCB_interval_record[rep:, 2, iter:_step_size] = (min_test_x_ucb[ucb_filter].max() - max_test_x_lcb[ucb_filter].max()).numpy()

                # early stop
                if reg_record[rep, :_step_size].min() < 1e-16:
                    break

                _filter_gap = _filter_ucb.min() - _filter_lcb.max()
                _iterator_info = {'beta': beta, 'fbeta': filter_beta, "roi_beta": _roi_beta.detach().item(), "regret":reg_record[rep, :_step_size].min(), "Filter Ratio": filter_ratio.detach().item(), 
                                  "Filter Gap": _filter_gap.detach().item(), 'roi noise': sim_dkbo.dkl.likelihood.noise.detach().item(), 'global noise': _f_model.likelihood.noise.detach().item()}

                iterator.set_postfix(_iterator_info)

                ucb_filtered_idx = util_array[ucb_filter]
                observed[ucb_filtered_idx[sim_dkbo.observed==1]] = 1

                # update model and therefore the confidence intervals for filtering
                _f_model = DKL(x_tensor[observed==1], y_tensor[observed==1].squeeze() if sum(observed) > 1 else y_tensor[observed==1],  
                            n_iter=train_times, low_dim=low_dim, pretrained_nn=ae, retrain_nn=retrain_nn, lr=lr, spectrum_norm=spectrum_norm,
                            exact_gp=exact_gp, noise_constraint=global_noise_constraint)

                if regularize:
                    _f_model.train_model_kneighbor_collision()
                else:
                    _f_model.train_model(verbose=verbose)

                
                lcb, ucb = _f_model.CI(x_tensor)

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    ratio_output_record = ratio_record.mean(axis=0)
    
    beta = 0 if default_beta else beta # for record

    if plot_result:
        # regret
        fig = plt.figure()
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/Filter{'-Exact' if exact_gp else ''}-{name}-B{beta}-FB{filter_beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}-RI{retrain_interval}{'-sec' if ci_intersection else ''}"
        plt.savefig(f"{_path}.png")
        # filter ratio
        fig = plt.figure()
        plt.plot(ratio_output_record)
        plt.ylabel("Ratio")
        plt.xlabel("Iteration")
        plt.title(f'ROI Ratio for {name}')
        _path = f"{save_path}/Filter{'-Exact' if exact_gp else ''}-{name}-ratio-B{beta}-FB{filter_beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}-RI{retrain_interval}{'-sec' if ci_intersection else ''}"
        plt.savefig(f"{_path}.png")
        # interval
        fig = plt.figure()
        max_LUCB_interval_record_output = max_LUCB_interval_record.mean(axis=0)
        plt.plot(max_LUCB_interval_record_output[0,:], label="Global")
        plt.plot(max_LUCB_interval_record_output[1,:], label="ROI")
        plt.plot(max_LUCB_interval_record_output[2,:], label="Intersection")
        plt.ylabel("Interval")
        plt.xlabel("Iteration")
        plt.legend()
        plt.title(f'Interval for {name}')
        _path = f"{save_path}/Filter{'-Exact' if exact_gp else ''}-{name}-interval-B{beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}-RI{retrain_interval}{'-sec' if ci_intersection else ''}"
        plt.savefig(f"{_path}.png")
        plt.close()
        # plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}", res=reg_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)
        
        # save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}-ratio", res=ratio_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
        #         init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

        # save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}-interval", res=max_LUCB_interval_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
        #         init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)


    if return_result:
        return reg_record
    else:
        return _f_model, sim_dkbo
