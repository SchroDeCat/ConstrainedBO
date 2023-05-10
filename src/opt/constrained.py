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
from .dkbo_ae_constrained import DK_BO_AE_C
from math import ceil, floor
from sparsemax import Sparsemax
from scipy.stats import ttest_ind
from scipy.stats import norm
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

def cbo(x_tensor, y_tensor, c_tensor, constraint_threshold, constraint_confidence=0.8, optimization_ratio=0.8, n_init=10, n_repeat=2, train_times=10, beta=2, regularize=False, low_dim=True, 
            spectrum_norm=False, retrain_interval=1, n_iter=40, filter_interval=1, acq="ci", ci_intersection=True, verbose=True, lr=1e-2, name="test", return_result=True, retrain_nn=True,
            plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, _minimum_pick = 10, 
            _delta = 0.2, filter_beta=.05, exact_gp=False, constrain_noise=False, local_model=True):
    
    ####### configurations
    if constrain_noise:
        global_noise_constraint = gpytorch.constraints.Interval(0.1,.6)
        roi_noise_constraint = gpytorch.constraints.Interval(1e-5,0.1)
        name = f"{name}-noise_c"
    else:
        global_noise_constraint = None
        roi_noise_constraint = None

    c_threshold = norm.ppf(constraint_confidence, loc=constraint_threshold, scale=1)
    feasibility_filter = c_tensor > c_threshold
    assert sum(feasibility_filter) > 0
    name = name if low_dim else name+'-hd'
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])
    ratio_record = np.zeros([n_repeat, n_iter])
    max_LUCB_interval_record = np.zeros([n_repeat, 3, n_iter]) # 0 - Global, 1 - ROI, 2 -- intersection

    ####### init dkl and generate f_ucb for partition
    data_size = x_tensor.size(0)
    assert y_tensor.squeeze().size(0) == data_size
    assert c_tensor.squeeze().size(0) == data_size
    if len(y_tensor.size()) > 2 or len(c_tensor.size()) > 2 or len(x_tensor.size()) > 2:
        raise ValueError(f"Shape of input tensor is ")    
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
            init_c = c_tensor[:n_init]
            # NOTE: AE is shared for f and c
            _f_model = DKL(init_x, init_y.squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, noise_constraint=global_noise_constraint, pretrained_nn=ae)
            _c_model = DKL(init_x, init_c.squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, noise_constraint=global_noise_constraint, pretrained_nn=ae)
            if regularize:
                _f_model.train_model_kneighbor_collision()
                _c_model.train_model_kneighbor_collision()
            else:
                _f_model.train_model(verbose=False)
                _c_model.train_model(verbose=False)
            f_lcb, f_ucb = _f_model.CI(x_tensor.to(DEVICE))
            c_lcb, c_ucb = _c_model.CI(x_tensor.to(DEVICE))


            ####### each test instance
            iterator = tqdm.tqdm(range(0, n_iter, filter_interval))
            f_c_total_iter = ceil(n_iter * optimization_ratio)
            cons_only_iter = n_iter - f_c_total_iter
            for iter in iterator:
                # optimization CI
                if default_beta:
                    beta = (2 * np.log((x_tensor.size(0) * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
                _f_lcb, _f_ucb = beta_CI(f_lcb, f_ucb, beta)
                _c_lcb, _c_ucb = beta_CI(c_lcb, c_ucb, beta)
                # Take intersection of all historical CIs
                if iter == 0:
                    f_max_test_x_lcb, f_min_test_x_ucb = _f_lcb.clone(), _f_ucb.clone()    
                    c_max_test_x_lcb, c_min_test_x_ucb = _c_lcb.clone(), _c_ucb.clone()
                else:
                    f_max_test_x_lcb, f_min_test_x_ucb = torch.max(_f_lcb, f_max_test_x_lcb), torch.min(_f_ucb, f_min_test_x_ucb)
                    c_max_test_x_lcb, c_min_test_x_ucb = torch.max(_c_lcb, c_max_test_x_lcb), torch.min(_c_ucb, c_min_test_x_ucb)
                    assert f_max_test_x_lcb.size(0) == data_size and f_min_test_x_ucb.size(0) == data_size
                    assert c_max_test_x_lcb.size(0) == data_size and c_min_test_x_ucb.size(0) == data_size
                    
                # filtering with another CI
                if default_fbeta:
                    filter_beta = beta
                _f_filter_lcb, _f_filter_ucb = beta_CI(f_lcb, f_ucb, filter_beta)
                _c_filter_lcb, _c_filter_ucb = beta_CI(c_lcb, c_ucb, filter_beta)
                
                c_sci_filter = _c_filter_lcb >= c_threshold
                c_roi_filter = _c_filter_ucb >= c_threshold
                c_uci_filter = c_roi_filter.logical_xor(c_sci_filter) 
                if sum(c_sci_filter) > 0:
                    f_roi_filter = _f_filter_ucb >= _f_filter_lcb[c_sci_filter.squeeze()].max() 
                else:
                    f_roi_filter = _f_filter_ucb >= _f_filter_lcb[feasibility_filter.squeeze()].min()
                roi_filter = c_roi_filter.logical_and(f_roi_filter)

                _minimum_pick = 10
                if sum(roi_filter[observed==1]) <= _minimum_pick:
                    _, indices = torch.topk(c_ucb[observed==1], min(_minimum_pick, data_size))
                    for idx in indices:
                        roi_filter[util_array[observed==1][idx]] = 1
                filter_ratio = roi_filter.sum()/data_size
                observed_unfiltered = np.min([observed, roi_filter.numpy()], axis=0)      # observed and not filtered outs
                init_x = x_tensor[observed_unfiltered==1]
                init_y = y_tensor[observed_unfiltered==1]
                init_c = c_tensor[observed_unfiltered==1]

                # TBD: optimization
                if local_model: # allow training a local model and optimize on top of it
                    _f_model_passed_in, _c_model_passed_in = None, None
                else:
                    _f_model_passed_in, _c_model_passed_in = _f_model, _c_model
                _cbo = DK_BO_AE_C(x_tensor, y_tensor, c_tensor, roi_filter, c_uci_filter, optimization_ratio, lr=lr, spectrum_norm=spectrum_norm, low_dim=low_dim,
                                    n_init=n_init,  train_iter=train_times, regularize=regularize, dynamic_weight=False,  retrain_nn=True, c_threshold=c_threshold,
                                    max=max_val, pretrained_nn=ae, verbose=verbose, init_x=init_x, init_y=init_y, init_c=init_c, exact_gp=exact_gp, noise_constraint=roi_noise_constraint,
                                    f_model=_f_model_passed_in, c_model=_c_model_passed_in)

                _roi_f_lcb, _roi_f_ucb = _cbo.f_model.CI(x_tensor)
                _roi_c_lcb, _roi_c_ucb = _cbo.c_model.CI(x_tensor)

                # if ci_intersection:
                if not (default_beta): # only for visualization & intersection
                    _roi_beta = min(1e2, max(1e-2, f_ucb.max()/_roi_f_ucb.max()) )
                else:
                    _roi_beta = (2 * np.log((x_tensor[roi_filter].shape[0] * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta


                def intersecting_ROI_globe(max_all_lcb, min_all_ucb, roi_lcb, roi_ucb, roi_beta, roi_filter, adaptive_scaling=False):
                    roi_lcb_scaled, roi_ucb_scaled = beta_CI(roi_lcb, roi_ucb, roi_beta)   
                    if adaptive_scaling:
                        _lcb_scaling_factor, _ucb_scaling_factor = max_all_lcb[roi_filter].max()/ roi_lcb_scaled[roi_filter].max(), min_all_ucb[roi_filter].max() / roi_lcb_scaled[roi_filter].max()
                    else:
                        _lcb_scaling_factor, _ucb_scaling_factor = 1, 1

                    _max_all_lcb, _min_all_ucb = torch.max(max_all_lcb, roi_lcb_scaled * _lcb_scaling_factor), torch.min(min_all_ucb, roi_ucb_scaled * _ucb_scaling_factor) 
                    max_all_lcb[roi_filter], min_all_ucb[roi_filter] = _max_all_lcb[roi_filter], _min_all_ucb[roi_filter]
                    return max_all_lcb, min_all_ucb, roi_lcb_scaled, roi_ucb_scaled

  
                # intersection of ROI CI and global CI
                f_max_test_x_lcb, f_min_test_x_ucb, _roi_f_lcb_scaled,  _roi_f_ucb_scaled   = intersecting_ROI_globe(f_max_test_x_lcb, f_min_test_x_ucb, _roi_f_lcb, _roi_f_ucb, _roi_beta, roi_filter)
                c_max_test_x_lcb, c_min_test_x_ucb, _,                  _                   = intersecting_ROI_globe(c_max_test_x_lcb, c_min_test_x_ucb, _roi_c_lcb, _roi_c_ucb, _roi_beta, roi_filter)
                # _roi_f_lcb_scaled, _roi_f_ucb_scaled = beta_CI(_roi_f_lcb, _roi_f_ucb, _roi_beta)
                # f_max_test_x_lcb[roi_filter], f_min_test_x_ucb[roi_filter] = beta_CI(f_lcb[roi_filter], f_ucb[roi_filter], _roi_beta)
                # _lcb_scaling_factor, _ucb_scaling_factor = f_max_test_x_lcb[roi_filter].max()/ _roi_f_lcb_scaled[roi_filter].max(), f_min_test_x_ucb[roi_filter].max() / _roi_f_ucb_scaled[roi_filter].max()
                # _max_test_x_lcb, _min_test_x_ucb = torch.max(f_max_test_x_lcb, _roi_f_lcb_scaled * _lcb_scaling_factor), torch.min(f_min_test_x_ucb, _roi_f_ucb_scaled * _ucb_scaling_factor) 
                # f_max_test_x_lcb[roi_filter], f_min_test_x_ucb[roi_filter] = _max_test_x_lcb[roi_filter], _min_test_x_ucb[roi_filter]
                
                # two stage optimization, first f and c, then only c.
                if iter < f_c_total_iter or sum(c_uci_filter) == 0:
                    interval_query_ceil = f_c_total_iter - iter if iter < f_c_total_iter else  n_iter - iter # when c_uci is empty
                    query_num = min(filter_interval, interval_query_ceil) 
                    assert query_num > 0
                    _acq = 'lcb' if f_c_total_iter - iter <= filter_interval else acq
                    _roi_beta_passed_in = _roi_beta  if not (default_beta) else 0 # allow it to calculate internal ROI_beta
                    _cbo.query_f_c(n_iter=query_num, acq=_acq, study_interval=10, study_res_path=save_path,  if_tqdm=False, retrain_interval=retrain_interval,
                                    ci_intersection=ci_intersection, f_max_test_x_lcb=f_max_test_x_lcb, f_min_test_x_ucb=f_min_test_x_ucb,
                                    c_max_test_x_lcb=c_max_test_x_lcb, c_min_test_x_ucb=c_min_test_x_ucb, 
                                    beta=_roi_beta_passed_in)
                else:
                    interval_query_ceil =  n_iter - iter
                    query_num = min(filter_interval, interval_query_ceil) 
                    assert query_num > 0
                    _acq = 'lcb' if f_c_total_iter - iter <= filter_interval else acq
                    _roi_beta_passed_in = _roi_beta  if not (default_beta) else 0 # allow it to calculate internal ROI_beta
                    _cbo.query_cons(n_iter=query_num, acq=_acq, study_interval=10, study_res_path=save_path,  if_tqdm=False, retrain_interval=retrain_interval,
                                    ci_intersection=ci_intersection, f_max_test_x_lcb=f_max_test_x_lcb, f_min_test_x_ucb=f_min_test_x_ucb, beta=_roi_beta_passed_in,
                                    c_max_test_x_lcb=c_max_test_x_lcb, c_min_test_x_ucb=c_min_test_x_ucb)

                # update records
                _step_size = iter + query_num
                reg_record[rep, iter:_step_size] = _cbo.regret[-query_num:]
                ratio_record[rep, iter:_step_size] = min(filter_ratio, ratio_record[rep, iter-1]) if iter > 0 else filter_ratio

                # update interval
                _f_lcb_filter = c_sci_filter if sum(c_sci_filter) > 0 else y_tensor.argmin() # approximation when sci is empty
                max_LUCB_interval_record[rep:, 0, iter:_step_size] = (_f_ucb.max() - _f_lcb[_f_lcb_filter].max()).numpy()   # global
                max_LUCB_interval_record[rep:, 1, iter:_step_size] = (_roi_f_ucb_scaled[roi_filter].max() - _roi_f_lcb_scaled[_f_lcb_filter].max()).numpy() # ROI
                max_LUCB_interval_record[rep:, 2, iter:_step_size] = (f_min_test_x_ucb[roi_filter].max() - f_max_test_x_lcb[_f_lcb_filter].max()).numpy() # intersection

                # early stop
                if reg_record[rep, :_step_size].min() < 1e-16:
                    break

                _filter_gap = _f_filter_ucb.min() - _f_filter_lcb[_f_lcb_filter].max()
                _iterator_info = {'beta': beta, 'fbeta': filter_beta, "roi_beta": _roi_beta, "regret":reg_record[rep, :_step_size].min(), "Filter Ratio": filter_ratio.detach().item(), 
                                  "Filter Gap": _filter_gap.detach().item(), 'roi noise': _cbo.f_model.likelihood.noise.detach().item(), 'global noise': _f_model.likelihood.noise.detach().item()}

                iterator.set_postfix(_iterator_info)

                ucb_filtered_idx = util_array[roi_filter]
                # observed[ucb_filtered_idx[_cbo.observed==1]] = 1
                observed[util_array[_cbo.observed==1]] = 1

                # update model and therefore the confidence intervals for filtering
                _f_model = DKL(x_tensor[observed==1], y_tensor[observed==1].squeeze() if sum(observed) > 1 else y_tensor[observed==1],  
                            n_iter=train_times, low_dim=low_dim, pretrained_nn=ae, retrain_nn=retrain_nn, lr=lr, spectrum_norm=spectrum_norm,
                            exact_gp=exact_gp, noise_constraint=global_noise_constraint)
                _c_model = DKL(x_tensor[observed==1], c_tensor[observed==1].squeeze() if sum(observed) > 1 else c_tensor[observed==1],  
                    n_iter=train_times, low_dim=low_dim, pretrained_nn=ae, retrain_nn=retrain_nn, lr=lr, spectrum_norm=spectrum_norm,
                    exact_gp=exact_gp, noise_constraint=global_noise_constraint)

                if regularize:
                    _f_model.train_model_kneighbor_collision()
                    _c_model.train_model_kneighbor_collision()
                else:
                    _f_model.train_model(verbose=False)
                    _c_model.train_model(verbose=False)

                
                f_lcb, f_ucb = _f_model.CI(x_tensor.to(DEVICE))
                c_lcb, c_ucb = _c_model.CI(x_tensor.to(DEVICE))

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    ratio_output_record = ratio_record.mean(axis=0)
    
    beta = 0 if default_beta else beta # for record

    ### Export results
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
        
        save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}-ratio", res=ratio_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

        save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}-interval", res=max_LUCB_interval_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)


    if return_result:
        return reg_record
    else:
        return _f_model, _c_model, _cbo
