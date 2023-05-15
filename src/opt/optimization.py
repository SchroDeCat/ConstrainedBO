'''
Full pipeline for DKBO
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
 
DEVICE = torch.device('cpu')
# STUDY_PARTITION = True
STUDY_PARTITION = False



def dkl_opt_test(x_tensor, y_tensor, name, n_repeat=2, lr=1e-2, n_init=10, n_iter=40, train_iter=100, return_result=True, fix_seed=True,
                    pretrained=False, ae_loc=None, plot_result=False, save_result=False, save_path=None, acq="ts", verbose=True, study_partition=STUDY_PARTITION):
    '''
    Check DKL opt behavior on full dataset
    '''
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fix_seed:
            # print(n_init+n_repeat*n_iter)
            # _seed = rep*n_iter + n_init
            _seed = 70
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        for rep in range(n_repeat):
            # no AE pretrain 
            if verbose:
                print("DKBO")
            sim_dkl = DKL(x_tensor, y_tensor.squeeze(), lr=lr, n_iter=train_iter, low_dim=True,  pretrained_nn=ae)
            sim_dkl.train_model(loss_type="nll", verbose=True)
            

    if plot_result:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = sim_dkl.likelihood(sim_dkl.model(x_tensor))
            _, ucb = observed_pred.confidence_region()
        _path = f"{save_path}/DKL-R{n_repeat}-whole_L{int(-np.log10(lr))}-TI{train_iter}"
        # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
        fig = plt.figure()
        plt.scatter(y_tensor, ucb, s=2)
        plt.scatter(y_tensor, ucb, color='r', marker="*", s=5)
        plt.xlabel("Label")
        plt.ylabel("UCB")
        plt.colorbar()
        plt.title(f"{name}")
        plt.savefig(f"{_path}.png")
        print(f"Fig stored to {_path}")

    if save_result:
        pass

    if return_result:
        pass

def pure_dkbo(x_tensor, y_tensor, name, n_repeat=2, lr=1e-2, n_init=10, n_iter=40, train_iter=100, return_result=True, fix_seed=True, low_dim=True, beta=2, retrain_interval:int=1,
                    pretrained=False, ae_loc=None, plot_result=False, save_result=False, save_path=None, acq="ts", verbose=True, exact_gp=False, study_partition=STUDY_PARTITION, constrain_noise=False):
    if constrain_noise:
        # global_noise_constraint = gpytorch.constraints.Interval(0.1,1.3)
        global_noise_constraint = gpytorch.constraints.Interval(1e-5, .1)
        # name = f"{name}-noise_c"
        name = f"{name}-noise_c_low"
    else:
        global_noise_constraint = None
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fix_seed:
            # print(n_init+n_repeat*n_iter)
            # _seed = rep*n_iter + n_init
            _seed = 70
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        for rep in tqdm.tqdm(range(n_repeat)):
            # no AE pretrain 
            if verbose:
                print("DKBO")
            sim_dkbo = DK_BO_AE(x_tensor, y_tensor, lr=lr, low_dim=low_dim,
                                n_init=n_init,  train_iter=train_iter, regularize=False, dynamic_weight=False, 
                                max=max_val, pretrained_nn=ae, verbose=verbose, exact_gp=exact_gp, noise_constraint=global_noise_constraint)
            sim_dkbo.query(n_iter=n_iter, acq=acq, study_ucb=STUDY_PARTITION, study_interval=10, retrain_interval=retrain_interval, beta=beta, 
                            study_res_path=save_path, if_tqdm=verbose)
            reg_record[rep, :] = sim_dkbo.regret

    reg_output_record = reg_record.mean(axis=0)
    name = name + f"b{beta}"
    if not low_dim:
        name = name + "_hd"

    if plot_result:
        plt.figure()
        plt.plot(reg_output_record.squeeze(), label="dkbo")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Regret")
        plt.title("DKBO performance")
        # plt.show()
        _path = f"{save_path}/Pure-DK-{name}{'-Exact' if exact_gp else ''}-{acq}-R{n_repeat}-T{n_iter}_L{int(-np.log10(lr))}-TI{train_iter}-RI{retrain_interval}"
        plt.savefig(f"{_path}.png")
        plt.close()

    if save_result:
        assert not (save_path is None)
        _path = f"{save_path}/Pure-DK-{name}{'-Exact' if exact_gp else ''}-{acq}-R{n_repeat}-T{n_iter}_L{int(-np.log10(lr))}-TI{train_iter}-RI{retrain_interval}.npy"
        np.save(_path, reg_record)
       
        # save_res(save_path=save_path, name=name if low_dim else name+"-hd", res=reg_record, n_repeat=n_repeat, num_GP=1, n_iter=n_init, train_iter=train_iter,
                # init_strategy='none', cluster_interval=1, acq=acq, lr=lr, verbose=verbose,)

    if return_result:
        return reg_record


def ol_filter_dkbo(x_tensor, y_tensor, n_init=10, n_repeat=2, train_times=10, beta=2, regularize=True, low_dim=True, spectrum_norm=False, retrain_interval=1,
                   n_iter=40, filter_interval=1, acq="ts", ci_intersection=True, verbose=True, lr=1e-2, name="test", return_result=True, retrain_nn=True,
                   plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, study_partition=STUDY_PARTITION, _minimum_pick = 10, 
                   _delta = 0.2, filter_beta=.05, exact_gp=False, constrain_noise=False):
    # print(ucb_strategy)
    if constrain_noise:
        # global_noise_constraint = gpytorch.constraints.Interval(0.7,1.3)
        # roi_noise_constraint = gpytorch.constraints.Interval(0.1,0.7)
        # name = f"{name}-noise_c"
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
     # init dkl and generate ucb for partition
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
                
            # init in each round
            observed = np.zeros(data_size)
            observed[:n_init] = 1
            init_x = x_tensor[:n_init]
            init_y = y_tensor[:n_init]
            # TBD: fix the ae loading here
            _dkl = DKL(init_x, init_y.squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, noise_constraint=global_noise_constraint)
            if regularize:
                _dkl.train_model_kneighbor_collision()
            else:
                _dkl.train_model(verbose=verbose)
            lcb, ucb = _dkl.CI(x_tensor.to(DEVICE))
            # max_test_x_lcb, min_test_x_ucb = lcb.clone(), ucb.clone()
            # print(max_test_x_lcb is lcb)


            # each test instance
            iterator = tqdm.tqdm(range(0, n_iter, filter_interval))
            for iter in iterator:
                if default_beta:
                    beta = (2 * np.log((x_tensor.shape[0] * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
                _lcb, _ucb = beta_CI(lcb, ucb, beta)
                max_test_x_lcb, min_test_x_ucb = _lcb.clone(), _ucb.clone()    # actually not taking all historical intersections             
                # if iter == 0:
                    # max_test_x_lcb, min_test_x_ucb = _lcb.clone(), _ucb.clone()    # actually not taking all historical intersections             
                # else:
                    # max_test_x_lcb, min_test_x_ucb = torch.max(max_test_x_lcb, _lcb), torch.min(min_test_x_ucb, _ucb) 
                if default_fbeta:
                    filter_beta = beta
                
                _filter_lcb, _filter_ucb = beta_CI(lcb, ucb, filter_beta)
                # ucb_filter = _filter_ucb >= _filter_lcb[observed==1].max() # filtering
                ucb_filter = _filter_ucb >= _filter_lcb.max()
                filter_ratio = ucb_filter.sum()/x_tensor.shape[0]
                _minimum_pick = 10
                # print(f"ucb_filter sum {sum(ucb_filter[observed==1])}/{_minimum_pick}")
                if sum(ucb_filter[observed==1]) <= _minimum_pick:
                    _, indices = torch.topk(ucb[observed==1], min(_minimum_pick, data_size))
                    for idx in indices:
                        ucb_filter[util_array[[observed==1]][idx]] = 1
                # print(f"Afterwards ucb_filter sum {sum(ucb_filter[observed==1])}/{_minimum_pick}")
                observed_unfiltered = np.min([observed, ucb_filter.numpy()], axis=0)      # observed and not filtered outs
                # init_x = torch.cat([init_x, x_tensor[observed_unfiltered==1]])
                # init_y = torch.cat([init_y, y_tensor[observed_unfiltered==1]])
                init_x = x_tensor[observed_unfiltered==1]
                init_y = y_tensor[observed_unfiltered==1]

                # global GP
                if study_partition:
                    _path = f"{save_path}/Filter-{name}-B{beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}"
                    fig = plt.figure()
                    plt.scatter(y_tensor, ucb, c=ucb_filter, s=2, label="$\hat{X}$")
                    plt.scatter(y_tensor[observed==1], ucb[observed==1], color='r', marker="*", s=5, label="Obs")
                    plt.scatter(y_tensor[observed_unfiltered==1], ucb[observed_unfiltered==1], color='g', marker="+", s=25, label="unfiltered_Obs")
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.legend()
                    plt.title(f"{name} Iter {iter} GLOBAL")
                    threshold = _lcb[observed==1].max()
                    _output_scale = _dkl.model.covar_module.base_kernel.outputscale
                    plt.savefig(f"{_path}-Scale{_output_scale:.2E}-thres{threshold:.2E}-Iter{iter}.png")
                    plt.close()
                    print(f"Fig stored to {_path}")
                
                sim_dkbo = DK_BO_AE(x_tensor[ucb_filter], y_tensor[ucb_filter], lr=lr, spectrum_norm=spectrum_norm, low_dim=low_dim,
                                    n_init=n_init,  train_iter=train_times, regularize=regularize, dynamic_weight=False,  retrain_nn=True,
                                    max=max_val, pretrained_nn=ae, verbose=verbose, init_x=init_x, init_y=init_y, exact_gp=exact_gp, noise_constraint=roi_noise_constraint)

                _roi_ucb = _ucb
                _roi_lcb, _roi_ucb = sim_dkbo.dkl.CI(x_tensor)
                # if ci_intersection:
                if not (default_beta): # only for plot & intersection
                    _roi_beta = min(1e2, max(1e-2, ucb.max()/_roi_ucb.max()) )
                else:
                    # _roi_beta = min(1e2, max(1e-2, ucb.max()/_roi_ucb.max()) )
                    # _roi_beta = beta
                    _roi_beta = (2 * np.log((x_tensor[ucb_filter].shape[0] * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
                _roi_lcb_scaled, _roi_ucb_scaled = beta_CI(_roi_lcb, _roi_ucb, _roi_beta)
                
                if study_partition:
                    _path = f"{save_path}/Filter-{name}-B{beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}-ROI"
                    fig = plt.figure()
                    plt.scatter(y_tensor, _roi_ucb_scaled, c=ucb_filter, s=2, label="$\hat{X}$")
                    plt.scatter(y_tensor[observed==1], _roi_ucb_scaled[observed==1], color='r', marker="*", s=5, label="Obs")
                    plt.scatter(y_tensor[observed_unfiltered==1], _roi_ucb_scaled[observed_unfiltered==1], color='g', marker="+", s=25, label="unfiltered_Obs")
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.legend()
                    plt.title(f"{name} Iter {iter} ROI")
                    _output_scale = sim_dkbo.dkl.model.covar_module.base_kernel.outputscale
                    plt.savefig(f"{_path}-Scale{_output_scale:.2E}_roi_beta{_roi_beta:.1}-Iter{iter}.png")
                    plt.close()
                    print(f"Fig stored to {_path}")
                
                # intersection of ROI CI
                max_test_x_lcb[ucb_filter], min_test_x_ucb[ucb_filter] = beta_CI(lcb[ucb_filter], ucb[ucb_filter], _roi_beta)
                # max_test_x_lcb[ucb_filter], min_test_x_ucb[ucb_filter] = torch.max(max_test_x_lcb[ucb_filter], _roi_lcb_scaled[ucb_filter]), torch.min(min_test_x_ucb[ucb_filter], _roi_ucb_scaled[ucb_filter]) # intersection of ROI CI
                _lcb_scaling_factor, _ucb_scaling_factor = max_test_x_lcb[ucb_filter].max()/ _roi_lcb_scaled[ucb_filter].max(), min_test_x_ucb[ucb_filter].max() / _roi_ucb_scaled[ucb_filter].max()
                # _max_test_x_lcb, _min_test_x_ucb = torch.max(max_test_x_lcb, _roi_lcb_scaled), torch.min(min_test_x_ucb, _roi_ucb_scaled) 
                _max_test_x_lcb, _min_test_x_ucb = torch.max(max_test_x_lcb, _roi_lcb_scaled * _lcb_scaling_factor), torch.min(min_test_x_ucb, _roi_ucb_scaled * _ucb_scaling_factor) 
                max_test_x_lcb[ucb_filter], min_test_x_ucb[ucb_filter] = _max_test_x_lcb[ucb_filter], _min_test_x_ucb[ucb_filter]
                # if max_test_x_lcb[ucb_filter].max() -  min_test_x_ucb[ucb_filter].max() < 0:
                #     print(iter,min_test_x_ucb[ucb_filter].max() -  max_test_x_lcb[ucb_filter].max())
                
                if study_partition:
                    _path = f"{save_path}/Filter-{name}-B{beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}-INTER"
                    fig = plt.figure()
                    plt.scatter(y_tensor, _min_test_x_ucb, c=ucb_filter, s=2, label="$\hat{X}$")
                    plt.scatter(y_tensor[observed==1], _min_test_x_ucb[observed==1], color='r', marker="*", s=5, label="Obs")
                    plt.scatter(y_tensor[observed_unfiltered==1], _min_test_x_ucb[observed_unfiltered==1], color='g', marker="+", s=25, label="unfiltered_Obs")
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.legend()
                    plt.title(f"{name} Iter {iter} INTER")
                    _output_scale = sim_dkbo.dkl.model.covar_module.base_kernel.outputscale
                    plt.savefig(f"{_path}-Scale{_output_scale:.2E}_roi_beta{_roi_beta:.1}-Iter{iter}.png")
                    plt.close()
                    print(f"Fig stored to {_path}")
                    print(f"global {ucb.max()}{ucb.min()} ROI {_roi_ucb_scaled.max()}{_roi_ucb_scaled.min()} Inter {_min_test_x_ucb.max()} {_min_test_x_ucb.min()} ")

                query_num = min(filter_interval, n_iter-iter)
                if filter_interval >= n_iter - iter:
                # if filter_interval >= n_iter - iter and acq.lower == 'ci':
                    _acq = 'lcb'
                else:
                    _acq = acq
                _roi_beta_passed_in = _roi_beta  if not (default_beta) else 0 # allow it to calculate internal ROI_beta
                sim_dkbo.query(n_iter=query_num, acq=_acq, study_ucb=False, study_interval=10, study_res_path=save_path,  if_tqdm=verbose, retrain_interval=retrain_interval,
                                ci_intersection=ci_intersection, max_test_x_lcb=max_test_x_lcb[ucb_filter], min_test_x_ucb=min_test_x_ucb[ucb_filter], beta=_roi_beta_passed_in)
                                # ci_intersection=ci_intersection if iter > n_iter //2 else False, max_test_x_lcb=max_test_x_lcb[ucb_filter], min_test_x_ucb=min_test_x_ucb[ucb_filter], beta=_roi_beta_passed_in)
                # sim_dkbo.query(n_iter=query_num, acq=acq, study_ucb=False, study_interval=10, study_res_path=save_path,
                                # ci_intersection=ci_intersection, max_test_x_lcb=max_test_x_lcb[ucb_filter], min_test_x_ucb=_roi_ucb[ucb_filter], beta=_roi_beta)
                # sim_dkbo.query(n_iter=query_num, acq=acq, study_ucb=False, study_interval=10, study_res_path=save_path)
                
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

                # iterator.set_postfix(loss=reg_record[rep, :_step_size].min())
                _filter_gap = _filter_ucb.min() - _filter_lcb.max()
                _iterator_info = {'beta': beta, 'fbeta': filter_beta, "roi_beta": _roi_beta.detach().item(), "regret":reg_record[rep, :_step_size].min(), "Filter Ratio": filter_ratio.detach().item(), 
                                  "Filter Gap": _filter_gap.detach().item(), 'roi noise': sim_dkbo.dkl.likelihood.noise.detach().item(), 'global noise': _dkl.likelihood.noise.detach().item()}

                iterator.set_postfix(_iterator_info)

                ucb_filtered_idx = util_array[ucb_filter]
                observed[ucb_filtered_idx[sim_dkbo.observed==1]] = 1

                # update ucb for filtering
                # _ae = AE(x_tensor[ucb_filter], lr=1e-3)
                # _ae.train_ae(epochs=train_times, batch_size=200, verbose=True)
                # print("Global GP: ", y_tensor[observed==1].squeeze().size(), f"n_iter={train_times}, low_dim={low_dim},retrain_nn={retrain_nn}, lr={lr}, spectrum_norm={spectrum_norm} regularize {regularize}")
                _dkl = DKL(x_tensor[observed==1], y_tensor[observed==1].squeeze() if sum(observed) > 1 else y_tensor[observed==1],  
                            n_iter=train_times, low_dim=low_dim, pretrained_nn=ae, retrain_nn=retrain_nn, lr=lr, spectrum_norm=spectrum_norm,
                            exact_gp=exact_gp, noise_constraint=global_noise_constraint)
                # _dkl = DKL(x_tensor[ucb_filter], y_tensor[ucb_filter].squeeze() if sum(ucb_filter) > 1 else y_tensor[ucb_filter],  
                            # n_iter=train_times, low_dim=True)
                # 
                if regularize:
                    _dkl.train_model_kneighbor_collision()
                else:
                    _dkl.train_model(verbose=verbose)

                
                lcb, ucb = _dkl.CI(x_tensor)

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
        
        save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}-ratio", res=ratio_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

        save_res(save_path=save_path, name=f"{name}{'-Exact' if exact_gp else ''}-B{beta}-FB{filter_beta}-RI{retrain_interval}-interval", res=max_LUCB_interval_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)


    if return_result:
        return reg_record
    else:
        return _dkl, sim_dkbo

def ol_partition_dkbo(x_tensor, y_tensor, init_strategy:str="kmeans", n_init=10, n_repeat=2, num_GP=2, train_times=10,
                    n_iter=40, cluster_interval=1, acq="ts", verbose=True, lr=1e-2, name="test", return_result=True, ucb_strategy="exact",
                    plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, study_partition=STUDY_PARTITION):
    # print(ucb_strategy)
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])
     # init dkl and generate ucb for partition
    data_size = x_tensor.size(0)
    util_array = np.arange(data_size)

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in tqdm.tqdm(range(n_repeat), desc=f"{init_strategy}"):
            # set seed
            if fix_seed:
                # print(n_init+n_repeat*n_iter)
                _seed = rep * n_iter + n_init
                # _seed = 70
                torch.manual_seed(_seed)
                np.random.seed(_seed)
                random.seed(_seed)
                torch.cuda.manual_seed(_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # init in each round
            observed = np.zeros(data_size)
            observed[:n_init] = 1
            init_x = x_tensor[:n_init]
            init_y = y_tensor[:n_init]
            _dkl = DKL(init_x, init_y.squeeze(), n_iter=10, low_dim=True)
            _dkl.train_model()
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                _observed_pred = _dkl.likelihood(_dkl.model(x_tensor.to(DEVICE)))
                _, ucb = _observed_pred.confidence_region()
            # print(f"pretrained {pretrained} ae {type(ae)}  n_iter {train_times}")
            cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), num_GP, init_strategy, dkl=True, pretrained_nn=ae, train_iter=train_times)
            cluster_id_init = cluster_id[observed==1]

            # each test instance
            for iter in range(0, n_iter, cluster_interval):
                scaled_input_list, scaled_output_list = [], []
                init_x_list, init_y_list = [], []
                for idx in range(num_GP):
                    cluster_filter = cluster_id == idx
                    cluster_filter[:n_init] = True
                    # print(f"cluster {idx} size {sum(cluster_filter)}")
                    scaled_input_list.append(x_tensor[cluster_filter])
                    scaled_output_list.append(y_tensor[cluster_filter])

                    cluster_init_filter = cluster_id_init == idx
                    cluster_init_filter[:n_init] = True
                    # print(f"cluster init {idx} size {sum(cluster_init_filter)} {cluster_id_init}")
                    # print(f"init_x size {init_x.size()}, observed num {sum(observed)}")
                    init_x_list.append(init_x[cluster_init_filter])
                    init_y_list.append(init_y[cluster_init_filter].reshape([-1,1]))
                    # print(f"sizes {init_x_list[0].size()}, {init_y_list[0].size()} num_GP {num_GP}")

                dkbo_olp = DK_BO_OLP(init_x_list, init_y_list, scaled_input_list, scaled_output_list, lr=lr, train_iter=train_times,
                                n_init=n_init, regularize=False, dynamic_weight=False, max_val=max_val, num_GP=num_GP, pretrained_nn=ae)
                # print("dkbo y list", torch.cat(dkbo_olp.init_y_list).max(), "dkbo Y list shape", torch.cat(dkbo_olp.init_y_list).size())

                # record regret
                query_num = min(cluster_interval, n_iter-iter)
                # print(f"query num {query_num}")
                dkbo_olp.query(n_iter=query_num, acq=acq, verbose=verbose)
                # print(dkbo_olp.regret.shape)
                # print("dkbo Reg", dkbo_olp.regret)
                reg_record[rep, iter:min(iter+cluster_interval, n_iter)] = dkbo_olp.regret
                
                # update ucb for clustering
                ucb = dkbo_olp.ucb_func(x_tensor=x_tensor, method=ucb_strategy, cluster_id=cluster_id)
                for loc in dkbo_olp.observed:
                    cluster_filter = cluster_id==loc[0]
                    cluster_filter[:n_init] = True  # align with previous workaround
                    if sum(cluster_filter) > 0:
                        observed[util_array[cluster_filter][loc[1]]] = 1
                
                cluster_id = clustering_methods(x_tensor, ucb, num_GP, init_strategy, dkl=True, pretrained_nn=ae, train_iter=train_times)

                # update observed clustering
                # init_x, init_y = torch.cat(dkbo_olp.init_x_list), torch.cat(dkbo_olp.init_y_list)
                init_x, init_y = x_tensor[observed==1], y_tensor[observed==1]
                # print(f"max n_init init_y {init_y[:n_init].max()} reg {obj - init_y[:n_init].max()}")
                cluster_id_init = cluster_id[observed==1]

                if study_partition:
                    _path = f"{save_path}/OL-{name}-{init_strategy}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_times}"
                    # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
                    fig = plt.figure()
                    plt.scatter(y_tensor, ucb, c=cluster_id, s=2)
                    plt.scatter(y_tensor[observed==1], ucb[observed==1], color='r', marker="*", s=5)
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.title(f"{name} Iter {iter}")
                    plt.savefig(f"{_path}-Iter{iter}.png")
                    print(f"Fig stored to {_path}")

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    
    if plot_result:
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        # plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=name, res=reg_record, n_repeat=n_repeat, num_GP=num_GP, n_iter=n_iter, train_iter=train_times,
                init_strategy=init_strategy, cluster_interval=cluster_interval, acq=acq, lr=lr, ucb_strategy=ucb_strategy, verbose=verbose,)

    if return_result:
        return reg_record

def ol_partition_kmeansY_dkbo(x_tensor, y_tensor, n_init=10, n_repeat=2, num_GP=2, train_times=10, low_dim=True,
                    n_iter=40, cluster_interval=1, acq="ts", verbose=True, lr=1e-2, name="test", return_result=True, ci_intersection=True,
                    plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, study_partition=STUDY_PARTITION):
    # print(ucb_strategy)
    init_strategy = "kmeans-y"
    ucb_strategy = 'exact' if not ci_intersection else 'min'
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])
    ratio_record = np.zeros([n_repeat, n_iter])
     # init dkl and generate ucb for partition
    data_size = x_tensor.size(0)
    util_array = np.arange(data_size)

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in tqdm.tqdm(range(n_repeat), desc=f"{init_strategy}"):
            # set seed
            if fix_seed:
                # print(n_init+n_repeat*n_iter)
                _seed = rep * 20 + n_init
                # _seed = 70
                torch.manual_seed(_seed)
                np.random.seed(_seed)
                random.seed(_seed)
                torch.cuda.manual_seed(_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # init in each round
            observed = np.zeros(data_size)
            observed[:n_init] = 1
            init_x = x_tensor[:n_init]
            init_y = y_tensor[:n_init]
            _dkl = DKL(init_x, init_y.squeeze(), n_iter=10, low_dim=low_dim)
            _dkl.train_model()
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                _observed_pred = _dkl.likelihood(_dkl.model(x_tensor.to(DEVICE)))
                _, ucb = _observed_pred.confidence_region()
            # print(f"pretrained {pretrained} ae {type(ae)}  n_iter {train_times}")
            cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), num_GP, init_strategy, dkl=True, pretrained_nn=ae, train_iter=train_times)
            cluster_id_init = cluster_id[observed==1]

            # each test instance
            iterator = tqdm.tqdm(range(0, n_iter, cluster_interval))
            for iter in iterator:
                scaled_input_list, scaled_output_list = [], []
                init_x_list, init_y_list = [], []
                for idx in range(num_GP):
                    cluster_filter = cluster_id == idx
                    cluster_filter[:n_init] = True
                    # print(f"cluster {idx} size {sum(cluster_filter)}")
                    scaled_input_list.append(x_tensor[cluster_filter])
                    scaled_output_list.append(y_tensor[cluster_filter])

                    cluster_init_filter = cluster_id_init == idx
                    cluster_init_filter[:n_init] = True
                    filter_ratio = sum(cluster_init_filter) / cluster_init_filter.shape[0]
                    # print(f"cluster init {idx} size {sum(cluster_init_filter)} {cluster_id_init}")
                    # print(f"init_x size {init_x.size()}, observed num {sum(observed)}")
                    init_x_list.append(init_x[cluster_init_filter])
                    init_y_list.append(init_y[cluster_init_filter].reshape([-1,1]))
                    # print(f"sizes {init_x_list[0].size()}, {init_y_list[0].size()} num_GP {num_GP}")

                dkbo_olp = DK_BO_OLP(init_x_list, init_y_list, scaled_input_list, scaled_output_list, lr=lr, train_iter=train_times,low_dim=low_dim,
                                n_init=n_init, regularize=False, dynamic_weight=False, max_val=max_val, num_GP=num_GP, pretrained_nn=ae)
                # print("dkbo y list", torch.cat(dkbo_olp.init_y_list).max(), "dkbo Y list shape", torch.cat(dkbo_olp.init_y_list).size())

                # record regret
                query_num = min(cluster_interval, n_iter-iter)
                dkbo_olp.query(n_iter=query_num, acq=acq, verbose=verbose)
                reg_record[rep, iter:min(iter+cluster_interval, n_iter)] = dkbo_olp.regret
                ratio_record[rep, iter:min(iter+cluster_interval, n_iter)] = filter_ratio

                iterator.set_postfix({"regret": reg_record[rep, :min(iter+cluster_interval, n_iter)].min(), "filter ratio": filter_ratio})
                
                # update ucb for clustering
                ucb = dkbo_olp.ucb_func(x_tensor=x_tensor, method=ucb_strategy, cluster_id=cluster_id)
                for loc in dkbo_olp.observed:
                    cluster_filter = cluster_id==loc[0]
                    cluster_filter[:n_init] = True  # align with previous workaround
                    if sum(cluster_filter) > 0:
                        observed[util_array[cluster_filter][loc[1]]] = 1
                
                cluster_id = clustering_methods(x_tensor, ucb, num_GP, init_strategy, dkl=True, pretrained_nn=ae, train_iter=train_times)

                # update observed clustering
                # init_x, init_y = torch.cat(dkbo_olp.init_x_list), torch.cat(dkbo_olp.init_y_list)
                init_x, init_y = x_tensor[observed==1], y_tensor[observed==1]
                # print(f"max n_init init_y {init_y[:n_init].max()} reg {obj - init_y[:n_init].max()}")
                cluster_id_init = cluster_id[observed==1]

                if study_partition:
                    _path = f"{save_path}/OL-{name}-{init_strategy}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_times}"
                    # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
                    fig = plt.figure()
                    plt.scatter(y_tensor, ucb, c=cluster_id, s=2)
                    plt.scatter(y_tensor[observed==1], ucb[observed==1], color='r', marker="*", s=5)
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.title(f"{name} Iter {iter}")
                    plt.savefig(f"{_path}-Iter{iter}.png")
                    print(f"Fig stored to {_path}")

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
        ratio_record[rep] = np.minimum.accumulate(ratio_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    ratio_output_record = ratio_record.mean(axis=0)

    
    if plot_result:
        fig = plt.figure()
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/Partition_kmeansY-{name}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}"
        plt.savefig(f"{_path}.png")
        plt.close()
        # filter ratio
        fig = plt.figure()
        plt.plot(ratio_output_record)
        plt.ylabel("Ratio")
        plt.xlabel("Iteration")
        plt.title(f'ROI Ratio for {name}')
        _path = f"{save_path}/Partition_kmeansY-ratio-{name}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}"
        plt.savefig(f"{_path}.png")
        plt.close()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=f"{name}", res=reg_record, n_repeat=n_repeat, num_GP=num_GP, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=cluster_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)
        
        save_res(save_path=save_path, name=f"{name}-ratio", res=ratio_record, n_repeat=n_repeat, num_GP=num_GP, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=cluster_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

    if return_result:
        return reg_record


def truvar(x_tensor, y_tensor, n_init=10, n_repeat=2, train_times=10, beta=2, low_dim=True, spectrum_norm=False, 
                   n_iter=40, filter_interval=1, ci_intersection=True, verbose=True, lr=1e-2, name="test", return_result=True, retrain_nn=True,
                   plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, _minimum_pick = 10, 
                   _delta = 0.2, filter_beta=.05):
    '''Truncated Variance Reduction'''
    name = name if low_dim else name+'-hd'
    acq = 'truvar'
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])
    max_LUCB_interval_record = np.zeros([n_repeat, 3, n_iter]) # 0 - Global, 1 - ROI, 2 -- intersection
     # init dkl and generate ucb for partition
    data_size = x_tensor.size(0)
    util_array = np.arange(data_size)


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
                
            # init in each round
            observed = np.zeros(data_size)
            observed[:n_init] = 1
            init_x = x_tensor[:n_init]
            init_y = y_tensor[:n_init]
            _dkl = DKL(init_x, init_y.squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm)
            lcb, ucb = _dkl.CI(x_tensor.to(DEVICE))


            # each test instance
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

                # global GP                
                sim_dkbo = DK_BO_AE(x_tensor[ucb_filter], y_tensor[ucb_filter], lr=lr, spectrum_norm=spectrum_norm, low_dim=low_dim,
                                    n_init=n_init,  train_iter=train_times, regularize=False, dynamic_weight=False,  retrain_nn=True,
                                    max=max_val, pretrained_nn=ae, verbose=verbose, init_x=init_x, init_y=init_y)

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
                _roi_beta_passed_in = _roi_beta  if not (default_beta) else 0 # allow it to calculate internal ROI_beta
                sim_dkbo.query(n_iter=query_num, acq=acq, study_ucb=False, study_interval=10, study_res_path=save_path,  if_tqdm=verbose,
                                ci_intersection=ci_intersection, max_test_x_lcb=max_test_x_lcb[ucb_filter], min_test_x_ucb=min_test_x_ucb[ucb_filter], beta=_roi_beta_passed_in)

                # update records
                _step_size = min(iter+filter_interval, n_iter)
                reg_record[rep, iter:_step_size] = sim_dkbo.regret[-_step_size:]
                # # update interval
                # max_LUCB_interval_record[rep:, 0, iter:_step_size] = (_ucb.max() - _lcb.max()).numpy()
                # max_LUCB_interval_record[rep:, 1, iter:_step_size] = (_roi_ucb_scaled[ucb_filter].max() - _roi_lcb_scaled[ucb_filter].max()).numpy()
                # max_LUCB_interval_record[rep:, 2, iter:_step_size] = (min_test_x_ucb[ucb_filter].max() - max_test_x_lcb[ucb_filter].max()).numpy()

                # early stop
                if reg_record[rep, :_step_size].min() < 1e-16:
                    break

                # iterator.set_postfix(loss=reg_record[rep, :_step_size].min())
                iterator.set_postfix({'beta': beta, 'fbeta': filter_beta, "roi_beta": _roi_beta, "regret":reg_record[rep, :_step_size].min(), "Filter Ratio": filter_ratio, "Filter Gap": _filter_ucb.min() - _filter_lcb.max()})

                ucb_filtered_idx = util_array[ucb_filter]
                observed[ucb_filtered_idx[sim_dkbo.observed==1]] = 1

                # update ucb for filtering
                _dkl = DKL(x_tensor[observed==1], y_tensor[observed==1].squeeze() if sum(observed) > 1 else y_tensor[observed==1],  
                            n_iter=train_times, low_dim=low_dim, pretrained_nn=ae, retrain_nn=retrain_nn, lr=lr, spectrum_norm=spectrum_norm)

                
                lcb, ucb = _dkl.CI(x_tensor)

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    
    beta = 0 if default_beta else beta # for record
    if plot_result:
        # regret
        fig = plt.figure()
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/TruVAR-{name}-B{beta}-FB{filter_beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}"
        plt.savefig(f"{_path}.png")
        # interval
        fig = plt.figure()
        # max_LUCB_interval_record_output = max_LUCB_interval_record.mean(axis=0)
        # plt.plot(max_LUCB_interval_record_output[0,:], label="Global")
        # plt.plot(max_LUCB_interval_record_output[1,:], label="ROI")
        # plt.plot(max_LUCB_interval_record_output[2,:], label="Intersection")
        # plt.ylabel("Interval")
        # plt.xlabel("Iteration")
        # plt.legend()
        # plt.title(f'Interval for {name}')
        # _path = f"{save_path}/TruVAR-{name}-interval-B{beta}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}"
        # plt.savefig(f"{_path}.png")
        plt.close()
        # plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=f"{name}-B{beta}-FB{filter_beta}", res=reg_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

        # save_res(save_path=save_path, name=f"{name}-B{beta}-FB{filter_beta}-interval", res=max_LUCB_interval_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                # init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

    if return_result:
        return reg_record
    else:
        return _dkl, sim_dkbo
