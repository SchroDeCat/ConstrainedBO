'''
Full pipeline for decoupled constrained BO
'''

import gpytorch
import random
import torch
import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import time

from typing import Tuple, Any
from ..SCBO import SCBO
from ..models import DKL, AE, beta_CI
from ..utils import save_res, model_list_CI, intersecting_ROI_globe, feasible_filter_gen
from .dkbo_ae_constrained_decoupled import DK_BO_AE_C_M_DEC
from math import ceil, floor
from scipy.stats import norm
 
DEVICE = torch.device('cpu')
RECORD_TIME = False

def _train_models(_f_model, _c_model_list:list, verbose:bool=False, regularize:bool=False)->Tuple:
    '''
    Train models
    '''
    c_num = len(_c_model_list)
    if regularize:
        _f_model.train_model_kneighbor_collision()
        for c_idx in range(c_num):
            _c_model_list[c_idx].train_model_kneighbor_collision()
    else:
        _f_model.train_model(verbose=False)
        for c_idx in range(c_num):
            _c_model_list[c_idx].train_model(verbose=False)
    return _f_model, _c_model_list

def init_rounds(data_size:int, n_init:int, train_times:int, low_dim:bool, spectrum_norm:bool, lr:float, exact_gp:bool, interpolate:bool, x_tensor:torch.Tensor, y_tensor:torch.Tensor, c_tensor_list:list, c_num:int, noisy_obs:bool=False,
                global_noise_constraint=None, regularize:bool=False, ae:Any=None, verbose:bool=False)->Tuple:
    '''
    Init DKL and generate f_ucb for partition
    '''
    observed = np.zeros(data_size)
    observed[:n_init] = 1
    c_observed_list = [np.zeros(data_size) for _ in range(c_num)]
    init_x = x_tensor[:n_init]
    for c_idx in range(c_num):
        c_observed_list[c_idx][:n_init] = 1
    

    init_y = y_tensor[:n_init]
    init_c_list = [c_tensor_list[c_idx][:n_init] for c_idx in range(c_num)]
    if noisy_obs:
        init_y = torch.normal(mean=init_y, std=torch.ones(1)/1e1)
        init_c_list = [torch.normal(mean=init_c_list[c_idx], std=torch.ones(1)/1e3) for c_idx in range(c_num)]
    # NOTE: AE is shared for f and c
    _f_model = DKL(init_x, init_y.squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, noise_constraint=global_noise_constraint, pretrained_nn=ae, interpolate=interpolate)
    _c_model_list = [DKL(init_x, init_c_list[c_idx].squeeze(), n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, 
                        exact_gp=exact_gp, noise_constraint=global_noise_constraint, pretrained_nn=ae, interpolate=interpolate) for c_idx in range(c_num)]

    # train models
    _f_model, _c_model_list = _train_models(_f_model, _c_model_list, verbose=verbose, regularize=regularize)

    f_lcb, f_ucb = _f_model.CI(x_tensor.to(DEVICE))
    c_lcb_list, c_ucb_list = model_list_CI(_c_model_list, x_tensor, DEVICE)
    return observed, c_observed_list, init_x, init_y, init_c_list, _f_model, _c_model_list, f_lcb, f_ucb, c_lcb_list, c_ucb_list

def finialize_rounds(util_array:np.ndarray, roi_filter:np.ndarray, _cbo_m:DK_BO_AE_C_M_DEC, x_tensor:torch.Tensor, y_tensor:torch.Tensor, c_tensor_list:list, c_num:int,
                    observed:np.ndarray, c_observed_list:list, train_times:int, low_dim:bool, spectrum_norm:bool, lr:float, exact_gp:bool, interpolate:bool, 
                    global_noise_constraint=None, regularize:bool=False, ae:Any=None, retrain_nn:bool=True, verbose:bool=False)->None:
    '''
    Finialize DKL and generate f_ucb for partition
    '''
    for c_idx in range(c_num):
        c_observed_list[c_idx][_cbo_m.c_observed_list[c_idx] == 1] = 1

    # update model and therefore the confidence intervals for filtering
    _f_model = DKL(x_tensor[observed==1], y_tensor[observed==1].squeeze() if sum(observed) > 1 else y_tensor[observed==1],  
                n_iter=train_times, low_dim=low_dim, pretrained_nn=ae, retrain_nn=retrain_nn, lr=lr, spectrum_norm=spectrum_norm,
                exact_gp=exact_gp, noise_constraint=global_noise_constraint, interpolate=interpolate)
    
    _c_model_list = [DKL(x_tensor[c_observed_list[c_idx]==1], c_tensor_list[c_idx][c_observed_list[c_idx]==1].squeeze() if sum(c_observed_list[c_idx]) > 1 else c_tensor_list[c_idx][c_observed_list[c_idx]==1], 
                            n_iter=train_times, low_dim=low_dim, lr=lr, spectrum_norm=spectrum_norm, 
                            exact_gp=exact_gp, noise_constraint=global_noise_constraint, pretrained_nn=ae, interpolate=interpolate) for c_idx in range(c_num)]

    # train models
    _f_model, _c_model_list = _train_models(_f_model, _c_model_list, verbose=verbose, regularize=regularize)

    f_lcb, f_ucb = _f_model.CI(x_tensor.to(DEVICE))
    c_lcb_list, c_ucb_list = model_list_CI(_c_model_list, x_tensor, DEVICE)
    return observed, c_observed_list, _f_model, _c_model_list, f_lcb, f_ucb, c_lcb_list, c_ucb_list

def export_results(save_path:str, name:str, n_repeat:int, n_iter:int, acq:str, lr:float,  ci_intersection:bool, verbose:bool,
                    beta:float=0, interpolate:bool=False, exact_gp:bool=False, filter_beta:float=0,  retrain_interval:int=1,
                    filter_interval:int=1, train_times:int=10, executionTime:float=0, plot_result:bool=False, save_result:bool=False,
                    reg_output_record:np.ndarray=None, ratio_output_record:np.ndarray=None, max_LUCB_interval_record:np.ndarray=None, 
                    reg_record:np.ndarray=None, ratio_record:np.ndarray=None)->None:
    '''
    Export results: plot & save
    '''
    _file_prefix = f"Figure_{name}{'-InterP' if interpolate else ''}{'-Exact' if exact_gp else ''}-B{beta:.2f}-FB{filter_beta:.2f}-RI{retrain_interval}"
    _file_postfix = f"-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}{'-sec' if ci_intersection else ''}"
    
    # plot results
    if RECORD_TIME:
        _file_postfix = f"{_file_postfix}-{executionTime:.2e}s"
    if plot_result:
        # regret
        fig = plt.figure()
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/Regret-{_file_prefix}{_file_postfix}"
        plt.savefig(f"{_path}.png")
        # filter ratio
        fig = plt.figure()
        plt.plot(ratio_output_record)
        plt.ylabel("Ratio")
        plt.xlabel("Iteration")
        plt.title(f'ROI Ratio for {name}')
        _path = f"{save_path}/Ratio-{_file_prefix}{_file_postfix}"
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
        _path = f"{save_path}/CI-{_file_prefix}{_file_postfix}"
        plt.savefig(f"{_path}.png")
        plt.close()
    
    # save results
    if save_result:
        save_res(save_path=save_path, name=f"Regret-{_file_prefix}-", res=reg_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                    init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)
            
        save_res(save_path=save_path, name=f"Ratio-{_file_prefix}-", res=ratio_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                    init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)

        save_res(save_path=save_path, name=f"CI-{_file_prefix}-", res=max_LUCB_interval_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                    init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", ci_intersection=ci_intersection, verbose=verbose,)


def cbo_multi_decoupled(x_tensor, y_tensor, c_tensor_list, constraint_threshold_list, constraint_confidence_list, 
              n_init=10, n_repeat=2, train_times=10, beta=2, regularize=False, low_dim=True, 
            spectrum_norm=False, retrain_interval=1, n_iter=40, filter_interval=1, acq="ci", 
            ci_intersection=True, verbose=True, lr=1e-2, name="test", return_result=True, retrain_nn=True,
            plot_result=False, save_result=False, save_path=None, fix_seed=False,  
            pretrained=False, ae_loc=None, _minimum_pick = 10, 
            _delta = 0.01, filter_beta=.05, exact_gp=False, constrain_noise=False, 
            interpolate=True, noisy_obs=False, check_validity=False, **kwargs):
    '''
    Proposed ROI based method, default acq = ci
    Support Multiple Constraints
    Inputs:
        @x_tensor: input tensor
        @y_tensor: output tensor
        @c_tensor_list: list of constraint tensors
        @constraint_threshold_list: list of constraint thresholds
        @constraint_confidence_list: list of constraint confidence levels
        @n_init: number of initial points
        @n_repeat: number of experiment repeats
        @train_times: number of training iterations
        @beta: beta for confidence interval
        @regularize: whether to regularize the model
        @low_dim: whether to use low dimensional model
        @spectrum_norm: whether to use spectrum normalization
        @retrain_interval: interval for retraining
        @n_iter: number of iterations
        @filter_interval: interval for filtering
        @acq: acquisition function
        @ci_intersection: whether to use intersection of CI
        @verbose: whether to print verbose information
        @lr: learning rate
        @name: name of the experiment
        @return_result: whether to return result
        @retrain_nn: whether to retrain neural network
        @plot_result: whether to plot result
    Outputs:
        @reg_record: regret record

    '''
    ####### configurations
    if constrain_noise:
        global_noise_constraint = gpytorch.constraints.Interval(1e-8, 1e-3)
        roi_noise_constraint = gpytorch.constraints.Interval(1e-5,0.1)
        name = f"{name}-noise_c"
    else:
        global_noise_constraint = None
        roi_noise_constraint = None
    
    _minimum_pick = min(_minimum_pick, n_init)
    c_threshold_list = [norm.ppf(constraint_confidence, loc=constraint_threshold, scale=1) 
                        for constraint_confidence, constraint_threshold in zip(constraint_confidence_list, constraint_threshold_list)]
    c_num = len(c_tensor_list)
    cost_query = kwargs.get('cost_query', np.ones(c_num)) # cost ratio compared to querying f
    feasibility_filter_real = c_tensor_list[0] > c_threshold_list[0]
    for c_idx in range(c_num):
        feasibility_filter_real.logical_and(c_tensor_list[c_idx] > c_threshold_list[c_idx])
    
    assert torch.any(feasibility_filter_real)
    name = name if low_dim else name+'-hd'

    feasible_filter = feasible_filter_gen(c_tensor_list, c_threshold_list)
    max_val = y_tensor[feasible_filter].max()
    max_pos = torch.arange(x_tensor.size(0))
    max_pos = max_pos[feasible_filter]
    max_pos = max_pos[y_tensor[feasible_filter].argmax()]
    reg_record = np.zeros([n_repeat, n_iter])
    ratio_record = np.zeros([n_repeat, n_iter])
    max_LUCB_interval_record = np.zeros([n_repeat, 3, n_iter]) # 0 - Global, 1 - ROI, 2 -- intersection

    ####### init dkl and generate f_ucb for partition
    data_size = x_tensor.size(0)
    assert y_tensor.squeeze().size(0) == data_size
    for c_idx in range(1, c_num):
        assert c_tensor_list[c_idx] .squeeze().size(0) == data_size
        if len(y_tensor.size()) > 2 or len(c_tensor_list[c_idx].size()) > 2 or len(x_tensor.size()) > 2:
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

    startTime = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in tqdm.tqdm(range(n_repeat), desc=f"Experiment Rep"):
            number_f_query = 0
            number_f_query_valid = 0
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
            _init_values = init_rounds(data_size, n_init, train_times, low_dim, spectrum_norm, lr, exact_gp, interpolate, 
                                       x_tensor, y_tensor, c_tensor_list, c_num, noisy_obs=noisy_obs, verbose=verbose,
                                        global_noise_constraint=global_noise_constraint, regularize=regularize, ae=ae)
            
            observed, c_observed_list, init_x, init_y, init_c_list, _f_model, _c_model_list, f_lcb, f_ucb, c_lcb_list, c_ucb_list = _init_values


            ####### each test instance
            iterator = tqdm.tqdm(range(0, n_iter, filter_interval))
            f_c_total_iter = ceil(n_iter)
            for iter in iterator:
                # optimization CI
                if default_beta:
                    _search_space_size = x_tensor.size(0)
                    _constraint_num = c_num
                    beta = (2 * np.log((_search_space_size * 2 * (_constraint_num + 1) * n_iter /_delta))) ** 0.5                 # analytic beta
                _f_lcb, _f_ucb = beta_CI(f_lcb, f_ucb, beta)
                _c_lcb_list, _c_ucb_list = [None for _ in range(c_num)], [None for _ in range(c_num)]
                for c_idx in range(c_num):
                    _c_lcb_list[c_idx], _c_ucb_list[c_idx] = beta_CI(c_lcb_list[c_idx], c_ucb_list[c_idx], beta)
                
                # Not intersection of all historical CIs
                f_max_test_x_lcb, f_min_test_x_ucb = _f_lcb.clone(), _f_ucb.clone()    
                c_max_test_x_lcb_list, c_min_test_x_ucb_list = [_c_lcb.clone() for _c_lcb in _c_lcb_list], [_c_ucb.clone() for _c_ucb in _c_ucb_list]

                # Identify f_roi, csi, cui, c_roi, and general ROI
                if default_fbeta:
                    filter_beta = beta
                observed_num = observed.sum()
                # filter_on_intersect = default_fbeta and _minimum_pick < observed_num
                _max_growth = 4
                _min_c_roi = min(100, feasible_filter.sum())
                _rate_growth = 1.2
                _filter_beta = filter_beta
                filter_on_intersect = False
                for i in range(_max_growth):
                    if filter_on_intersect:
                        _f_filter_lcb, _f_filter_ucb = f_max_test_x_lcb, f_min_test_x_ucb
                        _c_filter_lcb_list, _c_filter_ucb_list = c_max_test_x_lcb_list, c_min_test_x_ucb_list
                    else:
                        _f_filter_lcb, _f_filter_ucb = beta_CI(f_lcb, f_ucb, _filter_beta)
                        _c_filter_lcb_list, _c_filter_ucb_list = [None for _ in range(c_num)], [None for _ in range(c_num)]
                        for c_idx in range(c_num):
                            _c_filter_lcb_list[c_idx], _c_filter_ucb_list[c_idx] = beta_CI(c_lcb_list[c_idx], c_ucb_list[c_idx], _filter_beta)
                        
                    
                    c_sci_filter_list, c_roi_filter_list, c_uci_filter_list = [None for _ in range(c_num)], [None for _ in range(c_num)], [None for _ in range(c_num)]
                    

                    for c_idx in range(c_num):
                        c_sci_filter_list[c_idx] = _c_filter_lcb_list[c_idx] >= c_threshold_list[c_idx]
                        c_roi_filter_list[c_idx] = _c_filter_ucb_list[c_idx] >= c_threshold_list[c_idx]
                        c_uci_filter_list[c_idx] = c_roi_filter_list[c_idx].logical_xor(c_sci_filter_list[c_idx]) 

                    c_sci_filter = c_sci_filter_list[0].clone() # single c_sci
                    for c_idx in range(c_num):
                        c_sci_filter = c_sci_filter.logical_and(c_sci_filter_list[c_idx])
                    
                    # check if filter_beta is appropriate
                    _c_roi_filter = c_roi_filter_list[0].clone()
                    for c_roi_filter in c_roi_filter_list: 
                        _c_roi_filter = _c_roi_filter.logical_and(c_roi_filter)
                    _c_roi_filter_size = sum(_c_roi_filter)
                    if _c_roi_filter_size > _min_c_roi:
                        break
                    else:
                        _filter_beta = _filter_beta * _rate_growth

                f_roi_threshold = _f_filter_lcb[c_sci_filter.squeeze()].max() if torch.any(c_sci_filter) else -torch.inf         # single f_roi
                f_roi_filter = _f_filter_ucb >= f_roi_threshold


                roi_filter = f_roi_filter.clone()           # single general roi
                for c_roi_filter in c_roi_filter_list:      # general ROI.
                    roi_filter = roi_filter.logical_and(c_roi_filter)
                
                if sum(roi_filter[observed==1]) <= _minimum_pick:   # avoid too few points in ROI
                    c_ucb_observed_min = torch.min(torch.cat([c_ucb[observed==1].reshape(1,-1) for c_ucb in c_ucb_list], dim=0), dim=0).values
                    _, indices = torch.topk(c_ucb_observed_min, min(_minimum_pick, c_ucb_observed_min.size(0)))
                    for idx in indices:
                        roi_filter[util_array[observed==1][idx]] = 1

                for c_idx in range(c_num):              # c_uci_list intersects general ROI
                    c_uci_filter_list[c_idx] = c_uci_filter_list[c_idx].logical_and(roi_filter)


                # inherit previous data
                filter_ratio = roi_filter.sum()/data_size
                observed_unfiltered = observed      # observed and not filtered outs (update: avoid roi filtering)
                if iter == 0:
                    init_x = x_tensor[observed_unfiltered==1]
                    init_y = y_tensor[observed_unfiltered==1]
                else: # beacuse some points could be evaluated twice
                    init_x = _cbo_m.init_x_observed
                    init_y = _cbo_m.init_y_observed

                c_observed_unfiltered_list = [c_observed_list[c_idx] for c_idx in range(c_num)]
                for c_idx in range(c_num):
                    if iter == 0:
                        init_c_x_list = [x_tensor[c_observed_unfiltered_list[c_idx]==1] for c_idx in range(c_num)]
                        init_c_list = [c_tensor_list[c_idx][c_observed_unfiltered_list[c_idx]==1] for c_idx in range(c_num)]
                    else: # beacuse some points could be evaluated twice
                        init_c_x_list = _cbo_m.init_c_x_observed_list
                        init_c_list = _cbo_m.init_c_observed_list
                
                observed_num = observed_unfiltered.sum() + sum([c_observed_list[c_idx].sum() for c_idx in range(c_num)])

                if noisy_obs:
                    _obs_count = sum(observed_unfiltered==1)
                    init_y = torch.normal(mean=init_y, std=torch.ones(1)/1e1)
                    init_c_list = [torch.normal(mean=c_tensor, std=torch.ones(1)/1e3) for c_tensor in init_c_list]

                # optimization
                _cbo_m = DK_BO_AE_C_M_DEC(x_tensor, y_tensor, c_tensor_list, roi_filter, c_uci_filter_list, lr=lr, spectrum_norm=spectrum_norm, low_dim=low_dim,
                                    n_init=n_init,  train_iter=train_times, regularize=regularize, dynamic_weight=False,  retrain_nn=retrain_nn, c_threshold_list=c_threshold_list,
                                    max=max_val, pretrained_nn=ae, verbose=verbose, init_x=init_x, init_y=init_y, init_c_x_list=init_c_x_list, init_c_list=init_c_list, exact_gp=exact_gp, noise_constraint=roi_noise_constraint,
                                    f_model=_f_model, c_model_list=_c_model_list, observed=observed, c_observed_list=c_observed_list, interpolate_prior=interpolate, noisy_obs=noisy_obs)

                _roi_f_lcb, _roi_f_ucb = _cbo_m.f_model.CI(x_tensor)
                _roi_c_lcb_list, _roi_c_ucb_list  = model_list_CI(_cbo_m.c_model_list, x_tensor, DEVICE)

                # if ci_intersection:
                if not (default_beta): # only for visualization & intersection
                    if beta <= 1e2 and beta >= 3:
                        _roi_beta = min(1e2, max(1e-2, f_ucb.max()/_roi_f_ucb.max()))
                    else:
                        _roi_beta = beta
                else:
                    _search_space_size = x_tensor[roi_filter].shape[0]
                    _constraint_num = c_num
                    _roi_beta = (2 * np.log((_search_space_size * 2 * (_constraint_num + 1) * n_iter /_delta))) ** 0.5
                    _roi_beta = (2 * np.log((x_tensor[roi_filter].shape[0] * (np.pi * (iter + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta

  
                # intersection of ROI CI and global CI
                if ci_intersection:
                    f_max_test_x_lcb, f_min_test_x_ucb, _roi_f_lcb_scaled,  _roi_f_ucb_scaled  = intersecting_ROI_globe(f_max_test_x_lcb, f_min_test_x_ucb, _roi_f_lcb, _roi_f_ucb, _roi_beta, roi_filter)
                    for c_idx, (c_max_test_x_lcb, c_min_test_x_ucb, _roi_c_lcb, _roi_c_ucb) in enumerate(zip(c_max_test_x_lcb_list, c_min_test_x_ucb_list, _roi_c_lcb_list, _roi_c_ucb_list)):
                        c_max_test_x_lcb_list[c_idx], c_min_test_x_ucb_list[c_idx], _, _ = intersecting_ROI_globe(c_max_test_x_lcb, c_min_test_x_ucb, _roi_c_lcb, _roi_c_ucb, _roi_beta, roi_filter)
                else:
                    _, _, _roi_f_lcb_scaled,  _roi_f_ucb_scaled  = intersecting_ROI_globe(f_max_test_x_lcb, f_min_test_x_ucb, _roi_f_lcb, _roi_f_ucb, _roi_beta, roi_filter)

                # optimize f and learn c
                interval_query_ceil = f_c_total_iter - iter
                query_num = min(filter_interval, interval_query_ceil) 
                assert query_num > 0
                _acq = 'lcb' if f_c_total_iter - iter <= filter_interval else acq
                _roi_beta_passed_in = _roi_beta  if not (default_beta) else 0 # allow it to calculate internal ROI_beta
                _cbo_m.query_f_c_decoupled(n_iter=query_num, acq=_acq, study_interval=10, study_res_path=save_path,  if_tqdm=False, 
                               retrain_interval=retrain_interval, ci_intersection=ci_intersection, 
                               f_max_test_x_lcb=f_max_test_x_lcb, f_min_test_x_ucb=f_min_test_x_ucb,
                               c_max_test_x_lcb_list=c_max_test_x_lcb_list, c_min_test_x_ucb_list=c_min_test_x_ucb_list, 
                               beta=_roi_beta_passed_in, cost_query=cost_query, check_validity=check_validity)

                # update records
                _step_size = iter + query_num
                number_f_query += (_cbo_m._model_idx_list == -1).sum()
                if query_num == 1:
                    number_f_query_valid += all(_cbo_m.feasiblility_check_list)
                reg_record[rep, iter:_step_size] = _cbo_m.regret[-query_num:]
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
                # 'beta': beta, 'fbeta': filter_beta, "roi_beta": _roi_beta,
                _iterator_info = { "regret":reg_record[rep, :_step_size].min(), "Filter Ratio": filter_ratio.detach().item(), 
                                #   "Filter Gap": _filter_gap.detach().item(), "F roi threshold": f_roi_threshold.detach().item(),
                                #   'roi noise': _cbo_m.f_model.likelihood.noise.detach().item(), 'global noise': _f_model.likelihood.noise.detach().item()
                                  }
                roi_y_min, roi_y_max = y_tensor[roi_filter].min().detach().item(), y_tensor[roi_filter].max().detach().item()
                if roi_filter.logical_and(feasible_filter).any():
                    roi_fy_min, roi_fy_max = y_tensor[roi_filter.logical_and(feasible_filter)].min(), y_tensor[roi_filter.logical_and(feasible_filter)].max()
                else:
                    roi_fy_min, roi_fy_max = 0, 0
                _iterator_info['ROI Size'] = f"{roi_filter.sum().detach().item()}"
                _iterator_info['ROI Accuracy'] = f"{(roi_filter.logical_and(feasible_filter).sum()/roi_filter.sum()).detach().item():.2%}"
                # _iterator_info['Csi Accuracy'] = f"{(c_sci_filter.logical_and(feasible_filter).sum()/c_sci_filter.sum()).detach().item():.2%}"
                _iterator_info['Sci Accuracy'] = f"{(c_sci_filter.logical_and(feasible_filter).sum()/c_sci_filter.sum()).detach().item():.2%}"
                _iterator_info['ROI Y range'] = f"{roi_fy_min:.2f}, {roi_fy_max:.2f}"
                # _iterator_info['filter_on_intersect'] = filter_on_intersect
                _iterator_info['# F Queries'] = f"{number_f_query_valid}/{number_f_query}"
                _iterator_info['beta'] = beta
                _iterator_info[f'Acq Picked'] = f"c{_cbo_m._c_acq_list_max_value:.2e}, f{_cbo_m._f_acq_value:.2e}"
                _iterator_info['UCB Optimum'] = f"{_roi_f_ucb[max_pos]:.2e}"
                iterator.set_postfix(_iterator_info)

                # update model
                _final_values = finialize_rounds(util_array, roi_filter, _cbo_m, x_tensor, y_tensor, c_tensor_list,
                                             c_num, observed, c_observed_list, train_times, low_dim, spectrum_norm, 
                                            lr, exact_gp, interpolate, global_noise_constraint=global_noise_constraint, 
                                            regularize=regularize, ae=ae, retrain_nn=retrain_nn, verbose=verbose)
                observed, c_observed_list, _f_model, _c_model_list, f_lcb, f_ucb, c_lcb_list, c_ucb_list = _final_values


    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    ratio_output_record = ratio_record.mean(axis=0)
    
    beta = 0 if default_beta else beta # for record


    executionTime = (time.time() - startTime)

    ### Export results
    export_results(save_path=save_path, name=name, n_repeat=n_repeat, n_iter=n_iter, acq=acq, lr=lr,  ci_intersection=ci_intersection, verbose=verbose,
                    beta=beta, interpolate=interpolate, exact_gp=exact_gp, filter_beta=filter_beta,  retrain_interval=retrain_interval,
                    filter_interval=filter_interval, train_times=train_times, executionTime=executionTime, plot_result=plot_result, save_result=save_result,
                    reg_output_record=reg_output_record, ratio_output_record=ratio_output_record, max_LUCB_interval_record=max_LUCB_interval_record, 
                    reg_record=reg_record, ratio_record=ratio_record)

    if return_result:
        return reg_record
    else:
        return _f_model, _c_model_list, _cbo_m


