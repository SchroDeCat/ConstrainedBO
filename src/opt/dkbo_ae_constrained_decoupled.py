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

from ..models import DKL
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..utils import feasible_filter_gen
from .dkbo_ae_constrained import DK_BO_AE_C_M

DEVICE = torch.device('cpu')


class DK_BO_AE_C_M_DEC(DK_BO_AE_C_M):
    def __init__(self, x_tensor, y_tensor, c_tensor_list, roi_filter, c_uci_filter_list, c_threshold_list, 
                 n_init: int = 10, lr=0.000001, train_iter: int = 10, regularize=True, spectrum_norm=False, 
                 dynamic_weight=False, verbose=False, max=None, robust_scaling=True, pretrained_nn=None, 
                 low_dim=True, record_loss=False, retrain_nn=True, exact_gp=False, noise_constraint=None, 
                 output_scale_constraint=None, standardize=True, **kwargs):
        # scale input
        ScalerClass = RobustScaler if robust_scaling else StandardScaler
        self.scaler = ScalerClass().fit(x_tensor)
        if standardize:
            x_tensor = self.scaler.transform(x_tensor)
            self.x_tensor = torch.from_numpy(x_tensor).float()
        else:
            self.x_tensor = x_tensor.float()
        # init vars
        self.regularize = regularize
        self.lr = lr
        self.low_dim = low_dim
        self.c_num = len(c_tensor_list)
        self.verbose = verbose
        self.n_init = n_init
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.dynamic_weight = dynamic_weight
        
        self.y_tensor = y_tensor.float()
        self.c_tensor_list = [c_tensor.float() for c_tensor in c_tensor_list]
        self.data_size = self.x_tensor.size(0)
        self.train_iter = train_iter
        self.retrain_nn = retrain_nn
        self.output_scale_constraint = output_scale_constraint
        feasible_filter = feasible_filter_gen(c_tensor_list, c_threshold_list)
        self.maximum = torch.max(self.y_tensor[feasible_filter]) if max==None else max
        self.max_regret = self.maximum - torch.min(self.y_tensor)

        self.noisy_obs = kwargs.get('noisy_obs', False)
        self.init_x = kwargs.get("init_x", self.x_tensor[:n_init])        
        self.init_y = kwargs.get("init_y", self.y_tensor[:n_init])
        self.init_c_x_list = kwargs.get("init_c_x_list", [self.x_tensor[:n_init] for _ in range(self.c_num)])
        self.c_n_init_list  = [self.init_c_x_list[c_idx].size(0) for c_idx in range(self.c_num)]
        self.init_c_list = kwargs.get("init_c_list", [c_tensor[:n_init] for c_tensor in self.c_tensor_list])
        for c_idx in range(self.c_num):
            assert self.init_c_x_list[c_idx].size(0) == self.init_c_list[c_idx].size(0)
        if "init_x" in kwargs:
            self.init_x = torch.from_numpy(self.scaler.transform(self.init_x)).float()
        self.spectrum_norm = spectrum_norm
        self.exact = exact_gp # exact GP overide
        self.noise_constraint = noise_constraint
        self.observed = kwargs.get("observed", np.zeros(self.x_tensor.size(0)).astype("int"))
        self.c_observed_list =  kwargs.get("c_observed_list", [np.zeros(self.x_tensor.size(0)).astype("int") for _ in range(self.c_num)])

        self.pretrained_nn = pretrained_nn
        self.roi_filter = roi_filter
        self.c_uci_filter_list = c_uci_filter_list
        self.c_threshold_list = c_threshold_list

        self.interpolate = kwargs.get('interpolate_prior', False)

        # load input model / train the ROI model
        f_model = kwargs.get("f_model", None)
        c_model_list = kwargs.get("c_model_list", None)

        if f_model is None:
            self.f_model = DKL(self.init_x, self.init_y.squeeze(),
                            n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, 
                            pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                            noise_constraint = self.noise_constraint, output_scale_constraint=self.output_scale_constraint,
                            interpolate = self.interpolate,)
        else:
            self.f_model = f_model
        
        if c_model_list is None:
            self.c_model_list = [DKL(self.init_c_x_list[c_idx], self.init_c_list[c_idx].squeeze(), 
                                    n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim,
                                    spectrum_norm=spectrum_norm, exact_gp=exact_gp, pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn,
                                    noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint, 
                                    interpolate = self.interpolate,) for c_idx in range(self.c_num)]
        else:
            self.c_model_list = c_model_list
        
        self.record_loss = record_loss

        if self.record_loss:
            assert not (pretrained_nn is None)
            self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), 
                                   n_iter=self.train_iter, low_dim=self.low_dim, 
                                   pretrained_nn=None, lr=self.lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                                   interpolate = self.interpolate,)
            self.f_loss_record = {"DK-AE":[], "DK":[]}
        # self.cuda = torch.cuda.is_available()
        self.cuda = False

        self.train()
        

    def periodical_retrain_decoupled(self, idx, retrain_interval, model_idx=-1): 
        '''
        Note:   in the decoupled setting, only the model that is queries shall be updated to save cost, o.w. it is like ensemble
        '''
        i = idx
        # retrain for reuse
        if i % retrain_interval != 0 and self.low_dim: # allow skipping retrain in low-dim setting
            self._f_state_dict_record = self.f_model.feature_extractor.state_dict()
            self._f_output_scale_record = self.f_model.model.covar_module.base_kernel.outputscale
            self._f_length_scale_record = self.f_model.model.covar_module.base_kernel.base_kernel.lengthscale
            self._c_state_dict_record_list = [self.c_model_list[c_idx].feature_extractor.state_dict() for c_idx in range(self.c_num)]
            self._c_output_scale_record_list = [self.c_model_list[c_idx].model.covar_module.base_kernel.outputscale for c_idx in range(self.c_num)]
            self._c_length_scale_record_list = [self.c_model_list[c_idx].model.covar_module.base_kernel.base_kernel.lengthscale for c_idx in range(self.c_num)]

        if model_idx == -1:
            model_observed_filter = np.array([True for _ in range(self.init_x.size(0))])
            for query_idx, query_model_idx in enumerate(self._model_idx_list):
                if query_model_idx != model_idx:
                    model_observed_filter[query_idx+self.n_init] = False
            self.f_model = DKL(self.init_x[model_observed_filter], self.init_y[model_observed_filter].squeeze(),
                                    n_iter=self.train_iter, lr= self.lr, 
                                    low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                    spectrum_norm=self.spectrum_norm, exact_gp=self.exact, 
                                    noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint,
                                    interpolate = self.interpolate,)
        elif model_idx >= 0:
            model_observed_filter = np.array([True for _ in range(self.init_c_x_list[model_idx].size(0))])
            for query_idx, query_model_idx in enumerate(self._model_idx_list):
                if query_model_idx != model_idx:
                    model_observed_filter[query_idx+self.c_n_init_list[model_idx]] = False

            self.c_model_list[model_idx] = DKL(self.init_c_x_list[model_idx][model_observed_filter], self.init_c_list[model_idx][model_observed_filter].squeeze(),
                                n_iter=self.train_iter, lr= self.lr, 
                                low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                spectrum_norm=self.spectrum_norm, exact_gp=self.exact, 
                                noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint,
                                interpolate = self.interpolate,)
        
        if self.record_loss:
            self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), 
                                   n_iter=self.train_iter, 
                                   low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=self.spectrum_norm, 
                                   exact_gp=self.exact, interpolate = self.interpolate,)
        
        if i % retrain_interval != 0 and self.low_dim:
            self.f_model.feature_extractor.load_state_dict(self._f_state_dict_record, strict=False)
            self.f_model.model.covar_module.base_kernel.outputscale = self._f_output_scale_record
            self.f_model.model.covar_module.base_kernel.base_kernel.lengthscale = self._f_length_scale_record
            for c_idx in range(self.c_num):
                self.c_model_list[c_idx].feature_extractor.load_state_dict(self._c_state_dict_record_list[c_idx], strict=False)
                self.c_model_list[c_idx].model.covar_module.base_kernel.outputscale = self._c_output_scale_record_list[c_idx]
                self.c_model_list[c_idx].model.covar_module.base_kernel.base_kernel.lengthscale = self._c_length_scale_record_list[c_idx]
        else:
            self.train()
        if self.record_loss:
            self.f_loss_record["DK-AE"].append(self.f_model.mae_record[-1])
            self.f_loss_record["DK"].append(self._f_pure_dkl.mae_record[-1])

    def update_obs_decoupled(self, candidate_idx, model_idx):
        '''
        Note:   even in the decoupled setting, 
                the model_idx is used to determine which model to be evaluated, 
                we still track all corresponding obs for each model for easier regret calculation
        '''
        self.init_x = torch.cat([self.init_x, self.x_tensor[candidate_idx].reshape(1,-1)], dim=0)
        for c_idx in range(self.c_num):
            self.init_c_x_list[c_idx] = torch.cat([self.init_c_x_list[c_idx], self.x_tensor[candidate_idx].reshape(1,-1)], dim=0)
        

        if not self.noisy_obs:
            self.init_y = torch.cat([self.init_y, self.y_tensor[candidate_idx].reshape(1,-1)])
            for c_idx in range(self.c_num):
                self.init_c_list[c_idx] = torch.cat([self.init_c_list[c_idx], self.c_tensor_list[c_idx][candidate_idx].reshape(1,-1)])
                for c_idx in range(self.c_num):
                    assert self.init_c_x_list[c_idx].size(0) == self.init_c_list[c_idx].size(0)
        else:
            self.init_y = torch.cat([self.init_y, torch.normal(self.y_tensor[candidate_idx].reshape(1,-1), std=torch.ones(1)/1e1)])
            for c_idx in range(self.c_num):
                self.init_c_list[c_idx] = torch.cat([self.init_c_list[c_idx], torch.normal(self.c_tensor_list[c_idx][candidate_idx].reshape(1,-1), std=torch.ones(1)/1e3)])
                for c_idx in range(self.c_num):
                    assert self.init_c_x_list[c_idx].size(0) == self.init_c_list[c_idx].size(0)
        assert self.init_x.size(0) == self.init_y.size(0)
        

        # update observed record
        if model_idx == -1:
            self.observed[candidate_idx] = 1
            observed_num = self.observed.sum()
        else:
            self.c_observed_list[model_idx][candidate_idx] = 1
        return

    def update_regret_decoupled(self, idx, candidate_idx, model_idx, check_validity:bool=True):
        '''
        Note: only use the target fidelity to calculate regret
        Input:
            idx: the index of the iteration
            candidate_idx: the index of the pts to be evaluated
            model_idx: the index of the model to be evaluated
        
        '''
        self.feasiblility_check_list = [False]
        if model_idx == -1:
            self.feasiblility_check_list = [self.init_c_list[c_idx][-1] >= self.c_threshold_list[c_idx] for c_idx in range(self.c_num)]
            if all(self.feasiblility_check_list) or not check_validity:
                # when there is feasible points
                reward = self.init_y[-1]
                self.regret[idx] = self.maximum - reward
                self.regret[:idx] = np.minimum.accumulate(self.regret[:idx]) # guarantee monotonicity
                return

        # o.w.: when there is no feasible points or the objective is not evaluated this time
        self.regret[idx] = self.max_regret



    def query_f_c_decoupled(self, n_iter:int=10, acq="ci", retrain_interval:int=1, check_validity=False, **kwargs):
        '''
        Note:
            1. f and c are decoupled query
            2. The model record both the idx of the pts to be evaluated, and the model idx (with f as -1) to be evaluated
            3. The model idx is used to determine which model to be retrained
        '''
        for c_idx in range(self.c_num):
            assert self.init_c_x_list[c_idx].size(0) == self.init_c_list[c_idx].size(0)
        assert self.init_x.size(0) == self.init_y.size(0)
        self.regret = np.zeros(n_iter)
        if_tqdm = kwargs.get("if_tqdm", False)
        early_stop = kwargs.get("early_stop", True)
        iterator = tqdm.tqdm(range(n_iter)) if if_tqdm else range(n_iter)
        util_array = np.arange(self.data_size)
        ci_intersection = kwargs.get("ci_intersection", False)
        f_max_test_x_lcb = kwargs.get("f_max_test_x_lcb", None)
        f_min_test_x_ucb = kwargs.get("f_min_test_x_ucb", None)
        c_max_test_x_lcb_list = kwargs.get("c_max_test_x_lcb_list", None)
        c_min_test_x_ucb_list = kwargs.get("c_min_test_x_ucb_list", None)
        cost_query = kwargs.get('cost_query', np.ones(self.c_num))

        beta = kwargs.get("beta", 1)
        _delta = kwargs.get("delta", .01)

        real_beta = beta <= 0 # if using analytic beta
        _candidate_idx_list = np.zeros(n_iter)
        self._model_idx_list = np.zeros(n_iter).astype("int")

        ### optimization loop
        for i in iterator:
            _candidate_idx_c_list = [None for i in range(self.c_num)]
            _acq = 'ci'
            # _acq = 'lcb' if i // 2 and acq == 'ci' else 'ci'
            if real_beta:
                _search_space_size = self.x_tensor.size(0)
                _constraint_num = self.c_num
                beta = (2 * np.log((_search_space_size * 2 * (_constraint_num + 1) * n_iter /_delta))) ** 0.5
            # acq values
            if ci_intersection:
                assert not( f_max_test_x_lcb is None or f_min_test_x_ucb is None)
                _candidate_idx_f = self.f_model.intersect_CI_next_point(self.x_tensor[self.roi_filter], 
                                                                        max_test_x_lcb=f_max_test_x_lcb[self.roi_filter], 
                                                                        min_test_x_ucb=f_min_test_x_ucb[self.roi_filter], 
                                                                        acq=_acq if _acq != 'ci' else 'cucb', beta=beta, return_idx=True)

                for c_idx, (c_uci_filter, c_max_test_x_lcb, c_min_test_x_ucb) in enumerate(zip(self.c_uci_filter_list, c_max_test_x_lcb_list, c_min_test_x_ucb_list)):
                    if sum(c_uci_filter) > 0:
                        _candidate_idx_c = self.c_model_list[c_idx].intersect_CI_next_point(self.x_tensor[c_uci_filter], 
                                                                max_test_x_lcb=c_max_test_x_lcb[c_uci_filter], 
                                                                min_test_x_ucb=c_min_test_x_ucb[c_uci_filter], 
                                                                acq=_acq, beta=beta, return_idx=True)
                        _candidate_idx_c_list[c_idx] = _candidate_idx_c
            else:
                _candidate_idx_f = self.f_model.next_point(self.x_tensor[self.roi_filter], acq if acq != 'ci' else 'cucb', "love", return_idx=True, beta=beta,)
                for c_idx, c_uci_filter in enumerate(self.c_uci_filter_list):
                    if sum(c_uci_filter) > 0:
                        _candidate_idx_c = self.c_model_list[c_idx].next_point(self.x_tensor[c_uci_filter], acq, "love", return_idx=True, beta=beta,)
                        _candidate_idx_c_list[c_idx] = _candidate_idx_c
            
            # locate max acq
            if False:
            # if i // 2:
                _acq_value = torch.prod(torch.cat([model.acq_value for model in self.c_model_list]), dim=0)
                _acq_value = torch.mul(_acq_value. f_model.acq_value)
                candidate_idx = _acq_value.argmax()
            else:
                _f_acq = self.f_model.acq_val[_candidate_idx_f]
                _c_acq_list = torch.tensor([float('-inf') for _ in range(self.c_num)])
                for c_idx, c_uci_filter in enumerate(self.c_uci_filter_list):
                    if sum(c_uci_filter) > 0:
                        _c_acq_list[c_idx] = self.c_model_list[c_idx].acq_val[_candidate_idx_c_list[c_idx]] / cost_query[c_idx]
                _c_acq_list_max = torch.max(_c_acq_list, dim=0)

                self._c_acq_list_max_value = _c_acq_list_max.values 
                self._f_acq_value = _f_acq
                if _c_acq_list_max.values > _f_acq:
                    assert _c_acq_list_max.values > float("-inf")
                    _c_idx = _c_acq_list_max.indices
                    candidate_idx = util_array[self.c_uci_filter_list[_c_idx]][_candidate_idx_c_list[_c_idx]]
                    self._model_idx_list[i] = _c_idx
                else:
                    candidate_idx = util_array[self.roi_filter][_candidate_idx_f]
                    self._model_idx_list[i] = -1

            self._candidate_idx = candidate_idx
            self._model_idx = self._model_idx_list[i]
            # update obs
            _candidate_idx_list[i] = candidate_idx
            self.update_obs_decoupled(candidate_idx=candidate_idx, model_idx=self._model_idx_list[i])

            # Retrain
            self.periodical_retrain_decoupled(i, retrain_interval, model_idx=self._model_idx)

            # regret & early stop
            self.update_regret_decoupled(idx=i, candidate_idx=candidate_idx, model_idx=self._model_idx, check_validity=check_validity)

            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i], "Internal_beta": beta})
