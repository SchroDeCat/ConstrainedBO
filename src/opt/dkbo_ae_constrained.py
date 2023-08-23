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

DEVICE = torch.device('cpu')

class DK_BO_AE_C():
    """
    Initialize the network with auto-encoder for constrained setting
    """
    def __init__(self, x_tensor, y_tensor, c_tensor, roi_filter, c_uci_filter, optimization_ratio, c_threshold,
                    n_init:int=10, lr=1e-6, train_iter:int=10, regularize=True, spectrum_norm=False,
                    dynamic_weight=False, verbose=False, max=None, robust_scaling=True, pretrained_nn=None, low_dim=True,
                    record_loss=False, retrain_nn=True, exact_gp=False, noise_constraint=None, output_scale_constraint=None,
                    **kwargs):

        # scale input
        ScalerClass = RobustScaler if robust_scaling else StandardScaler
        self.scaler = ScalerClass().fit(x_tensor)
        x_tensor = self.scaler.transform(x_tensor)
        # init vars
        self.regularize = regularize
        self.lr = lr
        self.low_dim = low_dim
        self.verbose = verbose
        self.n_init = n_init
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.dynamic_weight = dynamic_weight
        self.x_tensor = torch.from_numpy(x_tensor).float()
        self.y_tensor = y_tensor.float()
        self.c_tensor = c_tensor.float()
        self.data_size = self.x_tensor.size(0)
        self.train_iter = train_iter
        self.retrain_nn = retrain_nn
        self.output_scale_constraint = output_scale_constraint
        feasible_filter = feasible_filter_gen([c_tensor], [c_threshold])
        self.maximum = torch.max(self.y_tenso[feasible_filter]) if max==None else max
        self.max_regret = self.maximum - torch.min(self.y_tensor)

        self.interpolate = kwargs.get('interpolate_prior', False)

        self.init_x = kwargs.get("init_x", self.x_tensor[:n_init])        
        self.init_y = kwargs.get("init_y", self.y_tensor[:n_init])
        self.init_c = kwargs.get("init_c", self.c_tensor[:n_init])
        assert self.init_x.size(0) == self.init_c.size(0)
        if "init_x" in kwargs:
            self.init_x = torch.from_numpy(self.scaler.transform(self.init_x)).float()
        self.spectrum_norm = spectrum_norm
        self.exact = exact_gp # exact GP overide
        self.noise_constraint = noise_constraint
        self.observed = kwargs.get("observed", np.zeros(self.x_tensor.size(0)).astype("int"))
        self.pretrained_nn = pretrained_nn
        self.roi_filter = roi_filter
        self.c_uci_filter = c_uci_filter
        self.c_threshold = c_threshold
        f_model = kwargs.get("f_model", None)
        c_model = kwargs.get("c_model", None)

        if f_model is None:
            self.f_model = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, 
                            pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                            noise_constraint = self.noise_constraint, output_scale_constraint=self.output_scale_constraint, interpolate_prior = self.interpolate,)
        else:
            self.f_model = f_model
        
        if c_model is None:
            self.c_model = DKL(self.init_x, self.init_c.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, 
                            pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                            noise_constraint = self.noise_constraint, output_scale_constraint=self.output_scale_constraint, interpolate_prior = self.interpolate,)
        else:
            self.c_model = c_model
        
        self.record_loss = record_loss

        if self.record_loss:
            assert not (pretrained_nn is None)
            self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, interpolate_prior = self.interpolate,)
            # self._c_pure_dkl = DKL(self.init_x, self.init_c.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp)
            self.f_loss_record = {"DK-AE":[], "DK":[]}
        self.cuda = torch.cuda.is_available()

        self.train()
    
    def train(self,):
        if self.regularize:
            self.f_model.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
            self.c_model.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        else:
            
            if self.record_loss:
                self._f_pure_dkl.train_model(record_mae=True)
                self.f_model.train_model(record_mae=True)
                self.c_model.train_model(record_mae=True)
            else:
                self.f_model.train_model(verbose=False)
                self.c_model.train_model(verbose=False)

        
    def query_f_c(self, n_iter:int=10, acq="ci", retrain_interval:int=1, **kwargs):
        '''
        First Stage: Query both f and c simultaneously
        '''
        assert self.init_x.size(0) == self.init_c.size(0)
        assert self.init_x.size(0) == self.init_y.size(0)
        self.regret = np.zeros(n_iter)
        if_tqdm = kwargs.get("if_tqdm", False)
        early_stop = kwargs.get("early_stop", True)
        iterator = tqdm.tqdm(range(n_iter)) if if_tqdm else range(n_iter)
        util_array = np.arange(self.data_size)
        ci_intersection = kwargs.get("ci_intersection", False)
        f_max_test_x_lcb = kwargs.get("f_max_test_x_lcb", None)
        f_min_test_x_ucb = kwargs.get("f_min_test_x_ucb", None)
        c_max_test_x_lcb = kwargs.get("c_max_test_x_lcb", None)
        c_min_test_x_ucb = kwargs.get("c_min_test_x_ucb", None)

        beta = kwargs.get("beta", 1)
        _delta = kwargs.get("delta", .2)

        real_beta = beta <= 0 # if using analytic beta
        _candidate_idx_list = np.zeros(n_iter)
        ### optimization loop
        for i in iterator:
            if real_beta:
                beta = (2 * np.log((self.x_tensor.size(0) * (np.pi * (self.init_x.size(0) + 1)) ** 2) /(6 * _delta))) ** 0.5
            if ci_intersection:
                assert not( f_max_test_x_lcb is None or f_min_test_x_ucb is None)
                _candidate_idx_f = self.f_model.intersect_CI_next_point(self.x_tensor[self.roi_filter], 
                                                                        max_test_x_lcb=f_max_test_x_lcb[self.roi_filter], 
                                                                        min_test_x_ucb=f_min_test_x_ucb[self.roi_filter], 
                                                                        acq=acq, beta=beta, return_idx=True)
                if sum(self.c_uci_filter) > 0:
                    _candidate_idx_c = self.c_model.intersect_CI_next_point(self.x_tensor[self.c_uci_filter], 
                                                            max_test_x_lcb=c_max_test_x_lcb[self.c_uci_filter], 
                                                            min_test_x_ucb=c_min_test_x_ucb[self.c_uci_filter], 
                                                            acq=acq, beta=beta, return_idx=True)
            else:
                _candidate_idx_f = self.f_model.next_point(self.x_tensor[self.roi_filter], acq, "love", return_idx=True, beta=beta,)
                if sum(self.c_uci_filter) > 0:
                    _candidate_idx_c = self.c_model.next_point(self.x_tensor[self.c_uci_filter], acq, "love", return_idx=True, beta=beta,)
            
            _f_acq = self.f_model.acq_val[_candidate_idx_f]
            if sum(self.c_uci_filter) > 0:
                _c_acq = self.c_model.acq_val[_candidate_idx_c]
        
            if sum(self.c_uci_filter) > 0 and _c_acq > _f_acq:
                candidate_idx = util_array[self.c_uci_filter][_candidate_idx_c]
            else:
                candidate_idx = util_array[self.roi_filter][_candidate_idx_f]


            _candidate_idx_list[i] = candidate_idx
            self.init_x = torch.cat([self.init_x, self.x_tensor[candidate_idx].reshape(1,-1)], dim=0)
            self.init_y = torch.cat([self.init_y, self.y_tensor[candidate_idx].reshape(1,-1)])
            self.init_c = torch.cat([self.init_c, self.c_tensor[candidate_idx].reshape(1,-1)])
            assert self.init_x.size(0) == self.init_c.size(0)
            assert self.init_x.size(0) == self.init_y.size(0)
            self.observed[candidate_idx] = 1

            # retrain
            if i % retrain_interval != 0 and self.low_dim: # allow skipping retrain in low-dim setting
                self._f_state_dict_record = self.f_model.feature_extractor.state_dict()
                self._f_output_scale_record = self.f_model.model.covar_module.base_kernel.outputscale
                self._f_length_scale_record = self.f_model.model.covar_module.base_kernel.base_kernel.lengthscale
                self._c_state_dict_record = self.c_model.feature_extractor.state_dict()
                self._c_output_scale_record = self.c_model.model.covar_module.base_kernel.outputscale
                self._c_length_scale_record = self.c_model.model.covar_module.base_kernel.base_kernel.lengthscale

            self.f_model = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint, interpolate_prior = self.interpolate,)
            self.c_model = DKL(self.init_x, self.init_c.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint, interpolate_prior = self.interpolate,)
            
            if self.record_loss:
                self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=self.spectrum_norm, exact_gp=self.exact, interpolate_prior = self.interpolate,)
            
            if i % retrain_interval != 0 and self.low_dim:
                self.f_model.feature_extractor.load_state_dict(self._f_state_dict_record, strict=False)
                self.f_model.model.covar_module.base_kernel.outputscale = self._f_output_scale_record
                self.f_model.model.covar_module.base_kernel.base_kernel.lengthscale = self._f_length_scale_record
                
                self.c_model.feature_extractor.load_state_dict(self._c_state_dict_record, strict=False)
                self.c_model.model.covar_module.base_kernel.outputscale = self._c_output_scale_record
                self.c_model.model.covar_module.base_kernel.base_kernel.lengthscale = self._c_length_scale_record
            else:
                self.train()
            if self.record_loss:
                self.f_loss_record["DK-AE"].append(self.f_model.mae_record[-1])
                self.f_loss_record["DK"].append(self._f_pure_dkl.mae_record[-1])

            # regret & early stop
            feasible_obs_filter = self.init_c > self.c_threshold
            if sum(feasible_obs_filter) > 0:
                _max_reward = torch.max(self.init_y[feasible_obs_filter])
                self.regret[i] = self.maximum - _max_reward
            else:
                self.regret[i] = self.max_regret

            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i], "Internal_beta": beta})


    def query_cons(self, n_iter:int=10, acq="ucb", retrain_interval:int=1, **kwargs):
        '''
        Second Stage: Optimize C only simultaneously
        '''
        self.regret = np.zeros(n_iter)
        if_tqdm = kwargs.get("if_tqdm", False)
        early_stop = kwargs.get("early_stop", True)
        iterator = tqdm.tqdm(range(n_iter)) if if_tqdm else range(n_iter)
        util_array = np.arange(self.data_size)
        ci_intersection = kwargs.get("ci_intersection", False)
        c_max_test_x_lcb = kwargs.get("c_max_test_x_lcb", None)
        c_min_test_x_ucb = kwargs.get("c_min_test_x_ucb", None)

        beta = kwargs.get("beta", 1)
        _delta = kwargs.get("delta", .2)

        real_beta = beta <= 0 # if using analytic beta
        _candidate_idx_list = np.zeros(n_iter)
        ### optimization loop
        for i in iterator:
            if real_beta:
                beta = (2 * np.log((self.x_tensor.size(0) * (np.pi * (self.init_x.size(0) + 1)) ** 2) /(6 * _delta))) ** 0.5
            if ci_intersection:
                _candidate_idx_c = self.c_model.intersect_CI_next_point(self.x_tensor[self.c_uci_filter], 
                                                        max_test_x_lcb=c_max_test_x_lcb[self.c_uci_filter], 
                                                        min_test_x_ucb=c_min_test_x_ucb[self.c_uci_filter], 
                                                        acq=acq, beta=beta, return_idx=True)
            else:
                _candidate_idx_c = self.c_model.next_point(self.x_tensor[self.c_uci_filter], acq, "love", return_idx=True, beta=beta,)

            candidate_idx = util_array[self.c_uci_filter][_candidate_idx_c]

            _candidate_idx_list[i] = candidate_idx
            self.init_x = torch.cat([self.init_x, self.x_tensor[candidate_idx].reshape(1,-1)], dim=0)
            self.init_y = torch.cat([self.init_y, self.y_tensor[candidate_idx].reshape(1,-1)])
            self.init_c = torch.cat([self.init_c, self.c_tensor[candidate_idx].reshape(1,-1)])
            self.observed[candidate_idx] = 1

            # retrain
            if i % retrain_interval != 0 and self.low_dim: # allow skipping retrain in low-dim setting
                self._c_state_dict_record = self.c_model.feature_extractor.state_dict()
                self._c_output_scale_record = self.c_model.model.covar_module.base_kernel.outputscale
                self._c_length_scale_record = self.c_model.model.covar_module.base_kernel.base_kernel.lengthscale

            self.c_model = DKL(self.init_x, self.init_c.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint, interpolate_prior = self.interpolate,)
               
            if i % retrain_interval != 0 and self.low_dim:
                self.c_model.feature_extractor.load_state_dict(self._c_state_dict_record, strict=False)
                self.c_model.model.covar_module.base_kernel.outputscale = self._c_output_scale_record
                self.c_model.model.covar_module.base_kernel.base_kernel.lengthscale = self._c_length_scale_record
            else:
                self.train()

            # regret & early stop
            feasible_obs_filter = self.init_c > self.c_threshold
            if sum(feasible_obs_filter) > 0:
                self.regret[i] = self.maximum - torch.max(self.init_y[feasible_obs_filter])
            else:
                self.regret[i] = self.max_regret

            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i], "Internal_beta": beta})


class DK_BO_AE_C_M():
    """
    Initialize the network with auto-encoder for constrained setting, support list of constraints
    """
    def __init__(self, x_tensor, y_tensor, c_tensor_list, roi_filter, c_uci_filter_list, c_threshold_list,
                    n_init:int=10, lr=1e-6, train_iter:int=10, regularize=True, spectrum_norm=False,
                    dynamic_weight=False, verbose=False, max=None, robust_scaling=True, pretrained_nn=None, low_dim=True,
                    record_loss=False, retrain_nn=True, exact_gp=False, noise_constraint=None, output_scale_constraint=None,
                    standardize=True,  **kwargs):

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

        self.init_x = kwargs.get("init_x", self.x_tensor[:n_init])        
        self.init_y = kwargs.get("init_y", self.y_tensor[:n_init])
        self.init_c_list = kwargs.get("init_c_list", [c_tensor[:n_init] for c_tensor in self.c_tensor_list])
        for c_idx in range(self.c_num):
            assert self.init_x.size(0) == self.init_c_list[c_idx].size(0)
        if "init_x" in kwargs:
            self.init_x = torch.from_numpy(self.scaler.transform(self.init_x)).float()
        self.spectrum_norm = spectrum_norm
        self.exact = exact_gp # exact GP overide
        self.noise_constraint = noise_constraint
        self.observed = kwargs.get("observed", np.zeros(self.x_tensor.size(0)).astype("int"))
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
                            interpolate_prior = self.interpolate,)
        else:
            self.f_model = f_model
        
        if c_model_list is None:
            self.c_model_list = [DKL(self.init_x, self.init_c_list[c_idx].squeeze(), 
                                    n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim,
                                    spectrum_norm=spectrum_norm, exact_gp=exact_gp, pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn,
                                    noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint, 
                                    interpolate_prior = self.interpolate,) for c_idx in range(self.c_num)]
        else:
            self.c_model_list = c_model_list
        
        self.record_loss = record_loss

        if self.record_loss:
            assert not (pretrained_nn is None)
            self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), 
                                   n_iter=self.train_iter, low_dim=self.low_dim, 
                                   pretrained_nn=None, lr=self.lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                                   interpolate_prior = self.interpolate,)
            self.f_loss_record = {"DK-AE":[], "DK":[]}
        self.cuda = torch.cuda.is_available()

        self.train()
    
    def train(self,):
        if self.regularize:
            self.f_model.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
            for c_idx in range(self.c_num):
                self.c_model_list[c_idx].train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        else:
            
            if self.record_loss:
                self._f_pure_dkl.train_model(record_mae=True)
                self.f_model.train_model(record_mae=True)
                for c_idx in range(self.c_num):
                    self.c_model_list[c_idx].train_model(record_mae=True)
            else:
                self.f_model.train_model(verbose=False)
                for c_idx in range(self.c_num):
                    self.c_model_list[c_idx].train_model(verbose=False)

    def periodical_retrain(self, i, retrain_interval): 
        self._interpolate_prior()
        # retrain for reuse
        if i % retrain_interval != 0 and self.low_dim: # allow skipping retrain in low-dim setting
            self._f_state_dict_record = self.f_model.feature_extractor.state_dict()
            self._f_output_scale_record = self.f_model.model.covar_module.base_kernel.outputscale
            self._f_length_scale_record = self.f_model.model.covar_module.base_kernel.base_kernel.lengthscale
            self._c_state_dict_record_list = [self.c_model_list[c_idx].feature_extractor.state_dict() for c_idx in range(self.c_num)]
            self._c_output_scale_record_list = [self.c_model_list[c_idx].model.covar_module.base_kernel.outputscale for c_idx in range(self.c_num)]
            self._c_length_scale_record_list = [self.c_model_list[c_idx].model.covar_module.base_kernel.base_kernel.lengthscale for c_idx in range(self.c_num)]

        self.f_model = DKL(self.init_x, self.init_y.squeeze(),
                                n_iter=self.train_iter, lr= self.lr, 
                                low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                spectrum_norm=self.spectrum_norm, exact_gp=self.exact, 
                                noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint,
                                interpolate_prior = self.interpolate,)
        self.c_model_list = [DKL(self.init_x, self.init_c_list[c_idx].squeeze(),
                                n_iter=self.train_iter, lr= self.lr, 
                                low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                spectrum_norm=self.spectrum_norm, exact_gp=self.exact, 
                                noise_constraint=self.noise_constraint, output_scale_constraint=self.output_scale_constraint,
                                interpolate_prior = self.interpolate,) for c_idx in range(self.c_num)]
        
        if self.record_loss:
            self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), 
                                   n_iter=self.train_iter, 
                                   low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=self.spectrum_norm, 
                                   exact_gp=self.exact, interpolate_prior = self.interpolate,)
        
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

    def update_obs(self, candidate_idx):
        self.init_x = torch.cat([self.init_x, self.x_tensor[candidate_idx].reshape(1,-1)], dim=0)
        self.init_y = torch.cat([self.init_y, self.y_tensor[candidate_idx].reshape(1,-1)])
        for c_idx in range(self.c_num):
            self.init_c_list[c_idx] = torch.cat([self.init_c_list[c_idx], self.c_tensor_list[c_idx][candidate_idx].reshape(1,-1)])
            assert self.init_x.size(0) == self.init_c_list[c_idx].size(0)
        assert self.init_x.size(0) == self.init_y.size(0)
        self.n_init = self.init_x.size(0)
        self.observed[candidate_idx] = 1
        observed_num = self.observed.sum()

    def update_regret(self, idx):
        feasible_obs_filter = feasible_filter_gen(self.init_c_list, self.c_threshold_list)
        if sum(feasible_obs_filter) > 0:
            _max_reward = torch.max(self.init_y[feasible_obs_filter])
            self.regret[idx] = self.maximum - _max_reward
        else:
            self.regret[idx] = self.max_regret

    def query_f_c(self, n_iter:int=10, acq="ci", retrain_interval:int=1, **kwargs):
        '''
        First Stage: Query both f and c simultaneously
        '''
        assert self.init_x.size(0) == self.init_c_list[0].size(0)
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

        beta = kwargs.get("beta", 1)
        _delta = kwargs.get("delta", .2)

        real_beta = beta <= 0 # if using analytic beta
        _candidate_idx_list = np.zeros(n_iter)
        ### optimization loop
        for i in iterator:
            _candidate_idx_c_list = [None for i in range(self.c_num)]
            _acq = 'ci'
            # _acq = 'lcb' if i // 2 and acq == 'ci' else 'ci'
            if real_beta:
                beta = (2 * np.log((self.x_tensor.size(0) * (np.pi * (self.init_x.size(0) + 1)) ** 2) /(6 * _delta))) ** 0.5
            # acq values
            if ci_intersection:
                assert not( f_max_test_x_lcb is None or f_min_test_x_ucb is None)
                _candidate_idx_f = self.f_model.intersect_CI_next_point(self.x_tensor[self.roi_filter], 
                                                                        max_test_x_lcb=f_max_test_x_lcb[self.roi_filter], 
                                                                        min_test_x_ucb=f_min_test_x_ucb[self.roi_filter], 
                                                                        acq=_acq, beta=beta, return_idx=True)

                for c_idx, (c_uci_filter, c_max_test_x_lcb, c_min_test_x_ucb) in enumerate(zip(self.c_uci_filter_list, c_max_test_x_lcb_list, c_min_test_x_ucb_list)):
                    if sum(c_uci_filter) > 0:
                        _candidate_idx_c = self.c_model_list[c_idx].intersect_CI_next_point(self.x_tensor[c_uci_filter], 
                                                                max_test_x_lcb=c_max_test_x_lcb[c_uci_filter], 
                                                                min_test_x_ucb=c_min_test_x_ucb[c_uci_filter], 
                                                                acq=_acq, beta=beta, return_idx=True)
                        _candidate_idx_c_list[c_idx] = _candidate_idx_c
            else:
                _candidate_idx_f = self.f_model.next_point(self.x_tensor[self.roi_filter], acq, "love", return_idx=True, beta=beta,)
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
                        _c_acq_list[c_idx] = self.c_model_list[c_idx].acq_val[_candidate_idx_c_list[c_idx]]
                _c_acq_list_max = torch.max(_c_acq_list, dim=0)

                if _c_acq_list_max.values > _f_acq:
                    assert _c_acq_list_max.values > float("-inf")
                    _c_idx = _c_acq_list_max.indices
                    candidate_idx = util_array[self.c_uci_filter_list[_c_idx]][_candidate_idx_c_list[_c_idx]]
                else:
                    candidate_idx = util_array[self.roi_filter][_candidate_idx_f]

            # update obs
            _candidate_idx_list[i] = candidate_idx
            self.update_obs(candidate_idx)

            # Retrain
            self.periodical_retrain(i, retrain_interval)

            # regret & early stop
            self.update_regret(idx=i)

            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i], "Internal_beta": beta})

    def query_f_passive_c(self, n_iter:int=10, acq='qei', retrain_interval:int=1, **kwargs):
        '''
        First Stage: Query both f and c simultaneously
        '''
        self.regret = np.zeros(n_iter)
        if_tqdm = kwargs.get("if_tqdm", False)
        early_stop = kwargs.get("early_stop", True)
        iterator = tqdm.tqdm(range(n_iter)) if if_tqdm else range(n_iter)

        _candidate_idx_list = np.zeros(n_iter)
        ### optimization loop
        for i in iterator:
            # first generate acq_f
            _acq = acq.lower()
            if _acq in ['qei', 'ts', 'cmes-ibo']: # qei should be different because f_max is not purely on f
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
                    with gpytorch.settings.fast_pred_samples():
                        # start_time = time.time()
                        if _acq in ['ts']:
                            _num_sample = 1
                        elif _acq in ['qei']:
                            _num_sample = 100
                        else:
                            _num_sample = kwargs.get('num_sample', 5)
                            # _num_sample = kwargs.get('num_sample', 20)
                            # _num_sample = kwargs.get('num_sample', 2)
                        self.f_model.model.eval()
                        _posterior = self.f_model.model(self.x_tensor)
                        _samples = _posterior.rsample(torch.Size([_num_sample]))
                    feasible_obs_filter = feasible_filter_gen(self.init_c_list, self.c_threshold_list)
                    if sum(feasible_obs_filter) > 0:
                        _best_y = torch.max(self.init_y[feasible_obs_filter])
                    else:
                        _best_y = self.y_tensor.min()

                    if _acq in ['ts']:
                        _acq_f = _samples.reshape([-1, self.data_size])
                    elif _acq in ['qei']:
                        _acq_f = (_samples.T - _best_y).clamp(min=0).mean(dim=-1)
                    elif _acq in ['cmes-ibo']:
                        # _subsample_num = kwargs.get("subsample_num", 1000)
                        _subsample_num = kwargs.get("subsample_num", self.data_size)
                        _subsample_num = min(_subsample_num, 10000)
                        # _subsample_num = min(_subsample_num, 4000)
                        subsample_filter = np.random.choice(self.data_size, _subsample_num, replace=False)
                        # subsample_filter = np.arange(self.data_size)
                        # sample c
                        _c_tensor_list = []
                        with gpytorch.settings.fast_pred_samples():
                            for _c_idx, _dk in enumerate(self.c_model_list):
                                _model = _dk.model
                                _model.eval()
                                _c_sample = _model(self.x_tensor[subsample_filter]).rsample(torch.Size([1])).reshape([-1,1])
                                _c_tensor_list.append(_c_sample)
                        _feasible_filter = feasible_filter_gen(_c_tensor_list, self.c_threshold_list)
                        if _feasible_filter.sum() == 0:
                            _feasible_filter = feasible_filter_gen(self.c_tensor_list, self.c_threshold_list)[subsample_filter]
                        _max_f_samples = _samples[:,subsample_filter][:,_feasible_filter].max(dim=-1).values.squeeze()
                        # sample f
                        self.f_model.model.eval()
                        _mvn = self.f_model.model(self.x_tensor[subsample_filter])
                        _acq_f = torch.cat([self.f_model.mvn_survival(_mvn, threshold).reshape([1, _subsample_num]) for threshold in _max_f_samples], dim=0)
                        assert _acq_f.size(0) == _num_sample
                        assert _acq_f.size(1) == subsample_filter.shape[0]

            elif _acq == 'random':
                _acq_f = torch.rand(self.data_size).unsqueeze(0)
            else:
                _  = self.f_model.next_point(self.x_tensor, acq, "love", return_idx=True)
                _acq_f = self.f_model.acq_val.reshape([1, -1])

            # then generate _c_prob
            if (_acq in ['cmes-ibo']):
                _c_prob = torch.cat([self.c_model_list[c_idx].marginal_survival(self.x_tensor[subsample_filter], self.c_threshold_list[c_idx]).unsqueeze(0) for c_idx in range(self.c_num)], dim=0)

            else:
                _c_prob = torch.cat([self.c_model_list[c_idx].marginal_survival(self.x_tensor, self.c_threshold_list[c_idx]).unsqueeze(0) for c_idx in range(self.c_num)], dim=0)
            
            _acq_value = torch.mul(torch.prod(_c_prob, dim=0), _acq_f)
            if (_acq in ['cmes-ibo']):
                _acq_value = 1 - _acq_value
                _acq_value = torch.where(_acq_value > 1e-2, _acq_value, 1e-2) # guarantee no numerical problem
                _acq_value = - torch.log(_acq_value)
                _acq_value = _acq_value.mean(dim=0)
            else:
                assert _acq_value.size(-1) == self.data_size
            
            candidate_idx = torch.argmax(_acq_value)

            if acq in ['cmes-ibo']:
                candidate_idx = subsample_filter[candidate_idx]

            # update obs
            _candidate_idx_list[i] = candidate_idx
            self.update_obs(candidate_idx)

            # Retrain
            self.periodical_retrain(i, retrain_interval)

            # regret & early stop
            self.update_regret(idx=i)

            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i]})
