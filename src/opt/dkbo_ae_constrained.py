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

class DK_BO_AE_C():
    """
    Initialize the network with auto-encoder for constrained setting
    """
    def __init__(self, x_tensor, y_tensor, c_tensor, roi_filter, c_uci_filter, optimization_ratio, c_threshold,
                    n_init:int=10, lr=1e-6, train_iter:int=10, regularize=True, spectrum_norm=False,
                    dynamic_weight=False, verbose=False, max=None, robust_scaling=True, pretrained_nn=None, low_dim=True,
                    record_loss=False, retrain_nn=True, exact_gp=False, noise_constraint=None, **kwargs):
        # scale input
        ScalerClass = RobustScaler if robust_scaling else StandardScaler
        self.scaler = ScalerClass().fit(train_x)
        train_x = self.scaler.transform(train_x)
        # init vars
        self.regularize = regularize
        self.lr = lr
        self.low_dim = low_dim
        self.verbose = verbose
        self.n_init = n_init
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.dynamic_weight = dynamic_weight
        self.x_tensor = x_tensor.float()
        self.y_tensor = y_tensor.float()
        self.c_tensor = c_tensor.float()
        self.data_size = x_tensor.size(0)
        self.train_iter = train_iter
        self.retrain_nn = retrain_nn
        self.maximum = torch.max(self.y_tensor) if max==None else max
        self.max_regret = self.maximum - torch.min(self.y_tensor)
        self.init_x = kwargs.get("init_x", self.x_tensor[:n_init])
        self.init_y = kwargs.get("init_y", self.y_tensor[:n_init])
        self.init_c = kwargs.get("init_c", self.c_tensor[:n_init])
        self.spectrum_norm = spectrum_norm
        self.exact = exact_gp # exact GP overide
        self.noise_constraint = noise_constraint
        self.observed = np.zeros(self.x_tensor.size(0)).astype("int")
        self.pretrained_nn = pretrained_nn
        self.roi_filter = roi_filter
        self.c_uci_filter = c_uci_filter
        self.c_threshold = c_threshold
        f_model = kwargs.get("f_model", None)
        c_model = kwargs.get("c_model", None)

        if f_model is None:
            self.f_model = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, 
                            pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                            noise_constraint = self.noise_constraint)
        else:
            self.f_model = f_model
        
        if c_model is None:
            self.c_model = DKL(self.init_x, self.init_c.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, 
                            pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, exact_gp=exact_gp, 
                            noise_constraint = self.noise_constraint)
        else:
            self.c_model = c_model
        
        self.record_loss = record_loss

        if self.record_loss:
            assert not (pretrained_nn is None)
            self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp)
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

        

    def query_f_c(self, n_iter:int=10, acq="ts", retrain_interval:int=1, **kwargs):
        '''
        First Stage: Query both f and c simultaneously
        '''
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
                _candidate_idx_c = self.c_model.intersect_CI_next_point(self.x_tensor[self.c_uci_filter], 
                                                        max_test_x_lcb=c_max_test_x_lcb[self.c_uci_filter], 
                                                        min_test_x_ucb=c_min_test_x_ucb[self.c_uci_filter], 
                                                        acq=acq, beta=beta, return_idx=True)
            else:
                _candidate_idx_f = self.f_model.next_point(self.x_tensor[self.roi_filter], acq, "love", return_idx=True, beta=beta,)
                _candidate_idx_c = self.c_model.next_point(self.x_tensor[self.c_uci_filter], acq, "love", return_idx=True, beta=beta,)
            _f_acq = self.f_model.acq_val[_candidate_idx_f]
            _c_acq = self.c_model.acq_val[_candidate_idx_c]
        
            candidate_idx = util_array[self.roi_filter][_candidate_idx_f] if _f_acq >= _c_acq else util_array[self.c_uci_filter][_candidate_idx_c]

            _candidate_idx_list[i] = candidate_idx
            self.init_x = torch.cat([self.init_x, self.x_tensor[candidate_idx].reshape(1,-1)], dim=0)
            self.init_y = torch.cat([self.init_y, self.y_tensor[candidate_idx].reshape(1,-1)])
            self.init_c = torch.cat([self.init_c, self.c_tensor[candidate_idx].reshape(1,-1)])
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
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint)
            self.c_model = DKL(self.init_x, self.init_c.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint)
            
            if self.record_loss:
                self._f_pure_dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=self.spectrum_norm, exact_gp=self.exact)
            
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
                self.regret[i] = self.maximum - torch.max(self.init_y[feasible_obs_filter])
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
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint)
               
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