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

class DK_BO_AE():
    """
    Initialize the network with auto-encoder
    """
    def __init__(self, train_x, train_y, n_init:int=10, lr=1e-6, train_iter:int=10, regularize=True, spectrum_norm=False,
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
        self.train_x = torch.from_numpy(train_x).float()
        self.train_y = train_y
        self.train_iter = train_iter
        self.retrain_nn = retrain_nn
        self.maximum = torch.max(self.train_y) if max==None else max
        self.init_x = kwargs.get("init_x", self.train_x[:n_init])
        self.init_y = kwargs.get("init_y", self.train_y[:n_init])
        self.spectrum_norm = spectrum_norm
        self.exact = exact_gp # exact GP overide
        self.noise_constraint = noise_constraint
        # print(f"init_y_max {self.init_y.max()}")
        self.observed = np.zeros(self.train_x.size(0)).astype("int")
        # self.observed[:n_init] = True
        self.pretrained_nn = pretrained_nn
        self.dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, 
                        pretrained_nn=self.pretrained_nn, retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, exact_gp=exact_gp, noise_constraint = self.noise_constraint)

        # print("ROI GP: ", self.init_y.size(), f"n_iter={train_iter}, low_dim={low_dim}, retrain_nn={self.dkl.retrain_nn} lr={lr}{self.dkl.lr}, spectrum_norm={spectrum_norm} regularize {regularize}")
        self.record_loss = record_loss

        if self.record_loss:
            assert not (pretrained_nn is None)
            self._pure_dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=spectrum_norm, exact_gp=exact_gp)
            self.loss_record = {"DK-AE":[], "DK":[]}
        self.cuda = torch.cuda.is_available()

        self.train()
    
    def train(self):
        if self.regularize:
            self.dkl.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        else:
            
            if self.record_loss:
                self._pure_dkl.train_model(record_mae=True)
                self.dkl.train_model(record_mae=True)
            else:
                # print(self.verbose)
                # self.dkl.train_model(verbose=self.verbose)
                self.dkl.train_model(verbose=False)

        

    def query(self, n_iter:int=10, acq="ts", study_ucb=False, retrain_interval:int=1, **kwargs):
        self.regret = np.zeros(n_iter)
        self.interval = np.zeros(n_iter)
        if_tqdm = kwargs.get("if_tqdm", False)
        early_stop = kwargs.get("early_stop", True)
        iterator = tqdm.tqdm(range(n_iter)) if if_tqdm else range(n_iter)
        study_interval = kwargs.get("study_interval", 10)
        _path = kwargs.get("study_res_path", None)
        ci_intersection = kwargs.get("ci_intersection", False)
        max_test_x_lcb = kwargs.get("max_test_x_lcb", None)
        min_test_x_ucb = kwargs.get("min_test_x_ucb", None)
        beta = kwargs.get("beta", 1)
        _delta = kwargs.get("delta", .2)

        real_beta = beta <= 0 # if using analytic beta
        # print(beta, acq)
        # _candidate_idx_list = np.hstack([np.arange(self.n_init), np.zeros(n_iter)])
        _candidate_idx_list = np.zeros(n_iter)
        for i in iterator:
            
            if real_beta:
                beta = (2 * np.log((self.train_x.shape[0] * (np.pi * (self.init_x.shape[0] + 1)) ** 2) /(6 * _delta))) ** 0.5
            if ci_intersection:
                assert not( max_test_x_lcb is None or min_test_x_ucb is None)
                # assert acq.lower() in ['ci','ucb','lcb']
                candidate_idx = self.dkl.intersect_CI_next_point(self.train_x, 
                    max_test_x_lcb=max_test_x_lcb, min_test_x_ucb=min_test_x_ucb, acq=acq, beta=beta, return_idx=True)
            else:
                candidate_idx = self.dkl.next_point(self.train_x, acq, "love", return_idx=True, beta=beta,)
            _candidate_idx_list[i] = candidate_idx
            # print(self.init_x.size(),  self.train_x[candidate_idx].size())
            self.init_x = torch.cat([self.init_x, self.train_x[candidate_idx].reshape(1,-1)], dim=0)
            self.init_y = torch.cat([self.init_y, self.train_y[candidate_idx].reshape(1,-1)])
            self.observed[candidate_idx] = 1


            if study_ucb and i % study_interval ==0:
                fig = plt.figure()
                # print(self.train_y.size(), self.dkl.acq_val.size(), self.init_y.squeeze().size(), self.dkl.acq_val[_candidate_idx_list[:i + self.n_init + 1]].size())
                plt.scatter(self.train_y.squeeze(), self.dkl.acq_val, s=2)
                plt.scatter(self.init_y.squeeze(), self.dkl.acq_val[_candidate_idx_list[:i + self.n_init + 1]], color='r', marker="*", s=5)
                plt.xlabel("Label")
                plt.ylabel(acq)
                plt.colorbar()
                plt.title(f"Iter {i}")
                plt.savefig(f"{_path}/pure_dkbo_Scaling{self.dkl.model.covar_module.base_kernel.outputscale}_Iter{i}.png")

            # retrain
            if i % retrain_interval != 0 and self.low_dim: # allow skipping retrain in low-dim setting
                self._state_dict_record = self.dkl.feature_extractor.state_dict()
                self._output_scale_record = self.dkl.model.covar_module.base_kernel.outputscale
                self._length_scale_record = self.dkl.model.covar_module.base_kernel.base_kernel.lengthscale
            self.dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, lr= self.lr, low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, retrain_nn=self.retrain_nn,
                                 spectrum_norm=self.spectrum_norm, exact_gp=self.exact, noise_constraint=self.noise_constraint)
            if self.record_loss:
                self._pure_dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_iter, low_dim=self.low_dim, pretrained_nn=None, lr=self.lr, spectrum_norm=self.spectrum_norm, exact_gp=self.exact)
            # self.dkl.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False)
            if i % retrain_interval != 0 and self.low_dim:
                self.dkl.feature_extractor.load_state_dict(self._state_dict_record, strict=False)
                self.dkl.model.covar_module.base_kernel.outputscale = self._output_scale_record
                self.dkl.model.covar_module.base_kernel.base_kernel.lengthscale = self._length_scale_record
            else:
                self.train()
            if self.record_loss:
                self.loss_record["DK-AE"].append(self.dkl.mae_record[-1])
                self.loss_record["DK"].append(self._pure_dkl.mae_record[-1])

            # regret & interval
            self.regret[i] = self.maximum - torch.max(self.init_y)
            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i], "Internal_beta": beta})

            _lcb, _ucb = self.dkl.CI(self.train_x)
            self.interval[i] = (_ucb.max() -_lcb.max()).numpy()
        # print(f"observed range in DKBO-AE {self.init_y.min()} {self.init_y.max()} max val {self.maximum}")
        # print(f"observed range in DKBO-AE {self.train_y.min()} {self.train_y.max()} max val {self.maximum}")
        # print(f"observed range in DKBO-AE {self.train_y[self.observed].min()} {self.train_y[self.observed].max()} max val {self.maximum}")
        # print(f"observed range in DKBO-AE {self.train_y[self.observed==1].min()} {self.train_y[self.observed==1].max()} max val {self.maximum}")


