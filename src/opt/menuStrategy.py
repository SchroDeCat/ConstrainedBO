
'''
Implementation for experiments on LLNL simulation env.
'''

import pandas as pd
import numpy as np
from sklearn import cluster
import torch
import sys
import os
import dill as pickle

import gpytorch
import random
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
import copy


from ..models import DKL, AE, beta_CI
from .baseStrategy import MenuStrategy
from ..models import AE, beta_CI, SGLD
from ..utils import save_res, load_res, clustering_methods
from .dkbo_olp import DK_BO_OLP, DK_BO_OLP_Batch
from .dkbo_ae import DK_BO_AE
from abc import ABCMeta, abstractmethod
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.turbo import TuRBO
# from abag_model_development.models.active_learning.base_strategy import BaseStrategy, MenuStrategy

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')


class DKBO_OLP(MenuStrategy):
# class DKBO_OLP():
    '''
    DKBO-Online Partition inherting Menu Strategy

    Attributes
    ----------
    name: str
        Name of active learning algo for record keeping.
    model: `DKL_BO_OLP_Batch`
        underlying model.
    train: bool
        Boolean whether we should train underlying model on some data before running selection.
    iters: int
        number of iterations to train underlying model for during intialization. Only used if train is `self.train = True`.
    partition_strategy:
        strategy of the online partition. (kmeans-y, kmeans, linear_rc, gp_rc).
    n_init: int
        number of points to initialize GP.
    num_GP: int
        number of partitions/ local GPs.
    train_times: int
        training iteration for each GP.
    acq: str
        acquisition function (UCB, TS).
    verbose: bool
        if printing verbose information.
    lr: float
        learning rate of the model.
    ucb_strategy: str
        strategy of coordinating UCB from multiple local GPs (exact, max, sum).
    pretrained: bool
        if using pretrained model to initialize the feature extracter of the DKL.
    ae_loc:
        location of the pretrained Auto Encoder.
    exact_gp:
        if using exact_gp or Deep Kernel
    '''

    def __init__(self, x_tensor:torch.tensor, y_tensor:torch.tensor, partition_strategy:str="ballet", num_GP:int=3, low_dim:bool=True,
                    train_times:int=10, acq:str="ts", verbose:bool=True, lr:float=1e-2, name:str="test", ucb_strategy:str="exact",
                    train:bool=True, pretrained:bool=False, ae_loc:str=None, abag_transform=False, exact_gp:bool=False):
        '''
        Args:
            @x_tensor: init x
            @y_tensor: init y
            @partition_strategy: strategy for the partition.
            @num_GP: desired number of local GPs.
            @train_times: number of training iteration for the models.
            @acq: acquisition function choice.
            @verbose: if printing verbose info.
            @lr: learning rate of the model.
            @name: instance name.
            @ucb_strategy: strategy to coordinate multiple local UCBs.
            @train: if updating the model.
            @pretrained: if using pretrained model to initilize the NN's weight.
            @ae_loc: location of the pretrained Auto Encoder.
            @abag_transform: boolean that indicates if we need to transform to abag closed loop

        '''
        use_gpu = (
            True if torch.cuda.is_available() or torch.backends.mps.is_available() else False
        )
        use_gpu = False # for current version
        if use_gpu:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        else:
            self.device = 'cpu'
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.partition_strategy = partition_strategy
        self.num_GP = num_GP
        self.acq = acq
        self.verbose = verbose
        self.lr = lr
        self.ucb_strategy = ucb_strategy
        self.pretrained = pretrained
        self.ae_loc = ae_loc
        self.low_dim = low_dim
        self.exact_gp = exact_gp
        # super().__init__(name=name, model=None, train=train, iters=train_times)
        self.iters = train_times
        self.train = train
        self.train_times = self.iters
        self.if_train = self.train
        self.abag_transform = abag_transform
        if self.abag_transform:
            x_tensor,y_tensor = self.transform_xy(x_tensor,y_tensor)
        
        # load pretrained model
        if self.pretrained:
            assert not (ae_loc is None)
            self.ae = AE(x_tensor, lr=1e-3)
            self.ae.load_state_dict(torch.load(ae_loc, map_location=self.device))
        else:
            self.ae = None

       
        # initialize the model and partitioning
        self.init_x = x_tensor
        self.init_y = y_tensor
        self.n_init = self.init_x.size(0)
        self._dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_times, low_dim=self.low_dim,  pretrained_nn=self.ae, exact_gp=self.exact_gp)
        self._dkl.train_model(verbose=verbose)
        self._dkl.model.eval()
        self._turbo = TuRBO(self.init_x, self.init_y.squeeze(), train_iter=self.train_times, low_dim=self.low_dim,  pretrained_nn=self.ae, acqf=acq, verbose=verbose)

    def fit(self,X,y,**kwargs):
        print("DKL OLP has no fit")
        pass

    def transform_xy(self,x,y, cat=False):
        y = - copy.deepcopy(y)
        if cat:
            x = torch.cat([x,y.unsqueeze(-1)],axis=1)
        x = torch.from_numpy(RobustScaler().fit_transform(copy.deepcopy(x))).float()
        return x,y
    
    def pretrain_ae(self, init_x=None, ae_path=None):
        if ae_path:
            self.ae_loc = ae_path
        ae = AE(init_x, lr=1e-3)
        ae.train_ae(epochs=100, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), self.ae_loc)
        print(f"pretrained ae stored in {self.ae_loc}")

    def select(self, X:torch.tensor, i:int, k:int, **kwargs):
        """
        Select k points from the N*D search space X of the fidelity i.

        Args:
            @X: N * D matrix to select from.
            @i: fidelity number.
            @k: number of points to be picked from X

        Return:
            :return: list of indices that were selected. 
            :rtype: list
        """
        if self.abag_transform:
            X,_ = self.transform_xy(X,X, cat=False) # y doesn't matter
        x_tensor = torch.cat([self.init_x, X]) # could result in duplication
        _data_size = x_tensor.size(0)
        _util_array = np.arange(_data_size)
        self.observed = np.zeros(_data_size)
        self.observed[:self.n_init] = 1
        # partition

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _observed_pred = self._dkl.likelihood(self._dkl.model(x_tensor.to(self.device)))
            _, ucb = _observed_pred.confidence_region()
        if self.partition_strategy in ["kmeans-y"]:
            self.cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), self.num_GP, self.partition_strategy, 
                                        dkl=True, pretrained_nn=self.ae, train_iter=self.train_times)
            self.cluster_id_init = self.cluster_id[self.observed==1]

            init_x_list, init_y_list, test_x_list = [], [], []
            self.cluster_filter_list, self.cluster_idx_list = [], []
            for idx in range(self.num_GP):
                cluster_filter = self.cluster_id == idx
                cluster_filter[:self.n_init] = False # avoid querying the initial pts since it is not necessarily in X
                if np.sum(cluster_filter) == 0:   # avoid empty class -- regress to single partition.
                    cluster_filter[self.n_init:] = True 
                self.cluster_filter_list.append(cluster_filter)
                self.cluster_idx_list.append(_util_array[cluster_filter])
                test_x_list.append(x_tensor[cluster_filter])
                init_x_list.append(self.init_x)
                init_y_list.append(self.init_y.reshape([-1,1]))

        elif self.partition_strategy.lower() in ["ballet"]:
            # Identify ROI as in BALLET
            self.num_GP = 2 # override input parameter
            lcb, ucb = self._dkl.CI(x_tensor.to(self.device))
            _delta = kwargs.get("delta", 0.2)
            _filter_beta = kwargs.get("filter_beta", 0.2)
            _beta = (2 * np.log((x_tensor.shape[0] * (np.pi * (self.n_init + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
            _filter_lcb, _filter_ucb = beta_CI(lcb, ucb, _filter_beta)
            ucb_filter = _filter_ucb >= _filter_lcb.max()
            _minimum_pick = min(kwargs.get("minimum_pick", 4), self.n_init)
            if sum(ucb_filter[self.observed==1]) <= _minimum_pick:
                _, indices = torch.topk(ucb[self.observed==1], min(_minimum_pick, _data_size))
                for idx in indices:
                    ucb_filter[_util_array[[self.observed==1]][idx]] = 1
            observed_unfiltered = np.min([self.observed, ucb_filter.numpy()], axis=0)      # observed and not filtered outs
            # two parition, one global, one ROI
            self.cluster_filter_list = [np.ones(_data_size), ucb_filter]
            self.cluster_idx_list = [_util_array, _util_array[ucb_filter]]
            test_x_list = [x_tensor, x_tensor[ucb_filter==1]]
            init_x_list = [self.init_x, x_tensor[observed_unfiltered==1]]
            init_y_list = [self.init_y.reshape([-1,1]), self.init_y.reshape([-1,1])[observed_unfiltered[:self.n_init]==1]]
        
        elif self.partition_strategy.lower() in ["turbo"]:
            selected_idx_list = self._turbo.opt(max_iter=k, test_x = x_tensor, retrain_interval=5)
            self.cluster_filter_list = [self._turbo.tr_filter]
            # early return
            return selected_idx_list, _observed_pred.mean[self.n_init:], _observed_pred.stddev[self.n_init:], _observed_pred.covariance_matrix[self.n_init:, self.n_init:]
        else:
            raise NotImplementedError(f"{self.partition_strategy} Unknown")

        self.model = DK_BO_OLP_Batch(init_x_list, init_y_list, lr=self.lr, train_iter=self.train_times,
                        n_init=self.n_init, num_GP=self.num_GP, pretrained_nn=self.ae)

        # batched query
        acq = kwargs.get("acq", "ts")
        verbose = kwargs.get("verbose", False)
        candidate_indices = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indices = self.model.query(test_x_list=test_x_list, n_iter=k, acq=acq, verbose=verbose) # list of (model_idx, candidate_idx)
        
        # recover indices in original input X
        # Question: what if the initial pts are queried? --> avoid picking these n_init pts
        selected_idx_list = []
        for (_model_idx, _candidate_idx) in indices:
            _original_idx = self.cluster_idx_list[_model_idx][_candidate_idx] - self.n_init
            assert _candidate_idx >= 0 and _candidate_idx < _data_size
            selected_idx_list.append(_original_idx)
        return selected_idx_list, _observed_pred.mean[self.n_init:], _observed_pred.stddev[self.n_init:], _observed_pred.covariance_matrix[self.n_init:, self.n_init:]

    def visualize(self,X_tensor,y_tensor, save_path='./'):
        colors = np.array(['r','g','b'])
        labels = np.array(['gp1','gp2','gp3'])
        fig = plt.figure()
        _path = f"{save_path}"
        # should use likliehood from the global GP not the local GP
        #  use alll X,y pairs
        # partition
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _observed_pred = self._dkl.likelihood(self._dkl.model(X_tensor.to(self.device)))
            _, ucb = _observed_pred.confidence_region()
        self.cluster_id = clustering_methods(X_tensor, ucb.to('cpu'), self.num_GP, self.partition_strategy, 
                                        dkl=True, pretrained_nn=self.ae, train_iter=self.train_times)
        # for gp_idx, (X_tensor,y_tensor,gp_model) in enumerate(zip(self.model.init_x_list, self.model.init_y_list, self.model.dkl_list)):
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         observed_pred = gp_model.likelihood(gp_model.model(X_tensor))
        #         _, ucb = observed_pred.confidence_region()
            # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
            # plt.scatter(y_tensor, ucb, s=2)
        # for gp_idx, (X_tensor,y_tensor,gp_model) in enumerate(zip(self.model.init_x_list, self.model.init_y_list, self.model.dkl_list)):
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         observed_pred = gp_model.likelihood(gp_model.model(X_tensor))
        #         _, ucb = observed_pred.confidence_region()
        #     # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
        #     # plt.scatter(y_tensor, ucb, s=2)
        fig  = plt.figure()
        for cluster_idx in self.cluster_id:
            inds = np.where(self.cluster_id  == cluster_idx)[0]
            plt.scatter(y_tensor[inds], ucb[inds], color=colors[cluster_idx], marker="*", s=5) # label=labels[cluster_idx]
        plt.xlabel("ddG")
        plt.ylabel("UCB")
        # plt.colorbar()
        plt.legend()
        plt.title(f"Add iteration name here")
        print('starting save')
        plt.savefig(f"{_path}.png", bbox_inches='tight', dpi=100)
        print(f"Fig stored to {_path}")
        plt.close(fig)

    def update(self,X,y,**kwargs):
        """
        Update internal model at `X` seqs with `y` observations
        """
        if self.abag_transform:
            X,y = self.transform_xy(X,y)
        # update the partitions
        self.init_x = torch.cat([self.init_x, X])
        self.init_y = torch.cat([self.init_y, y])
        self.n_init = self.init_x.size(0)
        self._dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_times, low_dim=self.low_dim,  pretrained_nn=self.ae, exact_gp=self.exact_gp)
        self._turbo = TuRBO(self.init_x, self.init_y.squeeze(), train_iter=self.train_times, low_dim=self.low_dim,  pretrained_nn=self.ae, acqf=self.acq, verbose=self.verbose)
        # settings to print NLL
        self._dkl.train_model(record_mae=True, record_interval=1, verbose=self.verbose)
        self._dkl.model.eval()

    def select_from_menu_df(self, menu_df, target_df, predictor_cols, type_col,
        target_ab_ids=None, lincoeffs_target_means_by_ab=None, cost_dict=None,**kwargs):
        
        # 1. Combine menu df with target_df .. is this necessary? or do we just assume that 
        # we have that? need inputs for evaluate function
        start_rows, end_rows, rows, target_rows, pam30_penalty_values  = self.preprocess_menu_rows(menu_df, target_df, target_ab_ids=target_ab_ids)

        # 2 compute / or recieve pam30 or other annotations to the sequence
        pred_x = torch.tensor(
        pd.concat(rows, axis=0, sort=False)[
            predictor_cols].values,
        dtype=torch.float
        )
        pred_i = torch.tensor(
            pd.concat(rows, axis=0, sort=False)[
                type_col].values.astype(np.double),
            dtype=torch.long
        )
        # TODO: DEFINE COST HERE?
        # TODO: k = 1 always since batching isn't done here
        # min_cost_rows = np.repeat(1, len(joint_mvn_mean))
        sel_idx, pred_mean, pred_std, covar = self.select(pred_x, pred_i, 1, **kwargs)
        del covar # we don't need this. though saving it could be a good sanity check
        # 4. add penalties # TODO: CANNOT ADD PEN
        # for i, p30i in enumerate(pam30_penalty_values):
        #     if not np.isnan(p30i):
        #         scores[i] = scores[i] + p30i
        fake_scores = np.ones(shape=len(pred_mean))
        fake_scores[sel_idx] = -100 # NOTE: subtract initialization points fromm selected index
        return fake_scores, pred_mean, pred_std
        

