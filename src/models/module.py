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

# from src.utils import beta_CI
from .exact_gp import ExactGPRegressionModel
from .sgld import SGLD
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

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, low_dim, add_spectrum_norm):
        
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', add_spectrum_norm(torch.nn.Linear(data_dim, 1000)))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2',  add_spectrum_norm(torch.nn.Linear(1000, 500)))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3',  add_spectrum_norm(torch.nn.Linear(500, 50)))
        # test if using higher dimensions could be better
        if low_dim:
            self.add_module('relu3', torch.nn.ReLU())
            self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(50, 1)))
        else:
            self.add_module('relu3', torch.nn.ReLU())
            self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(50, 10)))

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, gp_likelihood, gp_feature_extractor, low_dim=True):
            super(GPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
            self.feature_extractor = gp_feature_extractor
            try: # gpytorch 1.6.0 support
                self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())
            except Exception: # gpytorch 1.9.1
                self.mean_module = gpytorch.means.ConstantMean()
            if low_dim:
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1), 
                    outputscale_constraint=gpytorch.constraints.Interval(0.7,1.0)),
                    num_dims=1, grid_size=100)
            else:
                self.covar_module = gpytorch.kernels.LinearKernel(num_dims=10)

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            self.projected_x = self.feature_extractor(x)
            self.projected_x = self.scale_to_bounds(self.projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(self.projected_x)
            covar_x = self.covar_module(self.projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)