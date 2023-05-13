#!/usr/bin/env python3
# coding: utf-8

# # Scalable Constrained Bayesian Optimization (SCBO)
# In this tutorial, we show how to implement Scalable Constrained Bayesian Optimization (SCBO) [1] in a closed loop in BoTorch.
# 
# We optimize the 20𝐷 Ackley function on the domain $[−5,10]^{20}$. This implementation uses two simple constraint functions $c1$ and $c2$. Our goal is to find values $x$ which maximizes $Ackley(x)$ subject to the constraints $c1(x) \leq 0$ and $c2(x) \leq 0$.
# 
# [1]: David Eriksson and Matthias Poloczek. Scalable constrained Bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 730–738. PMLR, 2021.
# (https://doi.org/10.48550/arxiv.2002.08526)
# 
# Since SCBO is essentially a constrained version of Trust Region Bayesian Optimization (TuRBO), this tutorial shares much of the same code as the TuRBO Tutorial (https://botorch.org/tutorials/turbo_1) with small modifications made to implement SCBO.


import math
import numpy as np
import os
import warnings
from tqdm import tqdm
from dataclasses import dataclass

import sys
sys.path.append(f"{os.path.dirname(__file__)}/..")
from models import DKL

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
# from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from botorch.models.transforms.outcome import Standardize

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2,) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

class SCBO:
    def __init__(self, obj_func, c_func_list, dim:int, lower_bound, upper_bound, 
                 batch_size:int=1, n_init:int=10, verbose=True, dk=False, constrain_noise=True,
                **kwargs):
        self.max_cholesky_size = float("inf") 
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.batch_size = batch_size
        self.n_init = n_init
        self.func = obj_func
        self.c_func_list = c_func_list
        self.verbose = verbose
        self.dk = dk
        if self.dk:
            self.train_times = kwargs.get("train_times", 50)
            self.learning_rate = kwargs.get("lr", 1e-4)
        self.constrain_noise = constrain_noise
        self.state = ScboState(dim=dim, batch_size=batch_size)

        # get initial data either from input & sampling
        self.discrete = 'train_X' in kwargs and 'x_space' in kwargs
        self.n_constraint = len(c_func_list)
        self.train_X = kwargs.get("train_X", self.get_initial_points(dim, n_init))
        self.train_Y = kwargs.get("train_Y", torch.tensor(
                            [self.eval_objective(x) for x in self.train_X], dtype=dtype, device=device
                            ).unsqueeze(-1))
        self.train_C_list = [None for _ in range(self.n_constraint)]
        for c_idx in range(self.n_constraint):
            self.train_C_list[c_idx] = torch.tensor([self.eval_constraint(x, c_idx) for x in self.train_X], 
                                                    dtype=dtype, device=device).unsqueeze(-1)


    def eval_objective(self, x,):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return self.func(unnormalize(x, (self.lower_bound, self.upper_bound)))
    
    def eval_constraint(self, x, idx):
        """This is a helper function we use to unnormalize and evalaute a point"""
        c = self.c_func_list[idx]
        return c(unnormalize(x, (self.lower_bound, self.upper_bound)))

    def update_tr_length(self):
        # Update the length of the trust region according to
        # success and failure counters
        # (Just as in original TuRBO paper)
        state = self.state
        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        if state.length < state.length_min:  # Restart when trust region becomes too small
            state.restart_triggered = True

        self.state = state  # probably not necessary

    def update_state(self, Y_next, C_next):
        """Method used to update the TuRBO state after each step of optimization.

        Success and failure counters are updated according to the objective values 
        (Y_next) and constraint values (C_next) of the batch of candidate points 
        evaluated on the optimization step.

        As in the original TuRBO paper, a success is counted whenver any one of the 
        new candidate points improves upon the incumbent best point. The key difference 
        for SCBO is that we only compare points by their objective values when both points
        are valid (meet all constraints). If exactly one of the two points being compared 
        violates a constraint, the other valid point is automatically considered to be better. 
        If both points violate some constraints, we compare them inated by their constraint values.
        The better point in this case is the one with minimum total constraint violation
        (the minimum sum of constraint values)"""

        state = self.state

        # Determine which candidates meet the constraints (are valid)
        bool_tensor = C_next <= 0
        bool_tensor = torch.all(bool_tensor, dim=-1)
        Valid_Y_next = Y_next[bool_tensor]
        Valid_C_next = C_next[bool_tensor]
        if Valid_Y_next.numel() == 0:  # if none of the candidates are valid
            # pick the point with minimum violation
            sum_violation = C_next.sum(dim=-1)
            min_violation = sum_violation.min()
            # if the minimum voilation candidate is smaller than the violation of the incumbent
            if min_violation < state.best_constraint_values.sum():
                # count a success and update the current best point and constraint values
                state.success_counter += 1
                state.failure_counter = 0
                # new best is min violator
                state.best_value = Y_next[sum_violation.argmin()].item()
                state.best_constraint_values = C_next[sum_violation.argmin()]
            else:
                # otherwise, count a failure
                state.success_counter = 0
                state.failure_counter += 1
        else:  # if at least one valid candidate was suggested,
            # throw out all invalid candidates
            # (a valid candidate is always better than an invalid one)

            # Case 1: if the best valid candidate found has a higher objective value that 
            # incumbent best count a success, the obj valuse has been improved
            improved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(
                state.best_value
            )
            # Case 2: if incumbent best violates constraints
            # count a success, we now have suggested a point which is valid and thus better
            obtained_validity = torch.all(state.best_constraint_values > 0)
            if improved_obj or obtained_validity:  # If Case 1 or Case 2
                # count a success and update the best value and constraint values
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = max(Valid_Y_next).item()
                state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
            else:
                # otherwise, count a failure
                state.success_counter = 0
                state.failure_counter += 1

        # Finally, update the length of the trust region according to the
        # updated success and failure counters
        self.update_tr_length()
        

    def get_initial_points(self, n_pts, seed=0):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init


    def generate_batch(   
        self,
        model,          # GP model
        X,              # Evaluated points on the domain [0, 1]^d
        Y,              # Function values
        batch_size,
        n_candidates,   # Number of candidates for Thompson sampling
        constraint_model,
        X_space=None,   # Discrete Search space
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        state = self.state

        # Create the TR bounds
        x_center = X[Y.argmax(), :].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

        #### Thompson Sampling w/ Constraints (SCBO)
        if X_space is None:
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1 # guarantee at least one perturbation

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone() # clone n_cand times  
            X_cand[mask] = pert[mask]                           # apply the probability perturbation
        else:
            _filter = torch.all(X_space >= tr_lb, dim=-1).logical_and(torch.all(X_space <= tr_ub, dim=-1))
            assert _filter.sum() > 0
            X_cand = X_space[_filter]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
        )
        with torch.no_grad():
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        return X_next


    def get_fitted_model(self, X, Y):
        global_noise_constraint = Interval(1e-8, 1e-3)

        if self.dk:
            dk = DKL(X.float(), Y.float().squeeze(), n_iter=self.train_times, lr=self.learning_rate, low_dim=True, 
                pretrained_nn=None,  exact_gp=False, 
                noise_constraint = None if not self.constrain_noise else global_noise_constraint)
            dk.train_model(verbose=False)
            return dk.model
        
        dim = self.dim
        # global_noise_constraint = gpytorch.constraints.Interval(0.1,.6)

        likelihood = GaussianLikelihood(noise_constraint=global_noise_constraint)
        # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            ))
        # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            fit_gpytorch_mll(mll, max_attempts=100)

        return model

    def optimization(self, n_iter, **kwargs):
        process = tqdm(range(n_iter)) if self.verbose else range(n_iter)

        for _ in process:
            # Fit GP models for objective and constraints
            model = self.get_fitted_model(self.train_X, self.train_Y)
            c_model_list = [self.get_fitted_model(self.train_X, self.train_C_list[c_idx]) for c_idx in range(self.n_constraint)]


            # Generate a batch of candidates
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                X_next = self.generate_batch(
                    model=model,
                    X=self.train_X,
                    Y=self.train_Y,
                    batch_size=self.batch_size,
                    n_candidates=2000,
                    constraint_model=ModelListGP(*c_model_list),
                    X_space=kwargs.get("x_tensor", None)
                ).reshape([-1, self.dim])

            # Evaluate both the objective and constraints for the selected candidaates
            Y_next = torch.tensor(
                [self.eval_objective(x) for x in X_next], dtype=dtype, device=device
            ).unsqueeze(-1)

            _c_next_list = []
            for c_idx in range(self.n_constraint):
                _c_next = torch.tensor(
                    [self.eval_constraint(x, c_idx) for x in X_next], dtype=dtype, device=device
                ).unsqueeze(-1)
                _c_next_list.append(_c_next)


            C_next = torch.cat(_c_next_list, dim=-1)

            # Update TuRBO state
            self.update_state(Y_next, C_next)

            # Append data. Note that we append all data, even points that violate
            # the constraints. This is so our constraint models can learn more 
            # about the constraint functions and gain confidence in where violations occur.
            self.train_X = torch.cat((self.train_X, X_next), dim=0)
            self.train_Y = torch.cat((self.train_Y, Y_next), dim=0)
            for c_idx in range(self.n_constraint):
                self.train_C_list[c_idx] = torch.cat((self.train_C_list[c_idx], _c_next_list[c_idx]), dim=0)


            # Print current status. Note that state.best_value is always the best 
            # objective value found so far which meets the constraints, or in the case
            # that no points have been found yet which meet the constraints, it is the 
            # objective value of the point with the minimum constraint violation.
            if self.verbose:
                _v_info = f"Best value: {self.state.best_value:.2e}, TR length: {self.state.length:.2e}"
                process.set_postfix_str(_v_info)

        constraint_vals = torch.cat(self.train_C_list, dim=-1)
        bool_tensor = constraint_vals <= 0
        bool_tensor = torch.all(bool_tensor, dim=-1).unsqueeze(-1)
        _raw_reward = self.train_Y.squeeze() 
        self.reward = [_raw_reward[idx].detach().item() if bool_tensor[idx] else _raw_reward.min() for idx in range(_raw_reward.size(0))]

        return np.maximum.accumulate(self.reward)
            