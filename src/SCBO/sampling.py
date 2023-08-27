'''
Modify Original Sampling Implementation to support deep kernel.
'''

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gpytorch
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.sampling import batched_multinomial
from botorch.utils.transforms import standardize
from torch import Tensor
from torch.nn import Module

class SamplingStrategy(Module, ABC):
    r"""
    Abstract base class for sampling-based generation strategies.

    :meta private:
    """

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        elif not isinstance(objective, MCAcquisitionObjective):
            # TODO: Clean up once ScalarizedObjective is removed.
            if posterior_transform is not None:
                raise RuntimeError(
                    "A ScalarizedObjective (DEPRECATED) and a posterior transform "
                    "are not supported at the same time. Use only a posterior "
                    "transform instead."
                )
            else:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=objective.weights, offset=objective.offset
                )
                objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)


class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Sample from a set of points according to
    their max posterior value,
    which also likely meet a set of constraints
    c1(x) <= 0, c2(x) <= 0, ..., cm(x) <= 0
    c1, c2, ..., cm are black-box constraint functions
    Each constraint function is modeled by a seperate
    surrogate GP constraint model
    We sample points for which the posterior value
    for each constraint model <= 0,
    as described in https://doi.org/10.48550/arxiv.2002.08526

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(model,
                    constraint_model=ModelListGP(cmodel1, cmodel2,
                    ..., cmodelm)  # models w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        interpolate: bool = True,
        minimize_constraints_only: bool = False,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under
                which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel
                is a GP model for one constraint function,
                or a MultiTaskGP model where each task is one
                constraint function
                All constraints are of the form c(x) <= 0.
                In the case when the constraint model predicts
                that all candidates violate constraints,
                we pick the candidates with minimum violation.
            minimize_constraints_only: False by default, if true,
                we will automatically return the candidates
                with minimum posterior constraint values,
                (minimum predicted c(x) summed over all constraints)
                reguardless of predicted objective values.
        """
        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.interpolate = interpolate
        self.constraint_model = constraint_model
        self.minimize_constraints_only = minimize_constraints_only

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor
                from which to sample (in the `N`
                dimension) according to the maximum
                posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim
            Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        # posterior = self.model.posterior(X, observation_noise=observation_noise)
        # samples = posterior.rsample(sample_shape=torch.Size([num_samples])).T
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
            with gpytorch.settings.fast_pred_samples():
                posterior = self.model(X)
                samples = posterior.rsample(torch.Size([num_samples]))
                if self.interpolate:
                    samples = self.model.interpolation_calibrate(X, samples, cuda=self.model.if_cuda)
                samples = samples.T

        # c_posterior = self.constraint_model.posterior(
        #     X, observation_noise=observation_noise
        # )
        # constraint_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))
        # for DKL
        constraint_samples_list = []
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            for c_model in self.constraint_model.models:
                c_model.eval()
                with gpytorch.settings.fast_pred_samples():
                    posterior = c_model(X)
                    c_samples = posterior.rsample(torch.Size([num_samples]))
                    if self.interpolate:
                        c_samples = c_model.interpolation_calibrate(X, c_samples, cuda=c_model.if_cuda)
                    constraint_samples_list.append(c_samples)
        constraint_samples = torch.cat(constraint_samples_list, dim=0).T.unsqueeze(0)
        valid_samples = constraint_samples <= 0
        if valid_samples.shape[-1] > 1:  # if more than one constraint
            valid_samples = torch.all(valid_samples, dim=-1).unsqueeze(-1)
        if (valid_samples.sum() == 0) or self.minimize_constraints_only:
            # if none of the samples meet the constraints
            # we pick the one that minimizes total violation
            constraint_samples = constraint_samples.sum(dim=-1)
            idcs = torch.argmin(constraint_samples, dim=-1)
            if idcs.ndim > 1:
                idcs = idcs.permute(*range(1, idcs.ndim), 0)
            idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
            Xe = X.expand(*constraint_samples.shape[1:], X.size(-1))
            return torch.gather(Xe, -2, idcs)
        # replace all violators with -infinty so it will never choose them
        replacement_infs = -torch.inf * torch.ones(samples.shape).to(X.device).to(
            X.dtype
        )
        samples = torch.where(valid_samples, samples, replacement_infs)

        return self.maximize_samples(X, samples, num_samples)
