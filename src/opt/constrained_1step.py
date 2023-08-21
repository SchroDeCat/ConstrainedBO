"""
Full pipeline for constrained BO. Only pick one candidate.
"""

import gpytorch
import torch
import warnings
import numpy as np

from ..models import DKL, AE, beta_CI
from ..utils import model_list_CI, intersecting_ROI_globe
from .dkbo_ae_constrained import DK_BO_AE_C_M
from scipy.stats import norm

DEVICE = torch.device("cpu")


def cbo_multi_nontest(
    x_tensor,
    y_tensor,
    c_tensor_list,
    constraint_threshold_list,
    constraint_confidence_list,
    observed,
    train_times=10,
    beta=2,
    regularize=False,
    low_dim=True,
    spectrum_norm=False,
    acq="ci",
    ci_intersection=True,
    verbose=True,
    lr=1e-2,
    pretrained=False,
    ae_loc=None,
    _minimum_pick=10,
    _delta=0.2,
    filter_beta=0.05,
    exact_gp=False,
    constrain_noise=False,
    local_model=True,
    output_scale_constraint=None,
):
    """
    Proposed ROI based method, default acq = ci
    Support Multiple Constraints
    In real world application where observations are not available.
    """
    ####### configurations
    if constrain_noise:
        global_noise_constraint = gpytorch.constraints.Interval(1e-5, 0.08)
        roi_noise_constraint = gpytorch.constraints.Interval(1e-5, 0.1)
    else:
        global_noise_constraint = None
        roi_noise_constraint = None

    n_init = int(observed.sum())
    _minimum_pick = min(_minimum_pick, n_init)
    c_threshold_list = [
        norm.ppf(constraint_confidence, loc=constraint_threshold, scale=1)
        for constraint_confidence, constraint_threshold in zip(
            constraint_confidence_list, constraint_threshold_list
        )
    ]
    c_num = len(c_tensor_list)

    feasibility_filter = c_tensor_list[0] > c_threshold_list[0]
    for c_idx in range(c_num):
        feasibility_filter.logical_and(c_tensor_list[c_idx] > c_threshold_list[c_idx])

    assert torch.any(feasibility_filter)

    ####### init dkl and generate f_ucb for partition
    data_size = x_tensor.size(0)
    assert y_tensor.squeeze().size(0) == data_size
    for c_idx in range(1, c_num):
        assert c_tensor_list[c_idx].squeeze().size(0) == data_size
        if (
            len(y_tensor.size()) > 2
            or len(c_tensor_list[c_idx].size()) > 2
            or len(x_tensor.size()) > 2
        ):
            raise ValueError(f"Shape of input tensor is ")
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
        ####### init in each round
        # observed = np.zeros(data_size)
        # observed[:n_init] = 1
        init_x = x_tensor[observed==1]
        init_y = y_tensor[observed==1]
        init_c_list = [c_tensor_list[c_idx][observed==1] for c_idx in range(c_num)]
        # NOTE: AE is shared for f and c
        _f_model = DKL(
            init_x,
            init_y.squeeze(),
            n_iter=train_times,
            low_dim=low_dim,
            lr=lr,
            spectrum_norm=spectrum_norm,
            exact_gp=exact_gp,
            noise_constraint=global_noise_constraint,
            pretrained_nn=ae,
            output_scale_constraint=output_scale_constraint,
        )
        _c_model_list = [
            DKL(
                init_x,
                init_c_list[c_idx].squeeze(),
                n_iter=train_times,
                low_dim=low_dim,
                lr=lr,
                spectrum_norm=spectrum_norm,
                exact_gp=exact_gp,
                noise_constraint=global_noise_constraint,
                pretrained_nn=ae,
                output_scale_constraint=output_scale_constraint,
            )
            for c_idx in range(c_num)
        ]
        if regularize:
            _f_model.train_model_kneighbor_collision()
            for c_idx in range(c_num):
                _c_model_list[c_idx].train_model_kneighbor_collision()
        else:
            _f_model.train_model(verbose=False)
            for c_idx in range(c_num):
                _c_model_list[c_idx].train_model(verbose=False)

        f_lcb, f_ucb = _f_model.CI(x_tensor.to(DEVICE))
        c_lcb_list, c_ucb_list = model_list_CI(_c_model_list, x_tensor, DEVICE)

        ####### each test instance
        # optimization CI
        if default_beta:
            beta = (
                2
                * np.log((x_tensor.size(0) * (np.pi * (n_init + 1)) ** 2) / (6 * _delta))
            ) ** 0.5  # analytic beta
        _f_lcb, _f_ucb = beta_CI(f_lcb, f_ucb, beta)
        _c_lcb_list, _c_ucb_list = [None for _ in range(c_num)], [ None for _ in range(c_num)]
        for c_idx in range(c_num):
            _c_lcb_list[c_idx], _c_ucb_list[c_idx] = beta_CI(
                c_lcb_list[c_idx], c_ucb_list[c_idx], beta
            )

        # Take intersection of all historical CIs
        if True:
            f_max_test_x_lcb, f_min_test_x_ucb = _f_lcb.clone(), _f_ucb.clone()
            c_max_test_x_lcb_list, c_min_test_x_ucb_list = [_c_lcb.clone() for _c_lcb in _c_lcb_list], [_c_ucb.clone() for _c_ucb in _c_ucb_list]

        # Identify f_roi, csi, cui, c_roi, and general ROI
        if default_fbeta:
            filter_beta = beta

        filter_on_intersect = False
        if filter_on_intersect:
            _f_filter_lcb, _f_filter_ucb = f_max_test_x_lcb, f_min_test_x_ucb
            _c_filter_lcb_list, _c_filter_ucb_list = (
                c_max_test_x_lcb_list,
                c_min_test_x_ucb_list,
            )
        else:
            _f_filter_lcb, _f_filter_ucb = beta_CI(f_lcb, f_ucb, filter_beta)
            _c_filter_lcb_list, _c_filter_ucb_list = [None for _ in range(c_num)], [
                None for _ in range(c_num)
            ]
            for c_idx in range(c_num):
                _c_filter_lcb_list[c_idx], _c_filter_ucb_list[c_idx] = beta_CI(
                    c_lcb_list[c_idx], c_ucb_list[c_idx], filter_beta
                )

        c_sci_filter_list, c_roi_filter_list, c_uci_filter_list = (
            [None for _ in range(c_num)],
            [None for _ in range(c_num)],
            [None for _ in range(c_num)],
        )

        for c_idx in range(c_num):
            c_sci_filter_list[c_idx] = (
                _c_filter_lcb_list[c_idx] >= c_threshold_list[c_idx]
            )
            c_roi_filter_list[c_idx] = (
                _c_filter_ucb_list[c_idx] >= c_threshold_list[c_idx]
            )
            c_uci_filter_list[c_idx] = c_roi_filter_list[c_idx].logical_xor(
                c_sci_filter_list[c_idx]
            )

        c_sci_filter = c_sci_filter_list[0].clone()  # single c_sci
        for c_idx in range(c_num):
            c_sci_filter = c_sci_filter.logical_and(c_sci_filter_list[c_idx])

        f_roi_threshold = (
            _f_filter_lcb[c_sci_filter.squeeze()].max()
            if torch.any(c_sci_filter)
            else _f_filter_lcb[feasibility_filter.squeeze()].min()
        )  # single f_roi
        f_roi_filter = _f_filter_ucb >= f_roi_threshold

        roi_filter = f_roi_filter.clone()  # single general roi
        for c_roi_filter in c_roi_filter_list:  # general ROI.
            roi_filter = roi_filter.logical_and(c_roi_filter)

        if sum(roi_filter[observed == 1]) <= _minimum_pick:
            c_ucb_observed_min = torch.min(
                torch.cat(
                    [c_ucb[observed == 1].reshape(1, -1) for c_ucb in c_ucb_list], dim=0
                ),
                dim=0,
            ).values
            _, indices = torch.topk(
                c_ucb_observed_min, min(_minimum_pick, c_ucb_observed_min.size(0))
            )
            for idx in indices:
                roi_filter[util_array[observed == 1][idx]] = 1

        for c_idx in range(c_num):  # c_uci_list intersects general ROI
            c_uci_filter_list[c_idx] = c_uci_filter_list[c_idx].logical_and(roi_filter)

        # ROI data
        observed_unfiltered = np.min(
            [observed, roi_filter.numpy()], axis=0
        )  # observed and not filtered outs
        init_x = x_tensor[observed_unfiltered == 1]
        init_y = y_tensor[observed_unfiltered == 1]
        init_c_list = [c_tensor[observed_unfiltered == 1] for c_tensor in c_tensor_list]

        # optimization
        if local_model:  # allow training a local model and optimize on top of it
            _f_model_passed_in, _c_model_list_passed_in = None, None
        else:
            _f_model_passed_in, _c_model_list_passed_in = _f_model, _c_model_list

        _cbo_m = DK_BO_AE_C_M(
            x_tensor,
            y_tensor,
            c_tensor_list,
            roi_filter,
            c_uci_filter_list,
            lr=lr,
            spectrum_norm=spectrum_norm,
            low_dim=low_dim,
            n_init=n_init,
            train_iter=train_times,
            regularize=regularize,
            dynamic_weight=False,
            retrain_nn=True,
            c_threshold_list=c_threshold_list,
            max=0,
            pretrained_nn=ae,
            verbose=verbose,
            init_x=init_x,
            init_y=init_y,
            init_c_list=init_c_list,
            exact_gp=exact_gp,
            noise_constraint=roi_noise_constraint,
            f_model=_f_model_passed_in,
            c_model_list=_c_model_list_passed_in,
            observed=observed,
            output_scale_constraint=output_scale_constraint,
            standardize=False,
        )

        _roi_f_lcb, _roi_f_ucb = _cbo_m.f_model.CI(x_tensor)
        _roi_c_lcb_list, _roi_c_ucb_list = model_list_CI(
            _cbo_m.c_model_list, x_tensor, DEVICE
        )

        # if ci_intersection:
        if not (default_beta):  # only for visualization & intersection
            _roi_beta = min(1e2, max(1e-2, f_ucb.max() / _roi_f_ucb.max()))
        else:
            _roi_beta = (
                2
                * np.log(
                    (x_tensor[roi_filter].shape[0] * (np.pi * (n_init + 1)) ** 2)
                    / (6 * _delta)
                )
            ) ** 0.5  # analytic beta

        # intersection of ROI CI and global CI
        if ci_intersection:
            (
                f_max_test_x_lcb,
                f_min_test_x_ucb,
                _roi_f_lcb_scaled,
                _roi_f_ucb_scaled,
            ) = intersecting_ROI_globe(
                f_max_test_x_lcb,
                f_min_test_x_ucb,
                _roi_f_lcb,
                _roi_f_ucb,
                _roi_beta,
                roi_filter,
            )
            for c_idx, (
                c_max_test_x_lcb,
                c_min_test_x_ucb,
                _roi_c_lcb,
                _roi_c_ucb,
            ) in enumerate(
                zip(
                    c_max_test_x_lcb_list,
                    c_min_test_x_ucb_list,
                    _roi_c_lcb_list,
                    _roi_c_ucb_list,
                )
            ):
                (
                    c_max_test_x_lcb_list[c_idx],
                    c_min_test_x_ucb_list[c_idx],
                    _,
                    _,
                ) = intersecting_ROI_globe(
                    c_max_test_x_lcb,
                    c_min_test_x_ucb,
                    _roi_c_lcb,
                    _roi_c_ucb,
                    _roi_beta,
                    roi_filter,
                )
        else:
            _, _, _roi_f_lcb_scaled, _roi_f_ucb_scaled = intersecting_ROI_globe(
                f_max_test_x_lcb,
                f_min_test_x_ucb,
                _roi_f_lcb,
                _roi_f_ucb,
                _roi_beta,
                roi_filter,
            )

        # optimize f and learn c
        query_num = 1
        _roi_beta_passed_in = (
            _roi_beta if not (default_beta) else 0
        )  
        # allow it to calculate internal ROI_beta
        _cbo_m.query_f_c(
            n_iter=query_num,
            acq=acq,
            study_interval=10,
            study_res_path=None,
            if_tqdm=False,
            retrain_interval=1,
            ci_intersection=ci_intersection,
            f_max_test_x_lcb=f_max_test_x_lcb,
            f_min_test_x_ucb=f_min_test_x_ucb,
            c_max_test_x_lcb_list=c_max_test_x_lcb_list,
            c_min_test_x_ucb_list=c_min_test_x_ucb_list,
            beta=_roi_beta_passed_in,
        )

    return _f_model, _c_model_list, _cbo_m
