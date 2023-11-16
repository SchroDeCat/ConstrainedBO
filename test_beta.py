"""
Script for general purpose tests
"""
from src.utils import Constrained_Data_Factory
from src.opt import baseline_cbo_m, cbo_multi, baseline_scbo
from src.SCBO import SCBO
import torch
import numpy as np

EXPS = [
    "rastrigin_1d",
    "ackley_5d",
    "ackley_10d",
    "rosenbrock_5d",
    "rosenbrock_4d",
    "water_converter_32d",
    "water_converter_32d_neg",
    "water_converter_32d_neg_3c",
    "gpu_performance_16d",
    "vessel_4d_3c",
    "car_cab_7d_8c",
    "spring_3d_6c",
]
METHODs = [
    "cbo",
    "qei",
    "scbo",
    "ts",
    "random",
    "cmes-ibo",
]
PATH = "./res/beta"


def experiment(
    exp: str = "rastrigin_1d",
    method: str = "qei",
    n_repeat: int = 2,
    train_times: int = 5,
    n_iter: int = 20,
    n_init: int = 10,
    constrain_noise: bool = True,
    interpolate: bool = True,
    c_portion: float = None,
    low_dim: bool = True,
    exact_gp: bool = False,
    beta: float = 10,
    filter_beta: float = 10,
) -> None:
    exp = exp.lower()
    method = method.lower()
    assert exp in EXPS
    assert method in METHODs
    name = f"{exp.upper()}"
    lr = 1e-4

    ### exp
    if exp == "rastrigin_1d":  # rastrigin 1D
        cbo_factory = Constrained_Data_Factory(
            num_pts=20000 if c_portion is None else 1000
        )
        scbo = "scbo" in method
        if not c_portion is None:  # scanning the portion
            if scbo:
                x_tensor, y_func, c_func_list = cbo_factory.rastrigin_1D_1C(
                    scbo_format=scbo, c_scan=True, c_portion=c_portion
                )
            else:
                x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D_1C(
                    scbo_format=scbo, c_scan=True, c_portion=c_portion
                )
        else:
            if scbo:
                x_tensor, y_func, c_func_list = cbo_factory.rastrigin_1D_1C(
                    scbo_format=scbo
                )
            else:
                x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D_1C(
                    scbo_format=scbo
                )
        constraint_threshold_list, constraint_confidence_list = (
            cbo_factory.constraint_threshold_list,
            cbo_factory.constraint_confidence_list,
        )
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        constrain_noise = False

        filter_beta = 1
        # beta = 2.90
        cbo_factory.visualize_1d()

    elif exp == "ackley_5d":
        cbo_factory = Constrained_Data_Factory(num_pts=20000 // 2)
        scbo = "scbo" in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.ackley_5D_2C(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.ackley_5D_2C(
                scbo_format=scbo
            )
        constraint_threshold_list, constraint_confidence_list = (
            cbo_factory.constraint_threshold_list,
            cbo_factory.constraint_confidence_list,
        )
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)

        # beta = 0.1  # actually is 10 here?
        filter_beta = 4
        constrain_noise = True

    elif exp == "water_converter_32d_neg_3c":
        cbo_factory = Constrained_Data_Factory(num_pts=10000)
        scbo = "scbo" in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.water_converter_32d_neg_3c(
                scbo_format=scbo
            )
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.water_converter_32d_neg_3c(
                scbo_format=scbo
            )
        constraint_threshold_list, constraint_confidence_list = (
            cbo_factory.constraint_threshold_list,
            cbo_factory.constraint_confidence_list,
        )
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)
        constrain_noise = False
        filter_beta = 20
        # beta = 20

    else:
        raise NotImplementedError(f"Exp {exp} no implemented")

    ### method
    print(
        f"{method} initial reward {y_tensor[:n_init][feasible_filter[:n_init]].squeeze()} while global max {y_tensor[feasible_filter].max().item()}"
    )
    if method in ["cmes-ibo", "ts", "qei", "random"]:

        regret = baseline_cbo_m(
            x_tensor,
            y_tensor,
            c_tensor_list,
            constraint_threshold_list=constraint_threshold_list,
            constraint_confidence_list=constraint_confidence_list,
            n_init=n_init,
            n_repeat=n_repeat,
            train_times=train_times,
            n_iter=n_iter,
            regularize=False,
            low_dim=low_dim,
            spectrum_norm=False,
            retrain_interval=1,
            acq=method,
            verbose=True,
            lr=1e-4,
            name=name,
            return_result=True,
            retrain_nn=True,
            plot_result=True,
            save_result=True,
            save_path=f"{PATH}",
            fix_seed=True,
            pretrained=False,
            ae_loc=None,
            exact_gp=exact_gp,
            constrain_noise=constrain_noise,
            interpolate=interpolate,
        )

    elif method == "cbo":
        regret = cbo_multi(
            x_tensor,
            y_tensor,
            c_tensor_list,
            constraint_threshold_list=constraint_threshold_list,
            constraint_confidence_list=constraint_confidence_list,
            n_init=n_init,
            n_repeat=n_repeat,
            train_times=train_times,
            regularize=False,
            low_dim=low_dim,
            spectrum_norm=False,
            retrain_interval=1,
            n_iter=n_iter,
            filter_interval=1,
            acq="ci",
            ci_intersection=False,
            verbose=True,
            lr=1e-4,
            name=name,
            return_result=True,
            retrain_nn=True,
            plot_result=True,
            save_result=True,
            save_path=f"{PATH}",
            fix_seed=True,
            pretrained=False,
            ae_loc=None,
            _minimum_pick=10,
            _delta=0.01,
            beta=beta,
            filter_beta=filter_beta,
            exact_gp=exact_gp,
            constrain_noise=constrain_noise,
            local_model=False,
            interpolate=interpolate,
        )

    elif method == "scbo":
        init_feasible_reward = y_tensor[:n_init][feasible_filter[:n_init]]
        if init_feasible_reward.size(0) > 0:
            max_reward = init_feasible_reward.max().item()
        else:
            max_reward = -torch.inf
        max_global = y_tensor[feasible_filter].max().item()
        regret = np.zeros([n_repeat, n_iter])
        regret = baseline_scbo(
            x_tensor=x_tensor,
            y_func=y_func,
            c_func_list=c_func_list,
            max_global=max_global,
            lb=cbo_factory.lb,
            ub=cbo_factory.ub,
            dim=cbo_factory.dim,
            n_init=n_init,
            n_repeat=n_repeat,
            train_times=train_times,
            low_dim=low_dim,
            retrain_interval=1,
            n_iter=n_iter,
            verbose=True,
            lr=lr,
            name=name,
            return_result=True,
            plot_result=True,
            save_result=True,
            save_path=f"{PATH}",
            fix_seed=True,
            exact_gp=exact_gp,
            constrain_noise=constrain_noise,
            interpolate=interpolate,
        )

    else:
        raise NotImplementedError(f"Method {method} no implemented")

    print(f"With constraints, the minimum regret we found is: {regret.min(axis=-1)}")


if __name__ == "__main__":
    n_repeat = 15
    n_iter = 200
    # n_repeat = 2
    # n_iter = 2

    # for method in ["cbo", "cmes-ibo", "qei", "scbo"]:
    method = 'cbo'
    for beta in [0, 0.1, 2.9, 10]:
        experiment(
            exp="rastrigin_1d",
            n_init=5,
            n_iter=n_iter,
            n_repeat=n_repeat,
            method=method,
            train_times=10,
            beta = beta,
        )
        experiment(
            exp="ackley_5d", 
            n_init=20, 
            n_iter=n_iter, 
            n_repeat=n_repeat, 
            method=method,
            beta = beta,
        )
        experiment(
            exp="water_converter_32d_neg_3c",
            n_init=10,
            n_iter=n_iter,
            n_repeat=n_repeat,
            method=method,
            beta = beta,
        )
