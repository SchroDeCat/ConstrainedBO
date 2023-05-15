'''
Script for general purpose tests
'''
from src.utils import Constrained_Data_Factory
from src.opt import baseline_cbo_m, cbo_multi
from src.SCBO import SCBO
import torch
import numpy as np
import random
import warnings
import tqdm

EXPS = ['rastrigin_1d', 'rastrigin_10d', 'ackley_5d', 'rosenbrock_5d', 'water_converter_32d', 'gpu_performance_16d']
METHODs = ['cbo', 'scbo', 'cmes-ibo', 'ts', 'qei', 'random']

def experiment(exp:str='rastrigin_1d', method:str='qei', n_repeat:int=2, train_times:int=50, n_iter:int=20, n_init:int=10, constrain_noise:bool=True)->None:
    exp = exp.lower()
    method = method.lower()
    assert exp in EXPS
    assert method in METHODs
    name = f"{exp.upper()}"
    lr = 1e-4
    ### exp
    if exp == 'rastrigin_1d': # rastrigin 1D
        cbo_factory = Constrained_Data_Factory(num_pts=20000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.rastrigin_1D(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d()
    elif exp == "ackley_5d":
        cbo_factory = Constrained_Data_Factory(num_pts=100000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.ackley_5d(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.ackley_5d(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
    else:
        raise NotImplemented(f"Exp {exp} no implemented")

    ### method
    print(f"initial reward {y_tensor[:n_init][feasible_filter[:n_init]].squeeze()} while global max {y_tensor[feasible_filter].max().item()}")
    if method in ['cmes-ibo', 'ts', 'qei', 'random']:

        regret = baseline_cbo_m(x_tensor, y_tensor, c_tensor_list, 
                                constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
                                n_init=n_init, n_repeat=n_repeat, train_times=train_times, n_iter=n_iter,
                                regularize=False, low_dim=True,
                                spectrum_norm=False, retrain_interval=1, acq=method, 
                                verbose=True, lr=1e-4, name=name, 
                                return_result=True, retrain_nn=True,
                                plot_result=True, save_result=True, save_path=f'./res/baseline/{method}', 
                                fix_seed=True,  pretrained=False, ae_loc=None, 
                                exact_gp=False, constrain_noise=constrain_noise,)
        
    elif method == 'cbo':

        regret = cbo_multi(x_tensor, y_tensor, c_tensor_list, 
                    constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
                    n_init=n_init, n_repeat=n_repeat, train_times=train_times, regularize=False, low_dim=True,
                    spectrum_norm=False, retrain_interval=1, n_iter=n_iter, filter_interval=1, acq="ci", 
                    ci_intersection=True, verbose=True, lr=1e-4, name=name, return_result=True, retrain_nn=True,
                    plot_result=True, save_result=True, save_path='./res/cbo', fix_seed=True,  pretrained=False, ae_loc=None, 
                    _minimum_pick = 10, _delta = 0.2, beta=1, filter_beta=1, exact_gp=False, constrain_noise=constrain_noise, local_model=False)

    elif method =='scbo':
        init_feasible_reward = y_tensor[:n_init][feasible_filter[:n_init]]
        max_reward = init_feasible_reward.max().item()
        max_global = y_tensor[feasible_filter].max().item()
        regret = np.zeros([n_repeat, n_iter])
        print(f"Feasible Y {init_feasible_reward}")
        print(f"Before Optimization the best value is: {max_reward:.4f} / global opt {max_global:.4f} := regret {max_global - max_reward:.4f} ")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rep in tqdm.tqdm(range(n_repeat), desc=f"Experiment Rep"):
                # set seed
                _seed = rep * 20 + n_init
                torch.manual_seed(_seed)
                np.random.seed(_seed)
                random.seed(_seed)
                torch.cuda.manual_seed(_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                batch_size = 1
                scbo = SCBO(y_func, c_func_list, dim=cbo_factory.dim, lower_bound=cbo_factory.lb, upper_bound=cbo_factory.ub, 
                                    train_times=train_times, lr=lr,
                                    batch_size = batch_size, n_init=n_init, 
                                    train_X = x_tensor[:n_init].reshape([-1,cbo_factory.dim]), dk= True)

                rewards = scbo.optimization(n_iter=n_iter//batch_size, x_tensor=x_tensor)
                regrets = max_global - rewards
                regret[rep] = regrets[-n_iter:]
    else:
        raise NotImplemented(f"Method {method} no implemented")

    print(f"With constraints, the minimum regret we found is: {regret.min(axis=-1)}")



if __name__ == "__main__":
    n_repeat = 10
    n_init = 5
    n_iter = 50
    # experiment(n_init=5, method='qei')
    # experiment(n_init=5, method='ts')
    # experiment(n_init=5, method='cmes-ibo')
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=40, method='scbo')
    # experiment(n_init=5, method='cbo', n_iter=200)
    experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='cbo')
    experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='qei')
    experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='ts')
    experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='cmes-ibo')
    experiment(exp='ackley_5d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='cbo')
    experiment(exp='ackley_5d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='qei')
    experiment(exp='ackley_5d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='ts')
    experiment(exp='ackley_5d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method='cmes-ibo')
    # experiment(n_init=10, method='cbo')


