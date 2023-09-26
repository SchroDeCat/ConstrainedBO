'''
Script for general purpose tests
'''
from src.utils import Constrained_Data_Factory
from src.opt import baseline_cbo_m, cbo_multi, baseline_scbo
from src.SCBO import SCBO
import torch
import numpy as np
import random
import warnings
import tqdm

EXPS = ['rastrigin_1d', 'rastrigin_10d', 'ackley_5d', 'ackley_10d','rosenbrock_5d', 'rosenbrock_4d', 
        'water_converter_32d', 'water_converter_32d_neg', 'water_converter_32d_neg_3c', 'gpu_performance_16d', 
        'vessel_4d_3c', 'car_cab_7d_8c', 'spring_3d_6c']
METHODs = ['cbo',  'qei', 'scbo', 'ts','random', 'cmes-ibo', ]


def experiment(exp:str='rastrigin_1d', method:str='qei', n_repeat:int=2, train_times:int=5, n_iter:int=20, n_init:int=10, 
               constrain_noise:bool=True, interpolate:bool=True, c_portion:float=None, low_dim:bool=True, exact_gp:bool=False,
               beta:float = 10, filter_beta:float = 10  )->None:
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
        if not c_portion is None: # scanning the portion
            if scbo:
                x_tensor, y_func, c_func_list = cbo_factory.rastrigin_1D_1C(scbo_format=scbo, c_scan=True, c_portion=c_portion)
            else:
                x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D_1C(scbo_format=scbo, c_scan=True, c_portion=c_portion)
        else:
            if scbo:
                x_tensor, y_func, c_func_list = cbo_factory.rastrigin_1D_1C(scbo_format=scbo)
            else:
                x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D_1C(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d()
    elif exp == "ackley_5d":
        # cbo_factory = Constrained_Data_Factory(num_pts=100000)
        cbo_factory = Constrained_Data_Factory(num_pts=20000//2)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.ackley_5D_2C(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.ackley_5D_2C(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)

        beta = .1
        filter_beta = 4
        constrain_noise = True

    elif exp == "rosenbrock_5d":
        cbo_factory = Constrained_Data_Factory(num_pts=50000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.rosenbrock_5d(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.rosenbrock_5d(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)
    elif exp == "rosenbrock_4d":
        cbo_factory = Constrained_Data_Factory(num_pts=100000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.rosenbrock_4d(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.rosenbrock_4d(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)

    elif exp == "water_converter_32d":
        cbo_factory = Constrained_Data_Factory(num_pts=30000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.water_converter_32d(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.water_converter_32d(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)

    elif exp == "water_converter_32d_neg":
        cbo_factory = Constrained_Data_Factory(num_pts=30000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.water_converter_32d_neg(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.water_converter_32d_neg(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)

    elif exp == "water_converter_32d_neg_3c":
        cbo_factory = Constrained_Data_Factory(num_pts=10000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.water_converter_32d_neg_3c(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.water_converter_32d_neg_3c(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)
        # filter_beta = 2
        # beta = 2
        constrain_noise = False
        filter_beta = 20
        beta = 20

    elif exp == "vessel_4d_3c":
        # cbo_factory = Constrained_Data_Factory(num_pts=40000)
        cbo_factory = Constrained_Data_Factory(num_pts=10000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.RE2_4D_3C(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.RE2_4D_3C(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)
        constrain_noise = False
        # filter_beta = 2
        # beta = 2
        filter_beta = 2
        beta = 2

    elif exp == "spring_3d_6c":
        # cbo_factory = Constrained_Data_Factory(num_pts=10000)
        # cbo_factory = Constrained_Data_Factory(num_pts=2000)
        cbo_factory = Constrained_Data_Factory(num_pts=20000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.RE2_3D_6C(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.RE2_3D_6C(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)
        constrain_noise = False  
        filter_beta = 2
        # beta = 10
        beta = 2
    elif exp == "car_cab_7d_8c":
        # cbo_factory = Constrained_Data_Factory(num_pts=5000)
        cbo_factory = Constrained_Data_Factory(num_pts=20000)
        scbo = 'scbo' in method
        if scbo:
            x_tensor, y_func, c_func_list = cbo_factory.RE9_7D_8C(scbo_format=scbo)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.RE9_7D_8C(scbo_format=scbo)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        y_tensor = cbo_factory.y_tensor
        cbo_factory.visualize_1d(if_norm=True)   
        constrain_noise = False  
        # filter_beta = 2
        # beta = 2
        # filter_beta = 20
        # beta = 20
        filter_beta = 2
        # beta = 10
        beta = 2
    else:
        raise NotImplementedError(f"Exp {exp} no implemented")

    ### method
    print(f"{method} initial reward {y_tensor[:n_init][feasible_filter[:n_init]].squeeze()} while global max {y_tensor[feasible_filter].max().item()}")
    if method in ['cmes-ibo', 'ts', 'qei', 'random']:

        regret = baseline_cbo_m(x_tensor, y_tensor, c_tensor_list, 
                                constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
                                n_init=n_init, n_repeat=n_repeat, train_times=train_times, n_iter=n_iter,
                                regularize=False, low_dim=low_dim,
                                spectrum_norm=False, retrain_interval=1, acq=method, 
                                verbose=True, lr=1e-4, name=name, 
                                return_result=True, retrain_nn=True,
                                plot_result=True, save_result=True, save_path=f'./res/baseline/tmlr/{method}', 
                                fix_seed=True,  pretrained=False, ae_loc=None, 
                                exact_gp=exact_gp, constrain_noise=constrain_noise,
                                interpolate=interpolate,)
        
    elif method == 'cbo':

        # regret = cbo_multi(x_tensor, y_tensor, c_tensor_list, 
        #             constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
        #             n_init=n_init, n_repeat=n_repeat, train_times=train_times, regularize=False, low_dim=False,
        #             spectrum_norm=False, retrain_interval=1, n_iter=n_iter, filter_interval=1, acq="ci", 
        #             ci_intersection=True, verbose=True, lr=1e-4, name=name, return_result=True, retrain_nn=True,
        #             plot_result=True, save_result=True, save_path='./res/cbo/tmlr', fix_seed=True,  pretrained=False, ae_loc=None, 
        #             _minimum_pick = 10, _delta = 0.01, beta=0.5, filter_beta=0.5, exact_gp=False, constrain_noise=constrain_noise, 
        #             local_model=False,  interpolate=interpolate,)
        regret = cbo_multi(x_tensor, y_tensor, c_tensor_list, 
                    constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
                    n_init=n_init, n_repeat=n_repeat, train_times=train_times, regularize=False, low_dim=low_dim,
                    spectrum_norm=False, retrain_interval=1, n_iter=n_iter, filter_interval=1, acq="ci", 
                    ci_intersection=False, verbose=True, lr=1e-4, name=name, return_result=True, retrain_nn=True,
                    plot_result=True, save_result=True, save_path='./res/cbo/tmlr', fix_seed=True,  pretrained=False, ae_loc=None, 
                    _minimum_pick = 10, _delta = 0.01, beta=beta, filter_beta=filter_beta, exact_gp=exact_gp, constrain_noise=constrain_noise, 
                    local_model=False,  interpolate=interpolate,)

    elif method =='scbo':
        init_feasible_reward = y_tensor[:n_init][feasible_filter[:n_init]]
        if init_feasible_reward.size(0) > 0:
            max_reward = init_feasible_reward.max().item()
        else:
            max_reward = -torch.inf
        max_global = y_tensor[feasible_filter].max().item()
        regret = np.zeros([n_repeat, n_iter])
        # print(f"Feasible Y {init_feasible_reward}")
        # print(f"Before Optimization the best value is: {max_reward:.4f} / global opt {max_global:.4f} := regret {max_global - max_reward:.4f} ")
        regret = baseline_scbo(x_tensor=x_tensor, y_func=y_func, c_func_list=c_func_list,
            max_global=max_global, lb=cbo_factory.lb, ub=cbo_factory.ub, dim=cbo_factory.dim,
            n_init=n_init, n_repeat=n_repeat, train_times=train_times, low_dim=low_dim,
            retrain_interval=1, n_iter=n_iter,
            verbose=True, lr=lr, name=name, return_result=True, 
            plot_result=True, save_result=True, save_path='./res/scbo/tmlr', fix_seed=True,
            exact_gp=exact_gp, constrain_noise=constrain_noise, interpolate=interpolate)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     for rep in tqdm.tqdm(range(n_repeat), desc=f"Experiment Rep"):
        #         # set seed
        #         _seed = rep * 20 + n_init
        #         torch.manual_seed(_seed)
        #         np.random.seed(_seed)
        #         random.seed(_seed)
        #         torch.cuda.manual_seed(_seed)
        #         torch.backends.cudnn.benchmark = False
        #         torch.backends.cudnn.deterministic = True
        #         batch_size = 1
        #         scbo = SCBO(y_func, c_func_list, dim=cbo_factory.dim, lower_bound=cbo_factory.lb, upper_bound=cbo_factory.ub, 
        #                             train_times=train_times, lr=lr,
        #                             batch_size = batch_size, n_init=n_init, 
        #                             train_X = x_tensor[:n_init].reshape([-1, cbo_factory.dim]), dk= True)

        #         rewards = scbo.optimization(n_iter=n_iter//batch_size, x_tensor=x_tensor)
        #         regrets = max_global - rewards
        #         regret[rep] = regrets[-n_iter:]
    else:
        raise NotImplementedError(f"Method {method} no implemented")

    print(f"With constraints, the minimum regret we found is: {regret.min(axis=-1)}")



if __name__ == "__main__":
    # n_repeat = 10
    # n_init = 10
    # n_iter = 50
    # train_times = 50
    # n_repeat = 2
    n_repeat = 15
    n_init = 5
    n_init2 = 20
    n_init3 = 10
    # n_iter = 30
    n_iter = 200
    # n_iter = 50
    # train_times = 5
    # train_times = 10

    # experiment(n_init=5, method='qei')
    # experiment(n_init=5, method='ts')
    # experiment(n_init=5, method='cmes-ibo')
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=40, method='scbo')
    # experiment(n_init=5, method='cbo', n_iter=200)
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='cbo')
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='qei')
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='ts')
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='cmes-ibo')
    # experiment(exp='rastrigin_1d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='scbo')
    # experiment(exp='ackley_5d', n_init=n_init2, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='cbo')
    # experiment(exp='ackley_5d', n_init=n_init2, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='qei')
    # experiment(exp='ackley_5d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='ts')
    # experiment(exp='ackley_5d', n_init=n_init2, n_repeat=n_repeat, n_iter=n_iter,train_times=train_times,  method='cmes-ibo')
    # experiment(exp='ackley_5d', n_init=n_init2, n_repeat=n_repeat, n_iter=n_iter, train_times=train_times, method='scbo')
    # experiment(exp='rosenbrock_5d', n_init=10, n_repeat=1, n_iter=20, method='qei', constrain_noise=True)
    # experiment(exp='rosenbrock_5d', n_init=10, n_repeat=1, n_iter=20, method='cmes-ibo', constrain_noise=True)
    # experiment(exp='rosenbrock_5d', n_init=10, n_repeat=1, n_iter=20, method='scbo', constrain_noise=True)
    # experiment(exp='rosenbrock_4d', n_init=20, n_repeat=3, n_iter=20, method='cbo', constrain_noise=True)
    # for method in METHODs:
    #     print(f"Method {method}")
    #     experiment(exp='rosenbrock_4d', n_init=n_init, n_repeat=n_repeat, n_iter=n_iter, method=method, constrain_noise=True)
    
    # experiment(n_init=10, method='cbo')
    # experiment(exp='water_converter_32d_neg', n_init=10, n_repeat=n_repeat, n_iter=n_iter, method='cbo', constrain_noise=True)
    # originally train_iter = 50, now 5
    # experiment(exp='water_converter_32d_neg_3c', n_init=n_init3, n_repeat=n_repeat, n_iter=n_iter, method='cbo',)
    # experiment(exp='water_converter_32d_neg_3c', n_init=n_init3, n_repeat=n_repeat, n_iter=n_iter, method='scbo',)
    # experiment(exp='water_converter_32d_neg_3c', n_init=n_init3, n_repeat=n_repeat, n_iter=n_iter, method='qei', )
    # experiment(exp='water_converter_32d_neg_3c', n_init=n_init3, n_repeat=n_repeat, n_iter=n_iter, method='cmes-ibo', )

    for method in METHODs:
        # if method in ['cbo', 'cmes-ibo']:
        if method in ['ts', 'random']:
            continue
        # experiment(exp='water_converter_32d_neg_3c', n_init=20, n_repeat=10, n_iter=100, method=method, constrain_noise=True)
        experiment(exp="vessel_4D_3C", n_init=2, n_iter=200, n_repeat=15, method=method)

    for method in METHODs:
        # if method in ['cbo', 'qei']:
        #     continue
        if method in ['ts', 'random']:
            continue
        # experiment(exp='water_converter_32d_neg_3c', n_init=20, n_repeat=10, n_iter=100, method=method, constrain_noise=True)
        experiment(exp="car_cab_7d_8c", n_init=5, n_iter=200, n_repeat=15, method=method)


    for method in METHODs:
    #     # if method in ['cbo', 'qei']:
    #     #     continue
        if method in ['ts', 'random']:
            continue
        # experiment(exp='water_converter_32d_neg_3c', n_init=20, n_repeat=10, n_iter=100, method=method, constrain_noise=True)
        experiment(exp="spring_3d_6c", n_init=10, n_iter=200, n_repeat=15, method=method)
    # experiment(exp='spring_3d_6c', n_init=10, n_repeat=10, n_iter=n_iter, method='scbo')