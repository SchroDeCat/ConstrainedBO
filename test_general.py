'''
Script for general purpose tests
'''
from src.utils import Constrained_Data_Factory
from src.opt import baseline_cbo_m, cbo_multi

EXPS = ['rastrigin_1d', 'rastrigin_10d', 'ackley_20d', 'rosenbrock_5d', 'water_converter_32d', 'gpu_performance_16d']
METHODs = ['cbo', 'scbo', 'cmes-ibo', 'ts', 'qei', 'random']

def experiment(exp:str='rastrigin_1d', method:str='qei', n_repeat:int=2, train_times:int=50, n_iter:int=20, n_init:int=10)->None:
    exp = exp.lower()
    method = method.lower()
    assert exp in EXPS
    assert method in METHODs

    if exp == 'rastrigin_1d': # rastrigin 1D
        cbo_factory = Constrained_Data_Factory(num_pts=2000)
        x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D(scbo_format=False)
        constraint_threshold_list, constraint_confidence_list = cbo_factory.constraint_threshold_list, cbo_factory.constraint_confidence_list
        feasible_filter = cbo_factory.feasible_filter
        cbo_factory.visualize_1d()
    else:
        raise NotImplemented(f"Exp {exp} no implemented")

    if method in ['cmes-ibo', 'ts', 'qei', 'random']:
        print(f"initial reward {y_tensor[:n_init][feasible_filter[:n_init]].squeeze()} while global max {y_tensor[feasible_filter].max().item()}")
        regret = baseline_cbo_m(x_tensor, y_tensor, c_tensor_list, 
                                constraint_threshold_list=constraint_threshold_list, constraint_confidence_list=constraint_confidence_list,
                                n_init=10, n_repeat=n_repeat, train_times=train_times, n_iter=n_iter,
                                regularize=False, low_dim=True,
                                spectrum_norm=False, retrain_interval=1, acq=method, 
                                verbose=True, lr=1e-4, name=f"Basline-{exp}", 
                                return_result=True, retrain_nn=True,
                                plot_result=True, save_result=True, save_path=f'./res/baseline/{method}', 
                                fix_seed=True,  pretrained=False, ae_loc=None, 
                                exact_gp=False, constrain_noise=True,)
        print(f"With constraints, the minimum regret we found is: {regret.min(axis=-1)}")






if __name__ == "__main__":
    experiment()


