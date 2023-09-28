import numpy as np
import tqdm
import torch
from PIL import Image
from src.utils import Constrained_Data_Factory
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CBO_DIR = "./res/scan/"
SCBO_DIR = "./res/scan/"
BASELINE_DIR = "./res/scan/"
IMG_DIR = './res/illustration/'

fig, axes = plt.subplots(2, 5,figsize=[30, 10], gridspec_kw={'height_ratios': [1, 1]})
# fig = plt.figure(figsize=[30, 10])
fontsize = 14
n_repeat = 10

def visualize_1d(c_portion:float, if_norm:bool=False):
        cbo_factory = Constrained_Data_Factory(num_pts=20000 if c_portion is None else 1000)
        if not c_portion is None: # scanning the portion
            x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D_1C(scbo_format=False, c_scan=True, c_portion=c_portion)
        else:
            x_tensor, y_tensor, c_tensor_list = cbo_factory.rastrigin_1D_1C(scbo_format=False)
        plt.title(f"Rastrigin-1D-1C-{c_portion:.2%}", fontsize=fontsize)
        if if_norm:
            base_x = torch.linalg.vector_norm(cbo_factory.x_tensor_range, dim=-1)
        else:
            base_x = cbo_factory.x_tensor_range
        plt.scatter(base_x.squeeze().to(device='cpu').numpy(), cbo_factory.y_tensor.squeeze().to(device='cpu').numpy(), c='black', s=1, label='Objective value (out of feasible region)')
        feasible_x = base_x[cbo_factory.feasible_filter].to(device='cpu')
        feasible_y = cbo_factory.y_tensor[cbo_factory.feasible_filter].to(device='cpu')
        bounds = [feasible_x.min().to(device='cpu'), feasible_x.max().to(device='cpu')]
        plt.scatter(feasible_x.squeeze().to(device='cpu').numpy(), feasible_y.squeeze().to(device='cpu').numpy(), c='purple', s=1, label='Objective value (inside feasible region)')
        plt.scatter(base_x[cbo_factory.max_arg].to(device='cpu').numpy(), cbo_factory.y_tensor[cbo_factory.max_arg].to(device='cpu').numpy(), c='red', s=100, marker='*', label='Optimum' )
        # plt.legend(fontsize=fontsize/1.4)
        plt.xlabel('X')
        plt.ylabel("Y")

def visualize_regret(ax: plt.Axes, RES: dict, fontsize:int=14, n_repeat:int=15, 
                    n_iter:int=100) -> None:
    sqrt_n = np.sqrt(n_repeat)
    init_regret = RES["CBO"][:,0].mean(axis=0)
    for method in RES_num.keys():
        # CI = 1
        CI = 1.96
        coef = CI /sqrt_n
        final_regret = RES[method][:,-1]
        RES[method][:,0] = init_regret
        _RES = RES[method]
        _RES[final_regret == np.inf] = init_regret
        _RES = np.minimum.accumulate(_RES, axis=1)

        _base = np.arange(n_iter)
        _mean, _std = _RES[:,:n_iter].mean(axis=0),  _RES[:,:n_iter].std(axis=0)
        ax.plot(_mean, label=method)
        ax.fill_between(_base, _mean - _std * coef, _mean + _std * coef, alpha=0.3)
        ax.set_xlabel("Iteration", fontsize=fontsize)
        ax.set_ylabel("Simple Regret", fontsize=fontsize)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))


# ras-1d-1c
for idx, c_portion in enumerate(np.linspace(.1, .9, 5)):
    RES_num = {}
    RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_RASTRIGIN_1D-CP{c_portion:.2%}_noise-InterP-B2.00-FB2.00-RI1--none-ci-R15-P2-T2500_I1_L4-TI1-USexact.npy")
    RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_RASTRIGIN_1D-CP{c_portion:.2%}_noise-InterP-RI1--none-cmes-ibo-R15-P2-T2500_I1_L4-TI1-USexact.npy")
    try:
        RES_num["cEI"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_RASTRIGIN_1D-CP{c_portion:.2%}_noise-InterP-RI1--none-qei-R15-P2-T2500_I1_L4-TI1-USexact.npy")
    except:
        pass
    try:
        RES_num["SCBO"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_RASTRIGIN_1D-CP{c_portion:.2%}_noise-InterP-RI1--none-scbo-R15-P2-T2500_I1_L4-TI1-USexact.npy")
    except:
        pass
    # res/illustration/tmlr_Rastrigin 1D_P10%.png
    # img = Image.open(f"{IMG_DIR}tmlr_Rastrigin 1D_P{c_portion:.0%}.png")
    ax = plt.subplot(2, 5, idx+1)
    visualize_1d(c_portion=c_portion)
    # plt.axis('off')
    # ax.imshow(img, aspect='auto')
    if idx == 0:
        handles_1, labels_1 = ax.get_legend_handles_labels()


    ax = plt.subplot(2, 5, idx+6)
    
    visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=2500)
    ax.set_title(f"Rastrigin-1D-1C-{c_portion:.2%} Simple Regret")
    # plt.legend()
    if idx == 0:
        handles_2, labels_2 = ax.get_legend_handles_labels()
    plt.ylim(-1, 15)
# plt.subplots_adjust(hspace=0, wspace=0)


# TODO: split the actual visualization.

# plot results
# plt.tight_layout()
# fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.subplots_adjust(hspace=.4)
fig.legend(handles_1, labels_1, loc='upper center', ncol=len(labels_1))
fig.legend(handles_2, labels_2, loc='center', ncol=len(labels_2))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
# plt.savefig("simple_regret_scan.png")
plt.savefig("simple_regret_scan.pdf")
# plt.show()