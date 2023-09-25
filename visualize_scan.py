import numpy as np
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CBO_DIR = "./res/scan/"
SCBO_DIR = "./res/scan/"
BASELINE_DIR = "./res/scan/"
IMG_DIR = './res/illustration/'


fig = plt.figure(figsize=[30, 10])
fontsize = 14
n_repeat = 10

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
        ax.set_xlabel("Iteration", fontsize=-4)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))


# ras-1d-1c
for idx, c_portion in enumerate(np.linspace(.1, .9, 5)):
    RES_num = {}
    RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_RASTRIGIN_1D-CP{c_portion:.2%}_noise-InterP-B2.00-FB2.00-RI1--none-ci-R15-P2-T2500_I1_L4-TI1-USexact.npy")
    RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_RASTRIGIN_1D-CP{c_portion:.2%}_noise-InterP-RI1--none-cmes-ibo-R15-P2-T2500_I1_L4-TI1-USexact.npy")
    
    # res/illustration/tmlr_Rastrigin 1D_P10%.png
    img = Image.open(f"{IMG_DIR}tmlr_Rastrigin 1D_P{c_portion:.0%}.png")
    ax = plt.subplot(2, 5, idx+1)
    plt.axis('off')
    ax.imshow(img, aspect='auto')
    ax = plt.subplot(2, 5, idx+6)
    
    visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=2500)
    ax.set_title(f"Rastrigin-1D-1C-{c_portion:.2%}")
    # plt.legend()

handles, labels = ax.get_legend_handles_labels()
# plt.subplots_adjust(hspace=0, wspace=0)


# TODO: split the actual visualization.

# plot results
plt.tight_layout()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.savefig("simple_regret_scan.png")
# plt.show()