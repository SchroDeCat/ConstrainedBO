import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CBO_DIR = "./res/"
SCBO_DIR = "./res/"
BASELINE_DIR = "./res/"


fig = plt.figure(figsize=[18, 10])
fontsize = 14

n_repeat = 1
n_iter = 2

def visualize_regret(ax: plt.Axes, RES: dict, fontsize:int=14, n_repeat:int=15, 
                    n_iter:int=100) -> None:
    sqrt_n = np.sqrt(n_repeat)
    init_regret = RES["CBO"][:,0].mean(axis=0)
    for method in RES_num.keys():
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
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_RASTRIGIN_1D-InterP-B2.90-FB1.00-RI1--none-ci-R{n_repeat}-P2-T{n_iter}_I1_L4-TI10-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_RASTRIGIN_1D-InterP-RI1--none-cmes-ibo-R{n_repeat}-P2-T{n_iter}_I1_L4-TI10-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_RASTRIGIN_1D-InterP-RI1--none-qei-R{n_repeat}-P2-T{n_iter}_I1_L4-TI10-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_RASTRIGIN_1D-InterP-RI1--none-scbo-R{n_repeat}-P2-T{n_iter}_I1_L4-TI10-USexact.npy")

ax = plt.subplot(1,3,1)

visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("Rastrigin-1D-1C-60%", fontsize=fontsize)
handles, labels = ax.get_legend_handles_labels()

# ackley-5D
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-B0.10-FB4.00-RI1--none-ci-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-RI1--none-cmes-ibo-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-RI1--none-qei-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_ACKLEY_5D-InterP-RI1--none-scbo-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
ax = plt.subplot(1,3,2)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=100)
ax.set_title("Ackley-5D-2C-14%", fontsize=fontsize)

# Wave-Energy_Converter-36D
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-B20.00-FB20.00-RI1--none-ci-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-RI1--none-cmes-ibo-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-RI1--none-qei-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-RI1--none-scbo-R{n_repeat}-P2-T{n_iter}_I1_L4-TI5-USexact.npy")
ax = plt.subplot(1,3,3)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("Converter-36D-3C-27%", fontsize=fontsize)


# plot results
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=.3)
plt.savefig("simple_regret_iclr.pdf")