import numpy as np
from os import listdir
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CBO_DIR = "./res/beta"
# BETA = [0, 0.1, 2.9, 10]
# BETA = [0, .1, 2, 4, 8]
BETA = [0, .1, 2, 4, 8]

fig = plt.figure(figsize=[18, 10])
fontsize = 14

n_repeat = 15
n_iter = 200


def visualize_regret(
    ax: plt.Axes, RES: dict, fontsize: int = 14, n_repeat: int = 15, n_iter: int = 100
) -> None:
    sqrt_n = np.sqrt(n_repeat)
    # init_regret = RES[f"Beta {BETA[0]}"][:, 0].mean(axis=0)
    init_regret = RES[f"Theoretical Beta"][:, 0].mean(axis=0)
    for method in RES_num.keys():
        CI = 1.96
        coef = CI / sqrt_n
        final_regret = RES[method][:, -1]
        RES[method][:, 0] = init_regret
        _RES = RES[method]
        _RES[final_regret == np.inf] = init_regret
        _RES = np.minimum.accumulate(_RES, axis=1)

        _base = np.arange(n_iter)
        _mean, _std = _RES[:, :n_iter].mean(axis=0), _RES[:, :n_iter].std(axis=0)
        ax.plot(_mean, label=method)
        ax.fill_between(_base, _mean - _std * coef, _mean + _std * coef, alpha=0.3)
        ax.set_xlabel("Iteration", fontsize=fontsize)
        ax.set_ylabel("Simple Regret", fontsize=fontsize)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))


# ras-1d-1c
RES_num = {}
for beta in BETA:
    for _file in listdir(f"{CBO_DIR}"):
        _pre = f"OL-Regret-Figure_RASTRIGIN_1D-InterP-B{beta:.2f}"
        _ext = 'npy'
        if _file.startswith(_pre) and _file.endswith(_ext) and ('B0.00-FB1.00' not in _file):
            _key = f"Beta {beta}" if beta > 0 else f"Theoretical Beta"
            RES_num[_key] = np.load(
                f"{CBO_DIR}/{_file}"
            )

ax = plt.subplot(1, 3, 1)

visualize_regret(
    ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=n_repeat, n_iter=n_iter
)
ax.set_title("Rastrigin-1D-1C-60%", fontsize=fontsize)
handles, labels = ax.get_legend_handles_labels()

# ackley-5D
RES_num = {}
for beta in BETA:
    for _file in listdir(f"{CBO_DIR}"):
        _pre = f"OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-B{beta:.2f}"
        _ext = 'npy'
        if _file.startswith(_pre) and _file.endswith(_ext) and ('B0.00-FB4.00' not in _file):
            _key = f"Beta {beta}" if beta > 0 else f"Theoretical Beta"
            RES_num[_key] = np.load(
                f"{CBO_DIR}/{_file}"
            )

ax = plt.subplot(1, 3, 2)
visualize_regret(
    ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=n_repeat, n_iter=100,
)
ax.set_title("Ackley-5D-2C-14%", fontsize=fontsize)

# Wave-Energy_Converter-36D
RES_num = {}
for beta in BETA:
    for _file in listdir(f"{CBO_DIR}"):
        _pre = f"OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-B{beta:.2f}"
        _ext = 'npy'
        if _file.startswith(_pre) and _file.endswith(_ext) and ('B0.00-FB20.00' not in _file):
            _key = f"Beta {beta}" if beta > 0 else f"Theoretical Beta"
            RES_num[_key] = np.load(
                f"{CBO_DIR}/{_file}"
            )

ax = plt.subplot(1, 3, 3)
visualize_regret(
    ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=n_repeat, n_iter=n_iter
)
ax.set_title("Converter-36D-3C-27%", fontsize=fontsize)


# plot results
fig.legend(handles, labels, loc="upper center", ncol=len(labels))
plt.subplots_adjust(
    left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3
)
plt.savefig("simple_regret_fbeta_iclr.pdf")
