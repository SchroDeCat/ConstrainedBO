import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CBO_DIR = "./res/cbo/tmlr/"
SCBO_DIR = "./res/scbo/tmlr/"
BASELINE_DIR = "./res/baseline/tmlr/"


fig = plt.figure(figsize=[18, 10])
fontsize = 14
n_repeat = 10
n_iter=50

def visualize_regret(ax: plt.Axes, RES: dict, fontsize:int=14, n_repeat:int=15, 
                    n_iter:int=100) -> None:
    sqrt_n = np.sqrt(n_repeat)
    init_regret = RES["CBO"][:,0].mean(axis=0)
    for method in RES_num.keys():
        # CI = 1
        CI = 1.96
        coef = CI /sqrt_n
        RES[method][:,0] = init_regret
        RES[method] = np.minimum.accumulate(RES[method], axis=1)

        _base = np.arange(n_iter)
        _mean, _std = RES[method][:,:n_iter].mean(axis=0),  RES[method][:,:n_iter].std(axis=0)
        ax.plot(_mean, label=method)
        ax.fill_between(_base, _mean - _std * coef, _mean + _std * coef, alpha=0.3)
        ax.set_xlabel("Iteration", fontsize=-4)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))


# ras-1d-1c
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_RASTRIGIN_1D-noise_c-InterP-B10.00-FB10.00-RI1--none-ci-R15-P2-T200_I1_L4-TI10-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}cmes-ibo/OL-Regret-Figure_RASTRIGIN_1D-noise_c-InterP-RI1--none-cmes-ibo-R15-P2-T200_I1_L4-TI10-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}qei/OL-Regret-Figure_RASTRIGIN_1D-noise_c-InterP-RI1--none-qei-R15-P2-T200_I1_L4-TI10-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_RASTRIGIN_1D-InterP-RI1--none-scbo-R15-P2-T200_I1_L4-TI10-USexact.npy")

ax = plt.subplot(2,3,1)

visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("Rastrigin-1D-1C")
handles, labels = ax.get_legend_handles_labels()

# ackley-5D
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-B0.10-FB4.00-RI1--none-ci-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}cmes-ibo/OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-RI1--none-cmes-ibo-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}qei/OL-Regret-Figure_ACKLEY_5D-noise_c-InterP-RI1--none-qei-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_ACKLEY_5D-InterP-RI1--none-scbo-R15-P2-T200_I1_L4-TI5-USexact.npy")
ax = plt.subplot(2,3,2)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=100)
ax.set_title("Ackley-5D-2C")

# Wave-Energy_Converter-36D
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-B20.00-FB20.00-RI1--none-ci-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}cmes-ibo/OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-RI1--none-cmes-ibo-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}qei/OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-RI1--none-qei-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_WATER_CONVERTER_32D_NEG_3C-InterP-RI1--none-scbo-R15-P2-T200_I1_L4-TI5-USexact.npy")
ax = plt.subplot(2,3,3)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("Converter-36D-3C")

# vessel 
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_VESSEL_4D_3C-InterP-B10.00-FB10.00-RI1--none-ci-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}cmes-ibo/OL-Regret-Figure_VESSEL_4D_3C-InterP-RI1--none-cmes-ibo-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}qei/OL-Regret-Figure_VESSEL_4D_3C-InterP-RI1--none-qei-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_VESSEL_4D_3C-InterP-RI1--none-scbo-R15-P2-T200_I1_L4-TI5-USexact.npy")
ax = plt.subplot(2,3,4)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("Vessel-4D-3C")


# car cabin
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_CAR_CAB_7D_8C-InterP-B10.00-FB10.00-RI1--none-ci-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}cmes-ibo/OL-Regret-Figure_CAR_CAB_7D_8C-InterP-RI1--none-cmes-ibo-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}qei/OL-Regret-Figure_CAR_CAB_7D_8C-InterP-RI1--none-qei-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}OL-Regret-Figure_CAR_CAB_7D_8C-InterP-RI1--none-scbo-R15-P2-T200_I1_L4-TI5-USexact.npy")
ax = plt.subplot(2,3,5)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("Car_Cabin-7D-8C")

# spring
RES_num = {}
RES_num["CBO"] = np.load(f"{CBO_DIR}OL-Regret-Figure_SPRING_3D_6C-InterP-B10.00-FB10.00-RI1--none-ci-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["CMES-IBO"] = np.load(f"{BASELINE_DIR}cmes-ibo/OL-Regret-Figure_SPRING_3D_6C-InterP-RI1--none-cmes-ibo-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["cEI"] = np.load(f"{BASELINE_DIR}qei/OL-Regret-Figure_SPRING_3D_6C-InterP-RI1--none-qei-R15-P2-T200_I1_L4-TI5-USexact.npy")
RES_num["SCBO"] = np.load(f"{SCBO_DIR}/OL-Regret-Figure_SPRING_3D_6C-InterP-RI1--none-scbo-R15-P2-T200_I1_L4-TI5-USexact.npy")
ax = plt.subplot(2,3,6)
visualize_regret(ax=ax, RES=RES_num, fontsize=fontsize, n_repeat=15, n_iter=200)
ax.set_title("spring-3D-6C")


# plot results
# plt.tight_layout()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.savefig("simple_regret.png")
# plt.show()