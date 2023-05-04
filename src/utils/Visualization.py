import math
import torch
import time
import warnings
import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import mlab
from matplotlib import gridspec
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.exceptions import BadInitialCandidatesWarning
from IPython.display import Image

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 1
N_BATCH = 50
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512

# Task-specific bounds
bounds = torch.tensor([[0.0] * 2, [6.0] * 2])

verbose = True


def target(x1, x2):
    """
    Target function
    """
    a = np.exp(-((x1 - 2) ** 2 / 0.7 + (x2 - 4) ** 2 / 1.2) + (x1 - 2) * (x2 - 4) / 1.6)
    b = np.exp(-((x1 - 4) ** 2 / 3 + (x2 - 2) ** 2 / 2.))
    c = np.exp(-((x1 - 4) ** 2 / 0.5 + (x2 - 4) ** 2 / 0.5) + (x1 - 4) * (x2 - 4) / 0.5)
    d = np.sin(3.1415 * x1)
    e = np.exp(-((x1 - 5.5) ** 2 / 0.5 + (x2 - 5.5) ** 2 / .5))
    return 2 * a + b - c + 0.17 * d + 2 * e


def generate_initial_data(n=10):
    """
    generate training data
    """
    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, 2)
    train_Y = torch.tensor(target(train_X[:, 0].numpy(), train_X[:, 1].numpy())).unsqueeze(-1)
    best_observed_value = train_Y.max().item()
    return train_X, train_Y, best_observed_value


def initialize_model(train_X, train_Y):
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES
    )
    # observe new values
    exact_X = candidates.detach()
    exact_Y = torch.tensor(target(exact_X[:, 0].numpy(), exact_X[:, 1].numpy())).unsqueeze(-1)

    return exact_X, exact_Y, acq_value


def update_random_observations(best_random):
    """
    Simulates a random policy by taking the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(BATCH_SIZE, 2)
    next_random_best = torch.tensor(target(rand_X[:, 0].numpy(), rand_X[:, 1].numpy())).unsqueeze(-1).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random


def plot_2d(name=None):
    """
    Plots a 2x2 graph given BO parameters
    """
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    gridsize = 150

    fig.suptitle('Bayesian Optimization in Action', fontdict={'size': 30})

    # Hexbin scales varies from case to case
    ax[0][0].set_title('Gaussian Process Predicted Mean', fontdict={'size': 15})
    im00 = ax[0][0].hexbin(x, y, C=mu, mincnt=1, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=-0.9, vmax=2.1)
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][0].plot(train_x_ei[:, 1].numpy(), train_x_ei[:, 0].numpy(), 'D', markersize=4, color='k',
                  label='Observations')

    ax[0][1].set_title('Target Function', fontdict={'size': 15})
    im10 = ax[0][1].hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=-0.9, vmax=2.1)
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][1].plot(train_x_ei[:, 1].numpy(), train_x_ei[:, 0].numpy(), 'D', markersize=4, color='k')

    ax[1][0].set_title('Gaussian Process Variance', fontdict={'size': 15})
    im01 = ax[1][0].hexbin(x, y, C=var, mincnt=1, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])

    ax[1][1].set_title('Acquisition Function', fontdict={'size': 15})
    im11 = ax[1][1].hexbin(x, y, C=ei_val, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0.00, vmax=0.025)

    # Draw two lines indicating the coordinates on the acquisition graph
    # Scale down to the boundary
    ax[1][1].plot([np.where(ei_val.reshape((300, 300)) == ei_val.max())[1] / 50.,
                   np.where(ei_val.reshape((300, 300)) == ei_val.max())[1] / 50.],
                  [0, 6],
                  'k-', lw=2)

    ax[1][1].plot([0, 6],
                  [np.where(ei_val.reshape((300, 300)) == ei_val.max())[0] / 50.,
                   np.where(ei_val.reshape((300, 300)) == ei_val.max())[0] / 50.],
                  'k-', lw=2)

    ax[1][1].axis([x.min(), x.max(), y.min(), y.max()])

    for im, axis in zip([im00, im10, im01, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        cb.set_label('Value')

    if name is None:
        name = '_'

    plt.tight_layout()

    # Save or show figure
    fig.savefig('bo_eg_' + name + '.png')
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':

    x = y = np.linspace(0, 6, 300)
    X, Y = np.meshgrid(x, y)
    x = X.ravel()
    y = Y.ravel()
    X = np.vstack([x, y]).T[:, [1, 0]]
    z = target(y, x)

    best_observed_all_ei, best_observed_all_mes, best_random_all = [], [], []

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        best_observed_ei, best_observed_mes, best_random = [], [], []

        # call helper functions to generate initial training data and initialize model
        train_x_ei, train_obj_ei, best_observed_value_ei = generate_initial_data(n=5)
        mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)

        train_x_mes, train_obj_mes = train_x_ei, train_obj_ei
        best_observed_value_mes = best_observed_value_ei
        mll_mes, model_mes = initialize_model(train_x_mes, train_obj_mes)

        mu = model_ei.posterior(torch.from_numpy(X).float()).mean.detach().numpy()
        var = model_ei.posterior(torch.from_numpy(X).float()).variance.detach().numpy()
        ei_val = np.ones(90000)

        i = 0
        plot_2d(name=str(trial) + '_' + str(i).zfill(2))
        i += 1

        best_observed_ei.append(best_observed_value_ei)
        best_observed_mes.append(best_observed_value_mes)
        best_random.append(best_observed_value_ei)

        candidate_set = torch.rand(1000, bounds.size(1))
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll_ei)
            fit_gpytorch_model(mll_mes)

            EI = ExpectedImprovement(
                model=model_ei,
                best_f=train_obj_ei.max(),
                maximize=True
            )

            qMES = qMaxValueEntropy(
                model=model_mes,
                candidate_set=candidate_set,
                use_gumbel=True,
                num_fantasies=20
            )

            # optimize and get new observation
            new_x_ei, new_obj_ei, acq_val_ei = optimize_acqf_and_get_observation(EI)
            new_x_mes, new_obj_mes, acq_val_mes = optimize_acqf_and_get_observation(qMES)

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

            train_x_mes = torch.cat([train_x_mes, new_x_mes])
            train_obj_mes = torch.cat([train_obj_mes, new_obj_mes])

            # update progress
            best_random = update_random_observations(best_random)
            best_value_ei = train_obj_ei.max().item()
            best_value_mes = train_obj_mes.max().item()
            best_observed_ei.append(best_value_ei)
            best_observed_mes.append(best_value_mes)

            # reinitialize the models so they are ready for fitting on next iteration
            mll_ei, model_ei = initialize_model(
                train_x_ei,
                train_obj_ei
            )
            mll_mes, model_mes = initialize_model(
                train_x_mes,
                train_obj_mes
            )

            mu = model_ei.posterior(torch.from_numpy(X).float()).mean.detach().numpy()
            var = model_ei.posterior(torch.from_numpy(X).float()).variance.detach().numpy()
            ei_val = EI(torch.from_numpy(X).float().reshape(90000, 1, 2)).detach().numpy()

            plot_2d(name=str(trial) + '_' + str(i).zfill(2))
            i += 1

            t1 = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, EI, MES) = "
                    f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_mes:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")

        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_mes.append(best_observed_mes)
        best_random_all.append(best_random)

        # Generate gif file
        png_dir = '/Users/zejiezhu/PycharmProjects/pythonProject'
        images = []
        gif_file = 'demo.gif'
        for file_name in sorted(os.listdir(png_dir)):
            if file_name.endswith('.png'):
                file_path = os.path.join(png_dir, file_name)
                images.append(iio.imread(file_path))
        iio.mimwrite(gif_file, images, format='.gif', fps=2)

        # Display gif file
        Image("/Users/zejiezhu/PycharmProjects/pythonProject/demo.gif")
