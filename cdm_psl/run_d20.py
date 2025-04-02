import argparse
import numpy as np
import torch
import pickle
import sys
from problem import get

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymop.factory import get_problem
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from evolution.utils import *
from learning.model_init import *
from learning.model_update import *
from learning.prediction import *
from diffusion import gen_offspring
from utils import sbx, environment_selection, pm_mutation, sort_population

# Set up command line argument parsing
parser = argparse.ArgumentParser(
    description="Run experiments with specified problem set."
)
parser.add_argument(
    "--prob",
    type=str,
    default="dtlz2",
    choices=[
        "zdt1",
        "zdt2",
        "zdt3",
        "dtlz2",
        "dtlz3",
        "dtlz4",
        "dtlz5",
        "dtlz6",
        "dtlz7",
        "re1",
        "re2",
        "re3",
        "re4",
        "re5",
        "re6",
        "re7",
    ],
    help="Specify the problem set to run.",
)
# Parse command line arguments
args = parser.parse_args()
# Set problem set
ins_list = [args.prob]

# number of independent runs
n_run = 5
# number of initialized solutions
n_init = 100
# number of iterations, and batch size per iteration
n_iter = 20
n_sample = 5

# PSL parameters
# number of learning steps
n_steps = 1000
# number of sampled preferences per step
n_pref_update = 10
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000
# number of optional local search
n_local = 1
# device
device = "cuda"
# -----------------------------------------------------------------------------

hv_list = {}

# Ref point D=20
dic = {
    "zdt1": [0.9994, 6.0576],
    "zdt2": [0.9994, 6.8960],
    "zdt3": [0.9994, 6.0571],
    "dtlz2": [2.8390, 2.9011, 2.8575],
    "dtlz3": [2421.6427, 1905.2767, 2532.9691],
    "dtlz4": [3.2675, 2.6443, 2.4263],
    "dtlz5": [2.6672, 2.8009, 2.8575],
    "dtlz6": [16.8258, 16.9194, 17.7646],
    "dtlz7": [0.9984, 0.9961, 22.8114],
    "re1": [2.76322289e03, 3.68876972e-02],
    "re2": [528107.18990952, 1279320.81067113],
    "re3": [7.68527849, 7.28609807, 21.50103909],
    "re4": [6.79211111, 60.0, 0.4799612],
    "re5": [0.87449713, 1.05091656, 1.05328528],
    "re6": [749.92405125, 2229.37483405],
    "re7": [2.10336300e02, 1.06991599e03, 3.91967702e07],
}

# --------------------------------------------------------------------------------------------------

# Update historical data and calculate reference point
def get_ref_point_method2(Y, history_Y, k, scale_factor=1.1, quantile=0.95):
    history_Y.append(Y)
    if len(history_Y) > k:
        history_Y.pop(0)
    all_Y = np.vstack(history_Y)  # Combine historical data
    nds = NonDominatedSorting()
    idx_nds = nds.do(all_Y)
    Y_nds = all_Y[idx_nds[0]]  # Get non-dominated individuals
    quantile_values = np.quantile(Y_nds, quantile, axis=0)
    return scale_factor * quantile_values


for test_ins in ins_list:
    print(test_ins)

    # Get problem info
    hv_all_value = np.zeros([n_run, n_iter + 2])
    if test_ins.startswith("zdt"):
        problem = get_problem(test_ins, n_var=20)
    elif test_ins.startswith("dtlz"):
        problem = get_problem(test_ins, n_var=20, n_obj=3)
    else:
        problem = get(test_ins)
    n_dim = problem.n_var
    n_obj = problem.n_obj
    lbound = torch.zeros(n_dim).float()
    ubound = torch.ones(n_dim).float()
    ref_point = dic[test_ins]
    i_iter = 0

    # Repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(x_init)
        p_rel_map, s_rel_map = init_dom_rel_map(300)

        p_model = init_dom_nn_classifier(
            x_init, y_init, p_rel_map, pareto_dominance, problem
        )  # init Pareto-Net
        evaluated = len(y_init)

        X = x_init
        Y = y_init

        hv = HV(ref_point=np.array(ref_point))
        hv_value = hv(Y)
        hv_all_value[run_iter, i_iter] = hv_value
        i_iter = i_iter + 1

        print("[New independent run]")
        hv_text = "hv" + " {:.4e}".format(np.mean(hv_value))
        print(hv_text)
        print("***")

        z = torch.zeros(n_obj).to(device)

        # Counter for tracking iterations since last switch
        iteration_since_switch = 0

        # Parameters for switching methods
        hv_change_threshold = 0.05  # Threshold for HV value change
        hv_history_length = 3  # Number of recent iterations to consider
        hv_history = []  # Store recent HV values
        use_diffusion = True

        # Initialize list to store historical data
        history_Y = []

        while evaluated < 200:
            transformation = StandardTransform([0, 1])
            transformation.fit(X, Y)
            X_norm, Y_norm = transformation.do(X, Y)
            _, index = environment_selection(Y, len(X) // 3)
            real = X[index, :]
            label = np.zeros((len(Y), 1))
            label[index, :] = 1
            surrogate_model = GaussianProcess(n_dim, n_obj, nu=5)
            surrogate_model.fit(X_norm, Y_norm)

            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)

            Y_nds = Y_norm[idx_nds[0]]
            PopDec = real
            PopDec_dom_labels, PopDec_cfs = nn_predict_dom_intra(
                PopDec, p_model, device
            )
            sorted_pop = sort_population(PopDec, PopDec_dom_labels, PopDec_cfs)
            number_of_dv = sorted_pop.shape[1]

            if use_diffusion:
                # Generate offspring using Diffusion
                X_psl = gen_offspring(
                    sorted_pop, number_of_dv, surrogate_model, [lbound, ubound]
                )
            else:
                # Generate offspring using SBX
                rows_to_take = int(1 / 3 * sorted_pop.shape[0])
                offspringA = sorted_pop[:rows_to_take, :]

                if len(offspringA) % 2 == 1:
                    offspringA = offspringA[:-1]
                new_pop = np.empty((0, n_dim))
                for _ in range(1000):
                    result = sbx(offspringA, eta=15)
                    new_pop = np.vstack((new_pop, result))

                X_psl = new_pop

            # Mutate the new offspring
            X_psl = pm_mutation(X_psl, [lbound, ubound])

            Y_candidate_mean = surrogate_model.evaluate(X_psl)["F"]
            Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)["S"]

            rows_with_nan = np.any(np.isnan(Y_candidate_mean), axis=1)
            Y_candidate_mean = Y_candidate_mean[~rows_with_nan]
            Y_candidata_std = Y_candidata_std[~rows_with_nan]
            X_psl = X_psl[~rows_with_nan]

            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            Y_candidate_mean = Y_candidate

            best_subset_list = []
            Y_p = Y_nds
            for b in range(n_sample):
                hv = HV(
                    ref_point=np.max(np.vstack([Y_p, Y_candidate_mean]), axis=0)
                )
                best_hv_value = 0
                best_subset = None

                for k in range(len(Y_candidate_mean)):
                    Y_subset = Y_candidate_mean[k]
                    Y_comb = np.vstack([Y_p, Y_subset])
                    hv_value_subset = hv(Y_comb)
                    if hv_value_subset > best_hv_value:
                        best_hv_value = hv_value_subset
                        best_subset = [k]

                Y_p = np.vstack([Y_p, Y_candidate_mean[best_subset]])
                best_subset_list.append(best_subset)
            best_subset_list = np.array(best_subset_list).T[0]

            X_candidate = X_psl
            X_new = X_candidate[best_subset_list]

            Y_new = problem.evaluate(X_new)
            Y_new = torch.tensor(Y_new).to(device)

            X_new = torch.tensor(X_new).to(device)
            X = np.vstack([X, X_new.detach().cpu().numpy()])
            Y = np.vstack([Y, Y_new.detach().cpu().numpy()])

            rows_with_nan = np.any(np.isnan(Y), axis=1)
            X = X[~rows_with_nan, :]
            Y = Y[~rows_with_nan, :]

            update_dom_nn_classifier(
                p_model, X, Y, p_rel_map, pareto_dominance, problem
            )

            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, i_iter] = hv_value
            i_iter = i_iter + 1

            # Print current operator
            if use_diffusion == True:
                operator_text = "operator: Diffusion"
            else:
                operator_text = "operator: SBX"
            print(operator_text)
            hv_text = "hv" + " {:.4e}".format(np.mean(hv_value))
            print(hv_text)
            evaluated = evaluated + n_sample

            # Calculate approximate HV
            ref_point_method2 = get_ref_point_method2(Y, history_Y, k=3)
            hv_method2 = HV(ref_point=ref_point_method2)
            hv_value_method2 = hv_method2(Y)

            # Update HV value history
            hv_history.append(np.mean(hv_value_method2))
            if len(hv_history) > hv_history_length:
                hv_history.pop(0)

            if len(hv_history) == hv_history_length:
                avg_hv = sum(hv_history[:-1]) / (hv_history_length - 1)
                hv_change = abs((hv_history[-1] - avg_hv) / avg_hv)

                # Determine if method needs to be switched
                if iteration_since_switch >= 2:
                    if hv_change < hv_change_threshold:
                        use_diffusion = not use_diffusion
                        iteration_since_switch = 0  # Reset counter
                else:
                    iteration_since_switch += 1  # If already switched, increment counter
            print("***")

        hv_list[test_ins] = hv_all_value
        print("************************************************************")
        i_iter = 0
