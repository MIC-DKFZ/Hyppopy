# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical Image Computing.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import os
import pickle
import numpy as np
from math import pi
from pprint import pprint
import matplotlib.pyplot as plt



from hyppopy.SolverPool import SolverPool
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.BlackboxFunction import BlackboxFunction

OUTPUTDIR = "C:\\Users\\s635r\\Desktop\\solver_comparison"
SOLVER = ["hyperopt", "optunity", "randomsearch", "optuna"]#, "bayesopt"]
ITERATIONS = [25, 100, 250, 500]
STATREPEATS = 10
VFUNC = "5D3"
OVERWRITE = False

OUTPUTDIR = os.path.join(OUTPUTDIR, VFUNC)
if not os.path.isdir(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)


def compute_deviation(solver_name, vfunc_id, iterations, N, fname):
    project = HyppopyProject()
    project.add_hyperparameter(name="axis_00", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_01", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_02", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_03", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_04", domain="uniform", data=[0, 1], dtype="float")

    vfunc = VirtualFunction()
    vfunc.load_default(vfunc_id)
    minima = vfunc.minima()

    def my_loss_function(data, params):
        return vfunc(**params)

    blackbox = BlackboxFunction(data=[], blackbox_func=my_loss_function)

    results = {}
    results["gt"] = []
    for mini in minima:
        results["gt"].append(np.median(mini[0]))

    for iter in iterations:
        results[iter] = {"minima": {}, "loss": None}
        for i in range(vfunc.dims()):
            results[iter]["minima"]["axis_0{}".format(i)] = []

        project.add_settings(section="solver", name="max_iterations", value=iter)
        project.add_settings(section="custom", name="use_solver", value=solver_name)

        solver = SolverPool.get(project=project)
        solver.blackbox = blackbox

        axis_minima = []
        best_losses = []
        for i in range(vfunc.dims()):
            axis_minima.append([])
        for n in range(N):
            print("\rSolver={} iteration={} round={}".format(solver, iter, n), end="")

            solver.run(print_stats=False)
            df, best = solver.get_results()
            best_row = df['losses'].idxmin()
            best_losses.append(df['losses'][best_row])
            for i in range(vfunc.dims()):
                tmp = df['axis_0{}'.format(i)][best_row]
                axis_minima[i].append(tmp)
        for i in range(vfunc.dims()):
            results[iter]["minima"]["axis_0{}".format(i)] = [np.mean(axis_minima[i]), np.std(axis_minima[i])]
        results[iter]["loss"] = [np.mean(best_losses), np.std(best_losses)]

    file = open(fname, 'wb')
    pickle.dump(results, file)
    file.close()


def make_radarplot(results, title, fname=None):
    gt = results.pop("gt")
    categories = list(results[list(results.keys())[0]]["minima"].keys())
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(1, 1, 1, polar=True, )

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='grey', size=8)

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    gt += gt[:1]
    ax.fill(angles, gt, color=(0.2, 0.8, 0.2), alpha=0.2)

    colors = []
    cm = plt.get_cmap('Set1')
    if len(results) > 2:
        indices = list(range(0, len(results) + 1))
        indices.pop(2)
    else:
        indices = list(range(0, len(results)))
    for i in range(len(results)):
        colors.append(cm(indices[i]))

    for iter, data in results.items():
        values = []
        for i in range(len(categories)):
            values.append(data["minima"]["axis_0{}".format(i)][0])
        values += values[:1]
        color = colors.pop(0)
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label="iterations {}".format(iter))

    plt.title(title, size=11, color=(0.1, 0.1, 0.1), y=1.1)
    plt.legend(bbox_to_anchor=(0.08, 1.12))
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname + ".png")
        plt.savefig(fname + ".svg")
    plt.clf()


def make_deviationerrorplot(fnames):
    results = {}
    for fname in fnames:
        file = open(fname, 'rb')
        result = pickle.load(file)
        file.close()
        results[os.path.basename(fname)] = result
    pprint(results)

    plt.figure()
    for iter in results["hyperopt"].keys():
        y = []
        if iter == "gt":
            x = list(range(len(results["hyperopt"][iter])))
            for i in range(len(results["hyperopt"][iter])):
                y.append(results["hyperopt"][iter][i])
            plt.plot(x, y, "--g", label="groundtruth: {}".format(iter))
            continue

        x = list(range(len(results["hyperopt"][iter]["minima"])))
        for i in range(len(results["hyperopt"][iter]["minima"])):
            y.append(results["hyperopt"][iter]["minima"]["axis_0{}".format(i)][0])
        plt.plot(x, y, label="iterations: {}".format(iter))
    plt.title("")
    plt.legend()
    plt.show()



##################################################
############### create datasets ##################
fnames = []
for solver_name in SOLVER:
    fname = os.path.join(OUTPUTDIR, solver_name)
    fnames.append(fname)
    if OVERWRITE or not os.path.isfile(fname):
        compute_deviation(solver_name, VFUNC, ITERATIONS, N=STATREPEATS, fname=fname)
##################################################
##################################################

##################################################
############## create radarplots #################
for solver_name, fname in zip(SOLVER, fnames):
    file = open(fname, 'rb')
    results = pickle.load(file)
    file.close()
    make_radarplot(results, solver_name, fname + "_deviation")
##################################################
##################################################
