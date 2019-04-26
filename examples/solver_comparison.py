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

import os
import sys
import time
import pickle
import numpy as np
from math import pi
import matplotlib.pyplot as plt

from hyppopy.SolverPool import SolverPool
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.BlackboxFunction import BlackboxFunction

#OUTPUTDIR = "C:\\Users\\s635r\\Desktop\\solver_comparison"
OUTPUTDIR = "D:\\Projects\\Python\\hyppopy\\examples\\solver_comparison\\gfx"

SOLVER = []
#SOLVER.append("hyperopt")
#SOLVER.append("optunity")
#SOLVER.append("randomsearch")
#SOLVER.append("optuna")
SOLVER.append("quasirandomsearch")

ITERATIONS = []
ITERATIONS.append(50)
ITERATIONS.append(100)
ITERATIONS.append(250)
ITERATIONS.append(500)

STATREPEATS = 1

OVERWRITE = False


def compute_deviation(solver_name, vfunc_id, iterations, N, fname):
    project = HyppopyProject()
    project.add_hyperparameter(name="axis_00", domain="uniform", data=[0, 1], type=float)
    project.add_hyperparameter(name="axis_01", domain="uniform", data=[0, 1], type=float)
    project.add_hyperparameter(name="axis_02", domain="uniform", data=[0, 1], type=float)
    project.add_hyperparameter(name="axis_03", domain="uniform", data=[0, 1], type=float)
    project.add_hyperparameter(name="axis_04", domain="uniform", data=[0, 1], type=float)

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
        results[iter] = {"minima": {},
                         "distance": {},
                         "duration": None,
                         "set_difference": None,
                         "loss": None,
                         "loss_history": {}}
        for i in range(vfunc.dims()):
            results[iter]["minima"]["axis_0{}".format(i)] = []
            results[iter]["distance"]["axis_0{}".format(i)] = []

        project.add_setting("max_iterations", iter)
        project.add_setting("solver", solver_name)

        solver = SolverPool.get(project=project)
        solver.blackbox = blackbox

        axis_minima = []
        best_losses = []
        best_sets_diff = []
        for i in range(vfunc.dims()):
            axis_minima.append([])

        loss_history = []
        durations = []
        for n in range(N):
            print("\rSolver={} iteration={} round={}".format(solver, iter, n), end="")

            start = time.time()
            solver.run(print_stats=False)
            end = time.time()
            durations.append(end-start)

            df, best = solver.get_results()

            loss_history.append(np.flip(np.sort(df['losses'].values)))
            best_row = df['losses'].idxmin()
            best_losses.append(df['losses'][best_row])
            best_sets_diff.append(abs(df['axis_00'][best_row] - best['axis_00'])+
                                  abs(df['axis_01'][best_row] - best['axis_01'])+
                                  abs(df['axis_02'][best_row] - best['axis_02'])+
                                  abs(df['axis_03'][best_row] - best['axis_03'])+
                                  abs(df['axis_04'][best_row] - best['axis_04']))
            for i in range(vfunc.dims()):
                tmp = df['axis_0{}'.format(i)][best_row]
                axis_minima[i].append(tmp)

        results[iter]["loss_history"] = loss_history
        for i in range(vfunc.dims()):
            results[iter]["minima"]["axis_0{}".format(i)] = [np.mean(axis_minima[i]), np.std(axis_minima[i])]
            dist = np.sqrt((axis_minima[i]-results["gt"][i])**2)
            results[iter]["distance"]["axis_0{}".format(i)] = [np.mean(dist), np.std(dist)]
        results[iter]["loss"] = [np.mean(best_losses), np.std(best_losses)]
        results[iter]["set_difference"] = sum(best_sets_diff)
        results[iter]["duration"] = np.mean(durations)

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
        #plt.savefig(fname + ".svg")
    plt.clf()


def make_errrorbars_plot(results, fname=None):
    n_groups = len(results)

    for iter in ITERATIONS:
        means = []
        stds = []
        names = []
        colors = []
        axis = []
        fig = plt.figure(figsize=(10, 8))
        for solver_name, numbers in results.items():
            names.append(solver_name)
            means.append([])
            stds.append([])

            for axis_name, data in numbers[iter]["distance"].items():
                means[-1].append(data[0])
                stds[-1].append(data[1])
                if len(axis) < 5:
                    axis.append(axis_name)

        for c in range(len(names)):
            colors.append(plt.cm.Set2(c/len(names)))

        index = np.arange(len(axis))
        bar_width = 0.14
        opacity = 0.8
        error_config = {'ecolor': '0.3'}

        for k, name in enumerate(names):
            plt.bar(index + k*bar_width, means[k], bar_width,
                    alpha=opacity,
                    color=colors[k],
                    yerr=stds[k],
                    error_kw=error_config,
                    label=name)
        plt.xlabel('Axis')
        plt.ylabel('Mean [+/- std]')
        plt.title('Deviation per Axis and Solver for {} Iterations'.format(iter))
        plt.xticks(index + 2*bar_width, axis)
        plt.legend()

        if fname is None:
            plt.show()
        else:
            plt.savefig(fname + "_{}.png".format(iter))
            #plt.savefig(fname + "_{}.svg".format(iter))
        plt.clf()


def plot_loss_histories(results, fname=None):
    colors = []
    for c in range(len(SOLVER)):
        colors.append(plt.cm.Set2(c / len(SOLVER)))

    for iter in ITERATIONS:
        fig = plt.figure(figsize=(10, 8))
        added_solver = []
        for n, solver_name in enumerate(results.keys()):
            for history in results[solver_name][iter]["loss_history"]:
                if solver_name not in added_solver:
                    plt.plot(history, color=colors[n], label=solver_name, alpha=0.5)
                    added_solver.append(solver_name)
                else:
                    plt.plot(history, color=colors[n], alpha=0.5)
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Iteration')

        if fname is None:
            plt.show()
        else:
            plt.savefig(fname + "_{}.png".format(iter))
        plt.clf()


def print_durations(results, fname=None):
    colors = []
    for c in range(len(SOLVER)):
        colors.append(plt.cm.Set2(c / len(SOLVER)))

    f = open(fname, "w")
    lines = ["\t".join(SOLVER)+"\n"]

    for iter in ITERATIONS:
        txt = str(iter) + "\t"
        for solver_name in SOLVER:
            duration = results[solver_name][iter]["duration"]
            txt += str(duration) + "\t"
        txt += "\n"
        lines.append(txt)

    f.writelines(lines)
    f.close()


id2dirmapping = {"5D": "data_I", "5D2": "data_II", "5D3": "data_II"}
if __name__ == "__main__":
    vfunc_ID = "5D"
    if len(sys.argv) == 2:
        vfunc_ID = sys.argv[1]
    print("Start Evaluation on {}".format(vfunc_ID))

    OUTPUTDIR = os.path.join(OUTPUTDIR, id2dirmapping[vfunc_ID])
    if not os.path.isdir(OUTPUTDIR):
        os.makedirs(OUTPUTDIR)

    ##################################################
    ############### create datasets ##################
    fnames = []
    for solver_name in SOLVER:
        fname = os.path.join(OUTPUTDIR, solver_name)
        fnames.append(fname)
        if OVERWRITE or not os.path.isfile(fname):
            compute_deviation(solver_name, vfunc_ID, ITERATIONS, N=STATREPEATS, fname=fname)
    ##################################################
    ##################################################

    ##################################################
    ############## create radarplots #################
    all_results = {}
    for solver_name, fname in zip(SOLVER, fnames):
        file = open(fname, 'rb')
        results = pickle.load(file)
        file.close()
        make_radarplot(results, solver_name, fname + "_deviation")
        all_results[solver_name] = results

    fname = os.path.join(OUTPUTDIR, "errorbars")
    make_errrorbars_plot(all_results, fname)

    fname = os.path.join(OUTPUTDIR, "losshistory")
    plot_loss_histories(all_results, fname)

    fname = os.path.join(OUTPUTDIR, "durations.txt")
    print_durations(all_results, fname)

    for solver_name, iterations in all_results.items():
        for iter, numbers in iterations.items():
            if numbers["set_difference"] != 0:
                print("solver {} has a different parameter set match in iteration {}".format(solver_name, iter))

    ##################################################
    ##################################################
