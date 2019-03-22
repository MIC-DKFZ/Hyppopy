import os
import pickle
import numpy as np
from math import pi
import pandas as pd
import matplotlib.pyplot as plt



from hyppopy.SolverPool import SolverPool
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.BlackboxFunction import BlackboxFunction

def make_spider(results, row, title, groundtruth):
    categories = ["axis_00", "axis_01", "axis_02", "axis_03", "axis_04"]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(2, 2, row+1, polar=True, )

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='grey', size=8)

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    gt = []
    for i in range(5):
        gt.append(groundtruth[i])
    gt += gt[:1]
    ax.fill(angles, gt, color=(0.2, 0.8, 0.2), alpha=0.2)

    colors = [(0.8, 0.8, 0.0, 0.8), (0.7, 0.2, 0.2, 0.8), (0.2, 0.2, 0.7, 0.8)]
    for iter, data in results["iteration"].items():
        values = []
        for i in range(5):
            values.append(data["axis_0{}".format(i)][row])
        values += values[:1]
        ax.plot(angles, values, color=colors.pop(0), linewidth=2, linestyle='solid', label="iterations {}".format(iter))

    ax.plot(angles, gt, color=(0.2, 0.8, 0.2, 0.8), linewidth=2, linestyle='solid', label="groundtruth")
    plt.title(title, size=11, color=(0.1, 0.1, 0.1), y=1.1)
    plt.legend(bbox_to_anchor=(0.2, 1.2))




for vfunc_id in ["5D3"]:
    OUTPUTDIR = "C:\\Users\\s635r\\Desktop\\solver_comparison"
    EXPERIMENT = {"iterations": [100, 200, 300],
                  "solver": ["randomsearch", "hyperopt", "optunity"],
                  "repeat": 1,
                  "output_dir": os.path.join(OUTPUTDIR, vfunc_id)}

    if not os.path.isdir(EXPERIMENT["output_dir"]):
        os.makedirs(EXPERIMENT["output_dir"])

    project = HyppopyProject()
    project.add_hyperparameter(name="axis_00", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_01", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_02", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_03", domain="uniform", data=[0, 1], dtype="float")
    project.add_hyperparameter(name="axis_04", domain="uniform", data=[0, 1], dtype="float")
    project.add_settings(section="solver", name="max_iterations", value=100)
    project.add_settings(section="custom", name="use_solver", value="randomsearch")

    if os.path.isfile(os.path.join(EXPERIMENT["output_dir"], "results")):
        file = open(os.path.join(EXPERIMENT["output_dir"], "results"), 'rb')
        results = pickle.load(file)
        file.close()
    else:
        vfunc = VirtualFunction()
        vfunc.load_default(vfunc_id)
        for i in range(5):
            vfunc.plot(i)


        def my_loss_function(data, params):
            return vfunc(**params)


        results = {"group": EXPERIMENT["solver"],
                   "groundtruth": [],
                   'iteration': {}}

        minima = vfunc.minima()
        for mini in minima:
            results["groundtruth"].append(np.median(mini[0]))


        for iter in EXPERIMENT["iterations"]:
            results["iteration"][iter] = {"axis_00": [],
                                          "axis_01": [],
                                          "axis_02": [],
                                          "axis_03": [],
                                          "axis_04": []}
            for solver_name in EXPERIMENT["solver"]:
                axis_minima = [0, 0, 0, 0, 0]
                for n in range(EXPERIMENT["repeat"]):
                    print("\rSolver={} iteration={} round={}".format(solver_name, iter, n), end="")
                    project.add_settings(section="solver", name="max_iterations", value=iter)
                    project.add_settings(section="custom", name="use_solver", value=solver_name)

                    blackbox = BlackboxFunction(data=[], blackbox_func=my_loss_function)

                    solver = SolverPool.get(project=project)
                    solver.blackbox = blackbox
                    solver.run(print_stats=False)
                    df, best = solver.get_results()

                    best_row = df['losses'].idxmin()
                    best_loss = df['losses'][best_row]
                    for i in range(5):
                        axis_minima[i] += df['axis_0{}'.format(i)][best_row]/EXPERIMENT["repeat"]
                for i in range(5):
                    results["iteration"][iter]["axis_0{}".format(i)].append(axis_minima[i])
            print("")
        print("\n\n")

        file = open(os.path.join(EXPERIMENT["output_dir"], "results"), 'wb')
        pickle.dump(results, file)
        file.close()

    my_dpi = 96
    plt.figure(figsize=(1100/my_dpi, 1100/my_dpi), dpi=my_dpi)
    for row in range(3):
        make_spider(results, row=row, title=results['group'][row], groundtruth=results["groundtruth"])
    #plt.show()
    plt.savefig(os.path.join(EXPERIMENT["output_dir"], "radar_plots.svg"))
    plt.savefig(os.path.join(EXPERIMENT["output_dir"], "radar_plots.png"))
