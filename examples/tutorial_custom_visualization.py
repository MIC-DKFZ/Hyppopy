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

import matplotlib.pylab as plt

from hyppopy.SolverPool import SolverPool
from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.BlackboxFunction import BlackboxFunction

project = HyppopyProject()
project.add_hyperparameter(name="axis_00", domain="uniform", data=[0, 1], type=float)
project.add_hyperparameter(name="axis_01", domain="uniform", data=[0, 1], type=float)
project.add_hyperparameter(name="axis_02", domain="uniform", data=[0, 1], type=float)
project.add_hyperparameter(name="axis_03", domain="uniform", data=[0, 1], type=float)
project.add_hyperparameter(name="axis_04", domain="uniform", data=[0, 1], type=float)
project.add_setting("max_iterations", 500)
project.add_setting("solver", "randomsearch")

plt.ion()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey=True)
plot_data = {"iterations": [],
             "loss": [],
             "axis_00": [],
             "axis_01": [],
             "axis_02": [],
             "axis_03": [],
             "axis_04": []}


def my_visualization_function(**kwargs):
    print("\r{}".format(kwargs), end="")
    plot_data["iterations"].append(kwargs['iterations'])
    plot_data["loss"].append(kwargs['loss'])
    plot_data["axis_00"].append(kwargs['axis_00'])
    plot_data["axis_01"].append(kwargs['axis_01'])
    plot_data["axis_02"].append(kwargs['axis_02'])
    plot_data["axis_03"].append(kwargs['axis_03'])
    plot_data["axis_04"].append(kwargs['axis_04'])

    axes[0, 0].clear()
    axes[0, 0].scatter(plot_data["axis_00"], plot_data["loss"], c=plot_data["loss"], cmap="jet", marker='.')
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_xlabel("axis_00")

    axes[0, 1].clear()
    axes[0, 1].scatter(plot_data["axis_01"], plot_data["loss"], c=plot_data["loss"], cmap="jet", marker='.')
    axes[0, 1].set_xlabel("axis_01")

    axes[0, 2].clear()
    axes[0, 2].scatter(plot_data["axis_02"], plot_data["loss"], c=plot_data["loss"], cmap="jet", marker='.')
    axes[0, 2].set_xlabel("axis_02")

    axes[1, 0].clear()
    axes[1, 0].scatter(plot_data["axis_03"], plot_data["loss"], c=plot_data["loss"], cmap="jet", marker='.')
    axes[1, 0].set_ylabel("loss")
    axes[1, 0].set_xlabel("axis_03")

    axes[1, 1].clear()
    axes[1, 1].scatter(plot_data["axis_04"], plot_data["loss"], c=plot_data["loss"], cmap="jet", marker='.')
    axes[1, 1].set_xlabel("axis_04")

    axes[1, 2].clear()
    axes[1, 2].plot(plot_data["iterations"], plot_data["loss"], "--", c=(0.8, 0.8, 0.8, 0.5))
    axes[1, 2].scatter(plot_data["iterations"], plot_data["loss"], marker='.', c=(0.2, 0.2, 0.2))
    axes[1, 2].set_xlabel("iterations")

    plt.draw()
    plt.tight_layout()
    plt.pause(0.001)


def my_loss_function(data, params):
    vfunc = VirtualFunction()
    vfunc.load_default("5D")
    return vfunc(**params)


blackbox = BlackboxFunction(data=[],
                            blackbox_func=my_loss_function,
                            callback_func=my_visualization_function)

solver = SolverPool.get(project=project)
solver.blackbox = blackbox
solver.run()
df, best = solver.get_results()

print("\n")
print("*" * 100)
print("Best Parameter Set:\n{}".format(best))
print("*" * 100)
print("")
save_plot = input("Save Plot? [y/n] ")
if save_plot == "y":
    plt.savefig('plot_{}.png'.format(project.custom_use_solver))
