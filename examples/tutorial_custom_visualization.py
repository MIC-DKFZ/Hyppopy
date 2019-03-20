import numpy as np
import matplotlib.pylab as plt

from hyppopy.HyppopyProject import HyppopyProject
from hyppopy.VirtualFunction import VirtualFunction
from hyppopy.BlackboxFunction import BlackboxFunction
from hyppopy.solver.HyperoptSolver import HyperoptSolver
from hyppopy.solver.OptunitySolver import OptunitySolver
from hyppopy.solver.RandomsearchSolver import RandomsearchSolver

project = HyppopyProject()
project.add_hyperparameter(name="axis_00", domain="uniform", data=[0, 1], dtype="float")
project.add_hyperparameter(name="axis_01", domain="uniform", data=[0, 800], dtype="float")
project.add_hyperparameter(name="axis_02", domain="uniform", data=[0, 5], dtype="float")
project.add_hyperparameter(name="axis_03", domain="uniform", data=[1, 10000], dtype="float")
project.add_hyperparameter(name="axis_04", domain="uniform", data=[0, 10], dtype="float")
project.add_settings(section="solver", name="max_iterations", value=500)
project.add_settings(section="custom", name="use_solver", value="hyperopt")

plt.ion()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True)
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
    axes[0, 0].plot(plot_data["loss"], plot_data["axis_00"], ".r")
    axes[0, 0].set_ylabel("axis_00")

    axes[0, 1].clear()
    axes[0, 1].plot(plot_data["loss"], plot_data["axis_01"], ".r")
    axes[0, 1].set_ylabel("axis_01")

    axes[0, 2].clear()
    axes[0, 2].plot(plot_data["loss"], plot_data["axis_02"], ".r")
    axes[0, 2].set_ylabel("axis_02")

    axes[1, 0].clear()
    axes[1, 0].plot(plot_data["loss"], plot_data["axis_03"], ".r")
    axes[1, 0].set_xlabel("loss")
    axes[1, 0].set_ylabel("axis_03")

    axes[1, 1].clear()
    axes[1, 1].plot(plot_data["loss"], plot_data["axis_04"], ".r")
    axes[1, 1].set_xlabel("loss")
    axes[1, 1].set_ylabel("axis_04")

    axes[1, 2].clear()
    axes[1, 2].plot(plot_data["loss"], plot_data["iterations"], ".r")
    axes[1, 2].set_xlabel("loss")
    axes[1, 2].set_ylabel("iterations")

    plt.draw()
    plt.tight_layout()
    plt.pause(0.001)


def my_loss_function(data, params):
    vfunc = VirtualFunction()
    vfunc.load_default(5)
    return vfunc(**params)


blackbox = BlackboxFunction(data=[],
                            blackbox_func=my_loss_function,
                            callback_func=my_visualization_function)

if project.custom_use_solver == "hyperopt":
    solver = HyperoptSolver(project)
elif project.custom_use_solver == "optunity":
    solver = OptunitySolver(project)
elif project.custom_use_solver == "randomsearch":
    solver = RandomsearchSolver(project)

if solver is not None:
    solver.blackbox = blackbox
solver.run()
df, best = solver.get_results()

print("\n")
print("*" * 100)
print("Best Parameter Set:\n{}".format(best))
print("*" * 100)