# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical and Biological Informatics.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE.txt or http://www.mitk.org for details.
#
# Author: Sven Wanner (s.wanner@dkfz.de)



import os
import sys
import time
import argparse
import tempfile
import numpy as np
import pandas as pd


try:
    import hyppopy as hp
    from hyppopy.globals import ROOT
    from hyppopy.virtualfunction import VirtualFunction
except Exception as e:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import hyppopy as hp
    from hyppopy.globals import ROOT
    from hyppopy.virtualfunction import VirtualFunction

TEMP = tempfile.gettempdir()
DATADIR = os.path.join(os.path.join(ROOT, os.path.join('hyppopy', 'virtualparameterspace')), "6D")

vfunc = VirtualFunction()
vfunc.load_images(DATADIR)
minima = vfunc.minima()
# for i in range(6):
#     mini = minima[i]
#     vfunc.plot(i, title="axis_{} min_x={} min_loss={}".format(str(i).zfill(2), np.mean(mini[0]), mini[1]))


def blackboxfunction(data, params):
    return sum(vfunc(*params.values()))


def getConfig(*args, **kwargs):
    if 'output_dir' in kwargs.keys() and kwargs['output_dir'] is not None:
        output_dir = kwargs['output_dir']
    else:
        output_dir = TEMP
    if 'plugin' in kwargs.keys():
        plugin = kwargs['plugin']
    else:
        plugin = 'hyperopt'

    max_iterations = 0
    if 'max_iterations' in kwargs.keys():
        max_iterations = kwargs['max_iterations']

    if len(args) < 6:
        print("Missing hyperparameter abortion!")
        sys.exit()

    config = {
        "hyperparameter": {
            "axis_0": {
                "domain": "uniform",
                "data": args[0],
                "type": "float"
            },
            "axis_1": {
                "domain": "uniform",
                "data": args[1],
                "type": "float"
            },
            "axis_2": {
                "domain": "uniform",
                "data": args[2],
                "type": "float"
            },
            "axis_3": {
                "domain": "uniform",
                "data": args[3],
                "type": "float"
            },
            "axis_4": {
                "domain": "uniform",
                "data": args[4],
                "type": "float"
            },
            "axis_5": {
                "domain": "uniform",
                "data": args[5],
                "type": "float"
            }
        },
        "settings": {
            "solver_plugin": {
                "max_iterations": max_iterations,
                "use_plugin": plugin,
                "output_dir": output_dir
            }
        }
    }
    return config


def test_randomsearch(output_dir):
    print("#" * 30)
    print("#   RANDOMSEARCH")
    print("#   output_dir={}".format(output_dir))
    print("#" * 30)

    ranges = [[0, 1],
              [0, 800],
              [-1, 1],
              [0, 5],
              [0, 10000],
              [0, 10]]
    args = {'plugin': 'randomsearch', 'output_dir': output_dir}
    config = getConfig(*ranges, **args)
    return config


def test_hyperopt(output_dir):
    print("#" * 30)
    print("#   HYPEROPT")
    print("#   output_dir={}".format(output_dir))
    print("#" * 30)

    ranges = [[0, 1],
              [0, 800],
              [-1, 1],
              [0, 5],
              [0, 10000],
              [0, 10]]
    args = {'plugin': 'hyperopt', 'output_dir': output_dir}
    config = getConfig(*ranges, **args)
    return config


def test_optunity(output_dir):
    print("#" * 30)
    print("#   OPTUNITY")
    print("#   output_dir={}".format(output_dir))
    print("#" * 30)

    ranges = [[0, 1],
              [0, 800],
              [-1, 1],
              [0, 5],
              [0, 10000],
              [0, 10]]
    args = {'plugin': 'optunity', 'output_dir': output_dir}
    config = getConfig(*ranges, **args)
    return config


def analyse_iteration_characteristics(configs):
    N = 50
    num_of_iterations = [5, 10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000]
    results = {'iteration': [],
               'time_overhead': [],
               'time_overhead_std': [],
               'accuracy': [],
               'accuracy_std': [],
               'plugin': []}

    accuracies = {}
    time_overheads = {}
    for plugin in configs.keys():
        accuracies[plugin] = []
        time_overheads[plugin] = []

    for it in num_of_iterations:
        for plugin, config in configs.items():
            print("\riteration loop: {} for plugin {}".format(it, plugin))
            for p, v in accuracies.items():
                v.clear()
            for p, v in time_overheads.items():
                v.clear()
            for n in range(N):
                print("\rrepeat loop: {}".format(n), end="")
                config["settings"]["solver_plugin"]["max_iterations"] = it
                if not hp.ProjectManager.set_config(config):
                    print("Invalid config dict!")
                    sys.exit()

                solver = hp.SolverFactory.get_solver()
                solver.set_loss_function(blackboxfunction)
                solver.set_data(None)

                start = time.process_time()
                solver.run()
                end = time.process_time()
                time_overheads[plugin].append(end-start)
                res, best = solver.get_results()
                best_loss = 0
                for i, p in enumerate(best.items()):
                    best_loss += minima[i][1]
                reached_loss = np.min(res["losses"].values)
                accuracies[plugin].append(100.0/best_loss*reached_loss)

            print("\r")
            results['iteration'].append(it)
            results['time_overhead'].append(np.mean(time_overheads[plugin]))
            results['accuracy'].append(np.mean(accuracies[plugin]))
            results['time_overhead_std'].append(np.std(time_overheads[plugin]))
            results['accuracy_std'].append(np.std(accuracies[plugin]))
            results['plugin'].append(plugin)

    return results


def analyse_random_normal_search(output_dir):
    config = {
        "hyperparameter": {
            "axis_0": {
                "domain": "normal",
                "data": [0.0, 0.2],
                "type": "float"
            },
            "axis_1": {
                "domain": "normal",
                "data": [500, 700],
                "type": "float"
            },
            "axis_2": {
                "domain": "normal",
                "data": [-0.2, 0.9],
                "type": "float"
            },
            "axis_3": {
                "domain": "normal",
                "data": [0.0, 3.0],
                "type": "float"
            },
            "axis_4": {
                "domain": "normal",
                "data": [6000, 10000],
                "type": "float"
            },
            "axis_5": {
                "domain": "normal",
                "data": [3, 7],
                "type": "float"
            }
        },
        "settings": {
            "solver_plugin": {
                "max_iterations": 0,
                "use_plugin": 'randomsearch',
                "output_dir": output_dir
            }
        }
    }

    N = 50
    num_of_iterations = [5, 10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000]

    results = {'iteration': [],
               'time_overhead': [],
               'time_overhead_std': [],
               'accuracy': [],
               'accuracy_std': []}

    accuracies = []
    time_overheads = []
    for it in num_of_iterations:
        config["settings"]["solver_plugin"]["max_iterations"] = it
        print("\riteration loop: {}".format(it))
        accuracies.clear()
        time_overheads.clear()
        for n in range(N):
            print("\rrepeat loop: {}".format(n), end="")
            if not hp.ProjectManager.set_config(config):
                print("Invalid config dict!")
                sys.exit()

            solver = hp.SolverFactory.get_solver()
            solver.set_loss_function(blackboxfunction)
            solver.set_data(None)

            start = time.process_time()
            solver.run()
            end = time.process_time()
            time_overheads.append(end - start)
            res, best = solver.get_results()
            best_loss = 0
            for i, p in enumerate(best.items()):
                best_loss += minima[i][1]
            reached_loss = np.min(res["losses"].values)
            accuracies.append(100.0 / best_loss * reached_loss)

        print("\r")
        results['iteration'].append(it)
        results['time_overhead'].append(np.mean(time_overheads))
        results['accuracy'].append(np.mean(accuracies))
        results['time_overhead_std'].append(np.std(time_overheads))
        results['accuracy_std'].append(np.std(accuracies))

    return results


if __name__ == "__main__":
    print("")
    parser = argparse.ArgumentParser(description='Hyppopy Quality Test Executable')
    parser.add_argument('-o', '--output', type=str, default=None, help='output path to store result')
    parser.add_argument('-p', '--plugin', type=str, default=None, help='if set analysis is only executed on this plugin')
    args = parser.parse_args()

    do_analyse_iteration_characteristics = True
    do_analyse_random_normal_search = False

    funcs = [x for x in locals().keys() if x.startswith("test_")]
    configs = {}
    for f in funcs:
        if args.plugin is not None:
            if not f.endswith(args.plugin):
                continue
        configs[f.split("_")[1]] = locals()[f](args.output)

    if do_analyse_iteration_characteristics:
        start = time.process_time()
        data = analyse_iteration_characteristics(configs)
        end = time.process_time()
        print("Total duration analyse_iteration_characteristics: {} min".format((end-start)/60))
        df = pd.DataFrame.from_dict(data)
        fname = os.path.join(args.output, "analyse_iteration_characteristics.csv")
        df.to_csv(fname, index=False)

    if do_analyse_random_normal_search:
        start = time.process_time()
        data = analyse_random_normal_search(args.output)
        end = time.process_time()
        print("Total duration analyse_random_normal_search: {} min".format((end - start) / 60))
        df = pd.DataFrame.from_dict(data)
        fname = os.path.join(args.output, "analyse_random_normal_search.csv")
        df.to_csv(fname, index=False)
