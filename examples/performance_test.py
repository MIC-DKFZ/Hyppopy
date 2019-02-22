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
import shutil
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.svc_usecase.svc_usecase import svc_usecase
from hyppopy.workflows.knc_usecase.knc_usecase import knc_usecase
from hyppopy.workflows.adaboost_usecase.adaboost_usecase import adaboost_usecase
from hyppopy.workflows.randomforest_usecase.randomforest_usecase import randomforest_usecase
from hyppopy.workflows.gradientboost_usecase.gradientboost_usecase import gradientboost_usecase

sns.set(style="ticks")
sns.set(style="darkgrid")


class PerformanceTest(object):

    def __init__(self, root=None):
        if root is None:
            self.root = os.path.join(tempfile.gettempdir(), 'test_data')
        else:
            self.root = os.path.join(root, 'test_data')
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.test = None
        self.train = None
        self.config = None
        self.iter_sequence = [5, 10, 25, 50, 100, 150, 300, 500, 800, 1200]
        self.iter_sequence = [25]

    def run(self):
        self.set_up()
        #self.run_svc_usecase()
        #self.run_gradientboost_usecase()
        self.run_randomforest_usecase()
        #self.run_adaboost_usecase()
        #self.run_knc_usecase()
        #self.clean_up()

    def set_hyperparameter(self, params):
        self.config["hyperparameter"] = params

    def set_iterations(self, value):
        self.config["settings"]["solver_plugin"]["max_iterations"] = value

    def find_loss_and_time(self, solver_output, results):
        min_idx = solver_output["losses"].idxmin()
        results["losses"].append(solver_output["losses"][min_idx])
        results["duration"].append(solver_output["duration"][min_idx])

    def plot_matrix(self, df, name=""):
        sns_plot = sns.pairplot(df, height=1.8, aspect=1.8)

        fig = sns_plot.fig
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle(name, fontsize=14)
        plt.show()
        return sns_plot

    def plot(self, df, x, y, name="", save=None, show=True):
        fig, axs = plt.subplots(nrows=len(y), ncols=len(x), figsize=(12.0, len(y)*3))
        fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)

        argmin = df["losses"].idxmin()

        for nx, _x in enumerate(x):
            for ny, _y in enumerate(y):
                ax = axs[ny, nx]
                ax.plot(df[_x].values, df[_y].values, 'o')
                ax.plot(df[_x].values[argmin], df[_y].values[argmin], 'ro')
                ax.grid(True)
                if nx == 0:
                    ax.set_ylabel(_y)
                if ny == len(y)-1:
                    ax.set_xlabel(_x)
        fig.suptitle(name, fontsize=16)
        if save is not None:
            if not os.path.isdir(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save))
            plt.savefig(save)
        if show:
            plt.show()


    def set_up(self):
        breast_cancer_data = load_breast_cancer()
        x = breast_cancer_data.data
        y = breast_cancer_data.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=23)

        x_train_fname = os.path.join(self.root, 'x_train.npy')
        y_train_fname = os.path.join(self.root, 'y_train.npy')
        if not os.path.isfile(x_train_fname):
            np.save(x_train_fname, x_train)
        if not os.path.isfile(y_train_fname):
            np.save(y_train_fname, y_train)

        self.train = [x_train, y_train]
        self.test = [x_test, y_test]
        self.config = {
            "hyperparameter": {},
            "settings": {
                "solver_plugin": {
                    "max_iterations": 1,
                    "use_plugin": "hyperopt",
                    "output_dir": os.path.join(self.root, 'test_results')
                },
                "custom": {
                    "data_path": self.root,
                    "data_name": "x_train.npy",
                    "labels_name": "y_train.npy"
                }
            }}

    def run_svc_usecase(self):
        print("\n")
        print("*" * 30)
        print("SVC Classifier")
        print("*" * 30)
        print("\n")
        hp = {
            "C": {
                "domain": "uniform",
                "data": [0.0001, 300.0],
                "type": "float"
            },
            "kernel": {
                "domain": "categorical",
                "data": ["linear", "poly", "rbf"],
                "type": "str"
            }
        }

        self.set_hyperparameter(hp)

        results = {"iterations": [], "C": [], "kernel": [], "accuracy": [], "losses": [], "duration": []}
        for n in self.iter_sequence:
            self.set_iterations(n)
            ProjectManager.set_config(self.config)
            uc = svc_usecase()
            uc.run(save=False)
            res, best = uc.get_results()
            clf = SVC(C=best['n_estimators'],
                      kernel=hp['kernel']['data'][best['kernel']])
            clf.fit(self.train[0], self.train[1])
            train_predictions = clf.predict(self.test[0])
            acc = accuracy_score(self.test[1], train_predictions)

            results['accuracy'].append(acc)
            results['iterations'].append(n)
            results['kernel'].append(best['kernel'])
            results['C'].append(best['C'])

            self.find_loss_and_time(res, results)

            print("=" * 30)
            print("Number of iterations: {}".format(n))
            print("Classifier: {}".format(clf.__class__.__name__))
            print("=" * 30)
            print("=" * 30)
            for p in best.items():
                print(p[0], ":", p[1])
            print("=" * 30)
            print("Accuracy: {:.4%}".format(acc))
            print("=" * 30)
            print("\n")

        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(self.root, "final_{}.csv".format(clf.__class__.__name__)))
        self.plot(df, x=["iterations", "losses"],
                  y=['accuracy', 'duration'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final1_{}.png".format(clf.__class__.__name__)),
                  show=False)
        self.plot(df, x=["iterations", "losses"],
                  y=['n_estimators', 'max_depth', 'max_features'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final2_{}.png".format(clf.__class__.__name__)),
                  show=False)

    def run_randomforest_usecase(self):
        print("\n")
        print("*" * 30)
        print("RandomForest Classifier")
        print("*" * 30)
        print("\n")
        hp = {
            "n_estimators": {
                "domain": "uniform",
                "data": [3, 500],
                "type": "int"
            },
            "max_depth": {
                "domain": "uniform",
                "data": [1, 50],
                "type": "int"
            },
            "max_features": {
                "domain": "categorical",
                "data": ["auto", "sqrt", "log2"],
                "type": "str"
            }
        }

        self.set_hyperparameter(hp)

        results = {"iterations": [], "n_estimators": [], "max_depth": [], "max_features": [], "accuracy": [], "losses": [], "duration": []}
        for n in self.iter_sequence:
            try:
                self.set_iterations(n)
                ProjectManager.set_config(self.config)
                uc = randomforest_usecase()
                uc.run(save=False)
                res, best = uc.get_results()
                clf = RandomForestClassifier(n_estimators=best['n_estimators'],
                                             max_depth=best['max_depth'],
                                             max_features=hp['max_features']['data'][best['max_features']])
                clf.fit(self.train[0], self.train[1])
                train_predictions = clf.predict(self.test[0])
                acc = accuracy_score(self.test[1], train_predictions)

                results['accuracy'].append(acc)
                results['iterations'].append(n)
                results['n_estimators'].append(best['n_estimators'])
                results['max_depth'].append(best['max_depth'])
                results['max_features'].append(best['max_features'])

                self.find_loss_and_time(res, results)

                print("=" * 30)
                print("Number of iterations: {}".format(n))
                print("Classifier: {}".format(clf.__class__.__name__))
                print("=" * 30)
                print("=" * 30)
                for p in best.items():
                    print(p[0], ":", p[1])
                print("=" * 30)
                print("Accuracy: {:.4%}".format(acc))
                print("=" * 30)
                print("\n")
            except Exception as e:
                print("Failed at iteration step {}, reason: {}".format(n, e))

        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(self.root, "final_{}.csv".format(clf.__class__.__name__)))
        self.plot(df, x=["iterations", "losses"],
                  y=['accuracy', 'duration'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final1_{}.png".format(clf.__class__.__name__)),
                  show=False)
        self.plot(df, x=["iterations", "losses"],
                  y=['n_estimators', 'max_depth', 'max_features'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final2_{}.png".format(clf.__class__.__name__)),
                  show=False)

    def run_adaboost_usecase(self):
        print("\n")
        print("*"*30)
        print("AdaBoost Classifier")
        print("*"*30)
        print("\n")
        hp = {
            "n_estimators": {
                "domain": "uniform",
                "data": [1, 500],
                "type": "int"
            },
            "learning_rate": {
                "domain": "uniform",
                "data": [0.001, 10],
                "type": "float"
            }
        }

        self.set_hyperparameter(hp)

        results = {"iterations": [], "n_estimators": [], "learning_rate": [], "accuracy": [], "losses": [], "duration": []}
        for n in self.iter_sequence:
            try:
                self.set_iterations(n)
                ProjectManager.set_config(self.config)
                uc = adaboost_usecase()
                uc.run(save=False)
                res, best = uc.get_results()
                clf = AdaBoostClassifier(n_estimators=best['n_estimators'],
                                         learning_rate=best['learning_rate'])
                clf.fit(self.train[0], self.train[1])
                train_predictions = clf.predict(self.test[0])
                acc = accuracy_score(self.test[1], train_predictions)

                results['accuracy'].append(acc)
                results['iterations'].append(n)
                results['n_estimators'].append(best['n_estimators'])
                results['learning_rate'].append(best['learning_rate'])

                self.find_loss_and_time(res, results)

                print("=" * 30)
                print("Number of iterations: {}".format(n))
                print("Classifier: {}".format(clf.__class__.__name__))
                print("=" * 30)
                print("=" * 30)
                for p in best.items():
                    print(p[0], ":", p[1])
                print("=" * 30)
                print("Accuracy: {:.4%}".format(acc))
                print("=" * 30)
                print("\n")
            except Exception as e:
                print("Failed at iteration step {}, reason: {}".format(n, e))

        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(self.root, "final_{}.csv".format(clf.__class__.__name__)))
        self.plot(df, x=["iterations", "losses"],
                  y=['accuracy', 'duration'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final1_{}.png".format(clf.__class__.__name__)),
                  show=False)
        self.plot(df, x=["iterations", "losses"],
                  y=['n_estimators', 'learning_rate'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final2_{}.png".format(clf.__class__.__name__)),
                  show=False)

    def run_knc_usecase(self):
        print("\n")
        print("*" * 30)
        print("KN Classifier")
        print("*" * 30)
        print("\n")
        hp = {
                "n_neighbors": {
                    "domain": "uniform",
                    "data": [1, 100],
                    "type": "int"
                },
                "weights": {
                    "domain": "categorical",
                    "data": ["uniform", "distance"],
                    "type": "str"
                },
                "algorithm": {
                    "domain": "categorical",
                    "data": ["auto", "ball_tree", "kd_tree", "brute"],
                    "type": "str"
                }
            }

        self.set_hyperparameter(hp)

        results = {"iterations": [], "n_neighbors": [], "weights": [], "algorithm": [], "accuracy": [], "losses": [], "duration": []}
        for n in self.iter_sequence:
            try:
                self.set_iterations(n)
                ProjectManager.set_config(self.config)
                uc = knc_usecase()
                uc.run(save=False)
                res, best = uc.get_results()
                clf = KNeighborsClassifier(n_neighbors=best['n_neighbors'],
                                           weights=hp["weights"]["data"][best['weights']],
                                           algorithm=hp["algorithm"]["data"][best['algorithm']])
                clf.fit(self.train[0], self.train[1])
                train_predictions = clf.predict(self.test[0])
                acc = accuracy_score(self.test[1], train_predictions)

                results['accuracy'].append(acc)
                results['iterations'].append(n)
                results['n_neighbors'].append(best['n_neighbors'])
                results['weights'].append(best['weights'])
                results['algorithm'].append(best['algorithm'])

                self.find_loss_and_time(res, results)

                print("=" * 30)
                print("Number of iterations: {}".format(n))
                print("Classifier: {}".format(clf.__class__.__name__))
                print("=" * 30)
                print("=" * 30)
                for p in best.items():
                    print(p[0], ":", p[1])
                print("=" * 30)
                print("Accuracy: {:.4%}".format(acc))
                print("=" * 30)
                print("\n")
            except Exception as e:
                print("Failed at iteration step {}, reason: {}".format(n, e))

        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(self.root, "final_{}.csv".format(clf.__class__.__name__)))
        self.plot(df, x=["iterations", "losses"],
                  y=['accuracy', 'duration'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final1_{}.png".format(clf.__class__.__name__)),
                  show=False)
        self.plot(df, x=["iterations", "losses"],
                  y=['n_neighbors', 'weights', 'algorithm'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final2_{}.png".format(clf.__class__.__name__)),
                  show=False)

    def run_gradientboost_usecase(self):
        print("\n")
        print("*" * 30)
        print("GradientBoost Classifier")
        print("*" * 30)
        print("\n")
        hp = {
            "n_estimators": {
                "domain": "uniform",
                "data": [3, 500],
                "type": "int"
            },
            "learning_rate": {
                "domain": "uniform",
                "data": [0.001, 10],
                "type": "float"
            },
            "min_samples_split": {
                "domain": "uniform",
                "data": [0.0, 1.0],
                "type": "float"
            },
            "min_samples_leaf": {
                "domain": "uniform",
                "data": [0.00001, 0.5],
                "type": "float"
            },
            "max_depth": {
                "domain": "uniform",
                "data": [1, 50],
                "type": "int"
            }
        }

        self.set_hyperparameter(hp)

        results = {"iterations": [], "n_estimators": [], "max_depth": [],
                   "learning_rate": [], "min_samples_split": [], "min_samples_leaf": [],
                   "accuracy": [], "losses": [], "duration": []}
        for n in self.iter_sequence:
            try:
                self.set_iterations(n)
                ProjectManager.set_config(self.config)
                uc = gradientboost_usecase()
                uc.run(save=False)
                res, best = uc.get_results()
                clf = GradientBoostingClassifier(n_estimators=best['n_estimators'],
                                                 max_depth=best['max_depth'],
                                                 learning_rate=best['learning_rate'],
                                                 min_samples_split=best['min_samples_split'],
                                                 min_samples_leaf=best['min_samples_leaf'])
                clf.fit(self.train[0], self.train[1])
                train_predictions = clf.predict(self.test[0])
                acc = accuracy_score(self.test[1], train_predictions)

                results['accuracy'].append(acc)
                results['iterations'].append(n)
                results['n_estimators'].append(best['n_estimators'])
                results['max_depth'].append(best['max_depth'])
                results['learning_rate'].append(best['learning_rate'])
                results['min_samples_split'].append(best['min_samples_split'])
                results['min_samples_leaf'].append(best['min_samples_leaf'])

                self.find_loss_and_time(res, results)

                print("=" * 30)
                print("Number of iterations: {}".format(n))
                print("Classifier: {}".format(clf.__class__.__name__))
                print("=" * 30)
                print("=" * 30)
                for p in best.items():
                    print(p[0], ":", p[1])
                print("=" * 30)
                print("Accuracy: {:.4%}".format(acc))
                print("=" * 30)
                print("\n")
            except Exception as e:
                print("Failed at iteration step {}, reason: {}".format(n, e))

        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(self.root, "final_{}.csv".format(clf.__class__.__name__)))
        self.plot(df, x=["iterations", "losses"],
                  y=['accuracy', 'duration'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final1_{}.png".format(clf.__class__.__name__)),
                  show=False)
        self.plot(df, x=["iterations", "losses"],
                  y=['n_estimators', 'max_depth', 'learning_rate', 'min_samples_split', 'min_samples_leaf'],
                  name="Classifier: {}".format(clf.__class__.__name__),
                  save=os.path.join(self.root, "final2_{}.png".format(clf.__class__.__name__)),
                  show=False)


    def clean_up(self):
        if os.path.isdir(self.root):
            shutil.rmtree(self.root)


if __name__ == "__main__":
    performance_test = PerformanceTest(root="C:/Users/s635r/Desktop")
    performance_test.run()
