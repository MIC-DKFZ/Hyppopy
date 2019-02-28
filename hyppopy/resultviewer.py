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
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

sns.set(style="darkgrid")


class ResultViewer(object):

    def __init__(self, fname=None, save_only=False):
        self.df = None
        self.has_duration = False
        self.hyperparameter = None
        self.save_only = save_only
        self.path = None
        self.appendix = None
        if fname is not None:
            self.read(fname)

    def close_all(self):
        plt.close('all')

    def read(self, fname):
        self.path = os.path.dirname(fname)
        split = os.path.basename(fname).split("_")
        self.appendix = split[-1]
        self.appendix = self.appendix[:-4]
        self.df = pd.read_csv(fname, index_col=0)
        const_data = ["duration", "losses"]
        hyperparameter_columns = [item for item in self.df.columns if item not in const_data]
        self.hyperparameter = pd.DataFrame()
        for key in hyperparameter_columns:
            self.hyperparameter[key] = self.df[key]
        self.has_duration = "duration" in self.df.columns

    def plot_XYGrid(self, df, x, y, name="", save=None, show=True):
        argmin = df["losses"].idxmin()
        grid = [len(x), len(y)]
        if grid[0] == 1 and grid[1] == 1:
            fig = plt.figure(figsize=(10.0, 8))
            plt.plot(df[x[0]].values, df[y[0]].values, '.')
            plt.plot(df[x[0]].values[argmin], df[y[0]].values[argmin], 'ro')
            plt.grid(True)
            plt.ylabel(y[0])
            plt.xlabel(x[0])
            plt.title(name, fontsize=16)
        else:
            if grid[0] > 1 and grid[1] == 1:
                fig, axs = plt.subplots(ncols=grid[0], figsize=(10.0, grid[1] * 3.5))
            elif grid[0] == 1 and grid[1] > 1:
                fig, axs = plt.subplots(nrows=grid[1], figsize=(10.0, grid[1] * 3.5))
            else:
                fig, axs = plt.subplots(nrows=grid[1], ncols=grid[0], figsize=(10.0, grid[1] * 3.5))
            fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)

            for nx, _x in enumerate(x):
                for ny, _y in enumerate(y):
                    if grid[0] > 1 and grid[1] == 1:
                        ax = axs[nx]
                    elif grid[0] == 1 and grid[1] > 1:
                        ax = axs[ny]
                    else:
                        ax = axs[ny, nx]
                    ax.plot(df[_x].values, df[_y].values, '.')
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

    def plot_performance_and_feature_grids(self, save=True):
        x_axis = []
        if 'losses' in self.df.columns:
            x_axis.append('losses')
        if 'iterations' in self.df.columns:
            x_axis.append('iterations')
        y_axis_performance = []
        if 'accuracy' in self.df.columns:
            y_axis_performance.append('accuracy')
        if 'duration' in self.df.columns:
            y_axis_performance.append('duration')
        features = []
        for cit in self.df.columns:
            if cit not in x_axis and cit not in y_axis_performance:
                    features.append(cit)

        save_name = None
        if save:
            save_name = os.path.join(self.path, "performance" + self.appendix + ".png")
        self.plot_XYGrid(self.df, x=x_axis,
                  y=y_axis_performance,
                  name="Performance",
                  save=save_name,
                  show=not self.save_only)

        chunks = [features[x:x + 3] for x in range(0, len(features), 3)]
        for n, chunk in enumerate(chunks):
            save_name = None
            if save:
                save_name = os.path.join(self.path, "features_{}_".format(str(n).zfill(3)) + self.appendix + ".png")
            self.plot_XYGrid(self.df, x=x_axis,
                      y=chunk,
                      name="Feature set {}".format(n+1),
                      save=save_name,
                      show=not self.save_only)

    def plot_feature_matrix(self, save=True):
        sns_plot = sns.pairplot(self.df, height=1.8, aspect=1.8,
                          plot_kws=dict(edgecolor="k", linewidth=0.5),
                          diag_kind="kde", diag_kws=dict(shade=True))

        fig = sns_plot.fig
        fig.subplots_adjust(top=0.93, wspace=0.3)
        t = fig.suptitle('Pairwise Plots', fontsize=14)
        if not self.save_only:
            plt.show()
        if save:
            save_name = os.path.join(self.path, "matrixview_"+self.appendix+".png")
            try:
                sns_plot.savefig(save_name)
            except Exception as e:
                msg = "failed to save file {}, reason {}".format(save_name, e)
                LOG.error(msg)
                raise IOError(msg)

    def plot_duration(self, save=True):
        if "duration" in self.df.columns:
            sns_plot = sns.jointplot(y="duration", x="losses", data=self.df, kind="kde")
            if not self.save_only:
                plt.show()
            if save:
                save_name = os.path.join(self.path, "t_vs_loss_" + self.appendix + ".png")
                try:
                    sns_plot.savefig(save_name)
                except Exception as e:
                    msg = "failed to save file {}, reason {}".format(save_name, e)
                    LOG.error(msg)
                    raise IOError(msg)

    def show(self, save=True):
        self.plot_duration(save)
        self.plot_feature_matrix(save)
        self.plot_performance_and_feature_grids(save)

