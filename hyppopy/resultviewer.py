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

    def show(self, save=True):
        if self.has_duration:
            sns_plot = sns.jointplot(y="duration", x="losses", data=self.df, kind="kde")
            if not self.save_only:
                plt.show()
            if save:
                save_name = os.path.join(self.path, "t_vs_loss_"+self.appendix+".png")
                try:
                    sns_plot.savefig(save_name)
                except Exception as e:
                    msg = "failed to save file {}, reason {}".format(save_name, e)
                    LOG.error(msg)
                    raise IOError(msg)
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

