# Hyppopy - A Hyper-Parameter Optimization Toolbox
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

import warnings
import numpy as np
from visdom import Visdom
import matplotlib.pyplot as plt


def time_formatter(time_s):
    if time_s < 0.01:
        return int(time_s * 1000.0 * 1000) / 1000.0, "ms"
    elif 100 < time_s < 3600:
        return int(time_s / 60 * 1000) / 1000.0, "min"
    elif time_s >= 3600:
        return int(time_s / 3600 * 1000) / 1000.0, "h"
    else:
        return int(time_s * 1000) / 1000.0, "s"


class VisdomViewer(object):

    def __init__(self, project, port=8097, server="http://localhost"):
        self._viz = Visdom(port=port, server=server)
        self._enabled = self._viz.check_connection(timeout_seconds=3)
        if not self._enabled:
            warnings.warn("No connection to visdom server established. Visualization cannot be displayed!")

        self._project = project
        self._best_win = None
        self._best_loss = None
        self._loss_iter_plot = None
        self._status_report = None
        self._axis_tags = None
        self._axis_plots = None

    def plot_losshistory(self, input_data):
        loss = np.array([input_data["loss"]])
        iter = np.array([input_data["iterations"]])
        if self._loss_iter_plot is None:
            self._loss_iter_plot = self._viz.line(loss, X=iter, opts=dict(
                markers=True,
                markersize=5,
                dash=np.array(['dashdot']),
                title="Loss History",
                xlabel='iteration',
                ylabel='loss'
            ))
        else:
            self._viz.line(loss, X=iter, win=self._loss_iter_plot, update='append')

    def plot_hyperparameter(self, input_data):
        if self._axis_plots is None:
            self._axis_tags = []
            self._axis_plots = {}
            for item in input_data.keys():
                if item == "refresh_time" or item == "book_time" or item == "iterations" or item == "status" or item == "loss":
                    continue
                self._axis_tags.append(item)
            for axis in self._axis_tags:
                xlabel = "value"
                if isinstance(input_data[axis], str):
                    if self._project.hyperparameter[axis]["domain"] == "categorical":
                        xlabel = '-'.join(self._project.hyperparameter[axis]["data"])
                        input_data[axis] = self._project.hyperparameter[axis]["data"].index(input_data[axis])
                axis_loss = np.array([input_data[axis], input_data["loss"]]).reshape(1, -1)
                self._axis_plots[axis] = self._viz.scatter(axis_loss, opts=dict(
                                                                      markersize=5,
                                                                      title=axis,
                                                                      xlabel=xlabel,
                                                                      ylabel='loss'))
        else:
            for axis in self._axis_tags:
                if isinstance(input_data[axis], str):
                    if self._project.hyperparameter[axis]["domain"] == "categorical":
                        input_data[axis] = self._project.hyperparameter[axis]["data"].index(input_data[axis])
                axis_loss = np.array([input_data[axis], input_data["loss"]]).reshape(1, -1)
                self._viz.scatter(axis_loss, win=self._axis_plots[axis], update='append')

    def show_statusreport(self, input_data):
        duration = input_data['refresh_time'] - input_data['book_time']
        duration, time_format = time_formatter(duration.total_seconds())
        report = "Iteration {}:&nbsp;{}{}&nbsp;->&nbsp;{}\n".format(input_data["iterations"], duration, time_format, input_data["status"])
        if self._status_report is None:
            self._status_report = self._viz.text(report)
        else:
            self._viz.text(report, win=self._status_report, append=True)

    def show_best(self, input_data):
        if self._best_win is None:
            self._best_loss = input_data["loss"]
            txt = "Best Parameter Set:<hr>Loss: {}<hr><ul>".format(self._best_loss)
            for axis in self._axis_tags:
                if self._project.hyperparameter[axis]["domain"] == "categorical":
                    txt += "<li>{} = {}</li>".format(axis, self._project.hyperparameter[axis]["data"][input_data[axis]])
                else:
                    txt += "<li>{} = {}</li>".format(axis, input_data[axis])
            txt += "</ul>"
            self._best_win = self._viz.text(txt)
        else:
            if input_data["loss"] < self._best_loss:
                self._best_loss = input_data["loss"]
                txt = "Best Parameter Set:<hr>Loss: {}<hr><ul>".format(self._best_loss)
                for axis in self._axis_tags:
                    if self._project.hyperparameter[axis]["domain"] == "categorical":
                        txt += "<li>{} = {}</li>".format(axis, self._project.hyperparameter[axis]["data"][input_data[axis]])
                    else:
                        txt += "<li>{} = {}</li>".format(axis, input_data[axis])
                txt += "</ul>"
                self._viz.text(txt, win=self._best_win, append=False)

    def update(self, input_data):
        if self._enabled:
            self.show_statusreport(input_data)
            self.plot_losshistory(input_data)
            self.plot_hyperparameter(input_data)
            self.show_best(input_data)
