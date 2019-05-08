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

########################################################################################################################
# USAGE
#
# The class FunctionSimulator is meant to be a virtual energy function with an arbitrary dimensionality. The user can
# simply scribble functions as a binary image using e.g. Gimp, defining their ranges using .cfg file and loading them
# into the FunctionSimulator. An instance of the class can then be used like a normal function returning the sampling of
# each dimension loaded.
#
# 1. create binary images (IMPORTANT same shape for each), background black the function signature white, ensure that
#    each column has a white pixel. If more than one pixel appears in a column, only the lowest will be used.
#
# 2. create a .cfg file, see an example in hyppopy/virtualparameterspace
#
# 3. vfunc = FunctionSimulator()
#    vfunc.load_images(path/of/your/binaryfiles/and/the/configfile)
#
# 4. use vfunc like a normal function, if you loaded 4 dimension binary images use it like f = vfunc(a,b,c,d)
########################################################################################################################

__all__ = ['FunctionSimulator']

import os
import sys
import numpy as np
import configparser
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from hyppopy.globals import FUNCTIONSIMULATOR_DATAPATH


class FunctionSimulator(object):
    """
    The FunctionSimulator class serves as simulation tool for solver testing and evaluation purposes. It's designed to
    simulate an energy functional by setting axis data for each dimension via binary image files. The binary image files
    are sampled and a range interval is read from a config file. The class implements __call__ to act like a blackbox function
    when initialized.

    f=f(x1,x2,...,xn) [for n binary images and n range config files

    as image input .png grayscale images are expected
    as range config .cfg ascii files are expected containing 
    """
    def __init__(self):
        self.config = None
        self.data = None
        self.axis = []

    def __call__(self, *args, **kwargs):
        """
        the call function expects the hyperparameter
        :param args:
        :param kwargs:
        :return:
        """
        if len(kwargs) == self.dims():
            args = [0]*len(kwargs)
            for key, value in kwargs.items():
                index = int(key.split("_")[1])
                args[index] = value
        assert len(args) == self.dims(), "wrong number of arguments!"
        for i in range(len(args)):
            assert self.axis[i][0] <= args[i] <= self.axis[i][1], "out of range access on axis {}!".format(i)
        lpos, rpos, fracs = self.pos_to_indices(args)
        fl = self.data[(list(range(self.dims())), lpos)]
        fr = self.data[(list(range(self.dims())), rpos)]
        return np.sum(fl*np.array(fracs) + fr*(1-np.array(fracs)))

    def clear(self):
        self.axis.clear()
        self.data = None
        self.config = None

    def dims(self):
        return self.data.shape[0]

    def size(self):
        return self.data.shape[1]

    def range(self, dim):
        return np.abs(self.axis[dim][1] - self.axis[dim][0])

    def minima(self):
        glob_mins = []
        for dim in range(self.dims()):
            x = []
            fmin = np.min(self.data[dim, :])
            for _x in range(self.size()):
                if self.data[dim, _x] <= fmin:
                    x.append(_x/self.size()*(self.axis[dim][1]-self.axis[dim][0])+self.axis[dim][0])
            glob_mins.append([x, fmin])
        return glob_mins

    def pos_to_indices(self, positions):
        lpos = []
        rpos = []
        pfracs = []
        for n in range(self.dims()):
            pos = positions[n]
            pos -= self.axis[n][0]
            pos /= np.abs(self.axis[n][1]-self.axis[n][0])
            pos *= self.data.shape[1]-1
            lp = int(np.floor(pos))
            if lp < 0:
                lp = 0
            rp = int(np.ceil(pos))
            if rp > self.data.shape[1]-1:
                rp = self.data.shape[1]-1
            pfracs.append(1.0-(pos-np.floor(pos)))
            lpos.append(lp)
            rpos.append(rp)
        return lpos, rpos, pfracs

    def plot(self, dim=None, title=""):
        if dim is None:
            dim = list(range(self.dims()))
        else:
            dim = [dim]
        fig = plt.figure(figsize=(10, 8))
        for i in range(len(dim)):
            width = np.abs(self.axis[dim[i]][1]-self.axis[dim[i]][0])
            ax = np.arange(self.axis[dim[i]][0], self.axis[dim[i]][1], width/self.size())
            plt.plot(ax, self.data[dim[i], :], '.', label='axis_{}'.format(str(dim[i]).zfill(2)))
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.show()

    def add_dimension(self, data, x_range):
        if self.data is None:
            self.data = data
            if len(self.data.shape) == 1:
                self.data = self.data.reshape((1, self.data.shape[0]))
        else:
            if len(data.shape) == 1:
                data = data.reshape((1, data.shape[0]))
            assert self.data.shape[1] == data.shape[1], "shape mismatch while adding dimension!"
            dims = self.data.shape[0]
            size = self.data.shape[1]
            tmp = np.append(self.data, data)
            self.data = tmp.reshape((dims+1, size))
        self.axis.append(x_range)

    def load_default(self, name="3D"):
        path = os.path.join(FUNCTIONSIMULATOR_DATAPATH, "{}".format(name))
        if os.path.exists(path):
            self.load_images(path)
        else:
            raise FileExistsError("No FunctionSimulator of dimension {} available".format(name))

    def load_images(self, path):
        self.config = None
        self.data = None
        self.axis.clear()
        img_fnames = []
        for f in glob(path + os.sep + "*"):
            if f.endswith(".png"):
                img_fnames.append(f)
            elif f.endswith(".cfg"):
                self.config = self.read_config(f)
            else:
                print("WARNING: files of type {} not supported, the file {} is ignored!".format(f.split(".")[-1],
                                                                                                os.path.basename(f)))

        if self.config is None:
            print("Aborted, failed to read configfile!")
            sys.exit()
        sections = self.config.sections()
        if len(sections) != len(img_fnames):
            print("Aborted, inconsistent number of image tmplates and axis specifications!")
            sys.exit()
        img_fnames.sort()
        size_x = None
        size_y = None
        for n, fname in enumerate(img_fnames):
            img = mpimg.imread(fname)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if size_x is None:
                size_x = img.shape[1]
            if size_y is None:
                size_y = img.shape[0]
                self.data = np.zeros((len(img_fnames), size_x), dtype=np.float32)
            assert img.shape[0] == size_y, "Shape mismatch in dimension y {} is not {}".format(img.shape[0], size_y)
            assert img.shape[1] == size_x, "Shape mismatch in dimension x {} is not {}".format(img.shape[1], size_x)

            self.sample_image(img, n)

    def sample_image(self, img, dim):
        sec_name = "axis_{}".format(str(dim).zfill(2))
        assert sec_name in self.config.sections(), "config section {} not found!".format(sec_name)
        settings = self.get_axis_settings(sec_name)
        self.axis.append([float(settings['min_x']), float(settings['max_x'])])
        y_range = [float(settings['min_y']), float(settings['max_y'])]

        for x in range(img.shape[1]):
            candidates = np.where(img[:, x] > 0)
            assert len(candidates[0]) > 0, "non function value in image detected, ensure each column has at least one value > 0!"

            y_pos = candidates[0][0]/img.shape[0]
            self.data[dim, x] = 1-y_pos

        self.data[dim, :] *= np.abs(y_range[1] - y_range[0])
        self.data[dim, :] += y_range[0]

    def read_config(self, fname):
        try:
            config = configparser.ConfigParser()
            config.read(fname)
            return config
        except Exception as e:
            print(e)
            return None

    def get_axis_settings(self, section):
        dict1 = {}
        options = self.config.options(section)
        for option in options:
            try:
                dict1[option] = self.config.get(section, option)
                if dict1[option] == -1:
                    print("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1


