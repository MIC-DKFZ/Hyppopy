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

import os
import sys
import logging
from enum import Enum

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

LIBNAME = "hyppopy"
TESTDATA_DIR = os.path.join(ROOT, *(LIBNAME, "tests", "data"))

HYPERPARAMETERPATH = "hyperparameter"
SETTINGSPATH = "settings"
FUNCTIONSIMULATOR_DATAPATH = os.path.join(os.path.join(ROOT, LIBNAME), "virtualparameterspace")

SUPPORTED_DOMAINS = ["uniform", "normal", "loguniform", "categorical"]
SUPPORTED_DTYPES = ["int", "float", "str"]

DEFAULTGRIDFREQUENCY = 10

LOGFILENAME = os.path.join(ROOT, '{}_log.log'.format(LIBNAME))
DEBUGLEVEL = logging.DEBUG
logging.basicConfig(filename=LOGFILENAME, filemode='w', format='%(levelname)s: %(name)s - %(message)s')


class MPI_TAGS(Enum):
     MPI_SEND_CANDIDATE = 55
     MPI_SEND_RESULTS = 99
