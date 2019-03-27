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
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import os
import sys
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

LIBNAME = "hyppopy"
TESTDATA_DIR = os.path.join(ROOT, *(LIBNAME, "tests", "data"))

HYPERPARAMETERPATH = "hyperparameter"
SETTINGSPATH = "settings"
VFUNCDATAPATH = os.path.join(os.path.join(ROOT, LIBNAME), "virtualparameterspace")

SUPPORTED_DOMAINS = ["uniform", "normal", "loguniform", "categorical"]
SUPPORTED_DTYPES = ["int", "float", "str"]

DEFAULTITERATIONS = 500

LOGFILENAME = os.path.join(ROOT, '{}_log.log'.format(LIBNAME))
DEBUGLEVEL = logging.DEBUG
logging.basicConfig(filename=LOGFILENAME, filemode='w', format='%(levelname)s: %(name)s - %(message)s')
