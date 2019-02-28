# DKFZ
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

import os
import sys
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

LIBNAME = "hyppopy"
PLUGIN_DEFAULT_DIR = os.path.join(ROOT, *(LIBNAME, "plugins"))
TESTDATA_DIR = os.path.join(ROOT, *(LIBNAME, "tests", "data"))
SETTINGSSOLVERPATH = "settings/solver_plugin"
SETTINGSCUSTOMPATH = "settings/custom"
DEEPDICT_XML_ROOT = LIBNAME
RANDOMSAMPLES = 10000
DEFAULTITERATIONS = 500

LOGFILENAME = os.path.join(ROOT, '{}_log.log'.format(LIBNAME))
DEBUGLEVEL = logging.DEBUG
logging.basicConfig(filename=LOGFILENAME, filemode='w', format='%(levelname)s: %(name)s - %(message)s')


