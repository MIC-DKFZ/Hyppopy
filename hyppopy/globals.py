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

# -*- coding: utf-8 -*-

import os
import sys
import logging

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

PLUGIN_DEFAULT_DIR = os.path.join(ROOT, *("hyppopy", "plugins"))

LOGFILENAME = os.path.join(ROOT, 'logfile.log')
DEBUGLEVEL = logging.DEBUG
logging.basicConfig(filename=LOGFILENAME, filemode='w', format='%(levelname)s: %(name)s - %(message)s')


