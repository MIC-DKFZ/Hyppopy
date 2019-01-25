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
LOGFILENAME = os.path.join(ROOT, 'logfile.log')
PLUGIN_DEFAULT_DIR = os.path.join(ROOT, *("hyppopy", "solver"))
sys.path.insert(0, ROOT)

#LOG = logging.getLogger()
logging.getLogger('hyppopy').setLevel(logging.DEBUG)
logging.basicConfig(filename=LOGFILENAME, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

'''
LOG.debug('debug message')
LOG.info('info message')
LOG.warning('warning message')
LOG.error('error message')
LOG.critical('critical message')
'''
