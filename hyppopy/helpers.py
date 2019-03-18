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
import logging
from hyppopy.globals import DEBUGLEVEL

LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)


# define function spliting input dict
# into categorical and non-categorical
def split_categorical(pdict):
    categorical = {}
    uniform = {}
    for name, pset in pdict.items():
        for key, value in pset.items():
            if key == 'domain' and value == 'categorical':
                categorical[name] = pset
            elif key == 'domain':
                uniform[name] = pset
    return categorical, uniform
