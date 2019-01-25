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

import argparse

import logging
LOG = logging.getLogger('hyppopy')

from hyppopy.solver_factory import SolverFactory


def cmd_workflow():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-v', '--verbosity', type=int, required=False, default=0,
                        help='number of thoughts our thinker should produce')


    args_dict = vars(parser.parse_args())
    factory = SolverFactory.instance()
