#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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


from hyppopy.workflows.unet_usecase import unet_usecase
from hyppopy.workflows.svc_usecase import svc_usecase
from hyppopy.workflows.randomforest_usecase import randomforest_usecase


import os
import sys
import argparse
import hyppopy.solverfactory as sfac


solver_factory = sfac.SolverFactory.instance()


def print_warning(msg):
    print("\n!!!!! WARNING !!!!!")
    print(msg)
    sys.exit()


def args_check(args):
    if not args.workflow:
        print_warning("No workflow specified, check --help")
    if not args.config:
        print_warning("Missing config parameter, check --help")
    if not args.data:
        print_warning("Missing data parameter, check --help")
    if not os.path.isdir(args.data):
        print_warning("Couldn't find data path, please check your input --data")

    if not os.path.isfile(args.config):
        tmp = os.path.join(args.data, args.config)
        if not os.path.isfile(tmp):
            print_warning("Couldn't find the config file, please check your input --config")
        args.config = tmp
    if args.plugin not in solver_factory.list_solver():
        print_warning(f"The requested plugin {args.plugin} is not available, please check for typos. Plugin options :"
                      f"{', '.join(solver_factory.list_solver())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Hyppopy UseCase Example Optimization.')
    parser.add_argument('-w', '--workflow', type=str,
                        help='workflow to be executed')
    parser.add_argument('-p', '--plugin', type=str, default='hyperopt',
                        help='plugin to be used default=[hyperopt], optunity')
    parser.add_argument('-d', '--data', type=str, help='training data path')
    parser.add_argument('-c', '--config', type=str, help='config filename, .xml or .json formats are supported.'
                                                         'pass a full path filename or the filename only if the'
                                                         'configfile is in the data folder')
    parser.add_argument('-i', '--iterations', type=int, default=0,
                        help='number of iterations, default=[0] if set to 0 the value set via configfile is used, '
                             'otherwise the configfile value will be overwritten')

    args = parser.parse_args()

    args_check(args)

    if args.workflow == "svc_usecase":
        svc_usecase.svc_usecase(args)
    elif args.workflow == "randomforest_usecase":
        randomforest_usecase.randomforest_usecase(args)
    elif args.workflow == "unet_usecase":
        unet_usecase.unet_usecase(args)
    else:
        print(f"No workflow called {args.workflow} found!")
