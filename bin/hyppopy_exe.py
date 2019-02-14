#!/usr/bin/env python
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


from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.unet_usecase.unet_usecase import unet_usecase
from hyppopy.workflows.svc_usecase.svc_usecase import svc_usecase
from hyppopy.workflows.randomforest_usecase.randomforest_usecase import randomforest_usecase
from hyppopy.workflows.imageregistration_usecase.imageregistration_usecase import imageregistration_usecase


import os
import sys
import argparse


def print_warning(msg):
    print("\n!!!!! WARNING !!!!!")
    print(msg)
    sys.exit()


def args_check(args):
    if not args.workflow:
        print_warning("No workflow specified, check --help")
    if not args.config:
        print_warning("Missing config parameter, check --help")
    if not os.path.isfile(args.config):
        print_warning(f"Couldn't find configfile ({args.config}), please check your input --config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Hyppopy UseCase Example Optimization.')
    parser.add_argument('-w', '--workflow', type=str,
                        help='workflow to be executed')
    parser.add_argument('-c', '--config', type=str, help='config filename, .xml or .json formats are supported.'
                                                         'pass a full path filename or the filename only if the'
                                                         'configfile is in the data folder')

    args = parser.parse_args()
    args_check(args)

    ProjectManager.read_config(args.config)

    if args.workflow == "svc_usecase":
        uc = svc_usecase()
    elif args.workflow == "randomforest_usecase":
        uc = randomforest_usecase()
    elif args.workflow == "unet_usecase":
        uc = unet_usecase()
    elif args.workflow == "imageregistration_usecase":
        uc = imageregistration_usecase()
    else:
        print("No workflow called {} found!".format(args.workflow))
        sys.exit()

    uc.run()
    print(uc.get_results())
