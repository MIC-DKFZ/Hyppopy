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

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.svc_usecase.svc_usecase import svc_usecase
from hyppopy.workflows.knc_usecase.knc_usecase import knc_usecase
from hyppopy.workflows.lda_usecase.lda_usecase import lda_usecase
from hyppopy.workflows.unet_usecase.unet_usecase import unet_usecase
from hyppopy.workflows.randomforest_usecase.randomforest_usecase import randomforest_usecase
from hyppopy.workflows.imageregistration_usecase.imageregistration_usecase import imageregistration_usecase


import os
import sys
import time
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
    parser.add_argument('-w', '--workflow', type=str, help='workflow to be executed')
    parser.add_argument('-o', '--output', type=str, default=None, help='output path to store result')
    parser.add_argument('-c', '--config', type=str, help='config filename, .xml or .json formats are supported.'
                                                         'pass a full path filename or the filename only if the'
                                                         'configfile is in the data folder')

    args = parser.parse_args()
    args_check(args)

    ProjectManager.read_config(args.config)

    if args.output is not None:
        ProjectManager.output_dir = args.output

    if args.workflow == "svc_usecase":
        uc = svc_usecase()
    elif args.workflow == "randomforest_usecase":
        uc = randomforest_usecase()
    elif args.workflow == "knc_usecase":
        uc = knc_usecase()
    elif args.workflow == "lda_usecase":
        uc = lda_usecase()
    elif args.workflow == "unet_usecase":
        uc = unet_usecase()
    elif args.workflow == "imageregistration_usecase":
        uc = imageregistration_usecase()
    else:
        print("No workflow called {} found!".format(args.workflow))
        sys.exit()

    print("\nStart optimization...")
    start = time.process_time()
    uc.run(save=True)
    end = time.process_time()

    print("Finished optimization!\n")
    print("Total Time: {}s\n".format(end-start))
    res, best = uc.get_results()
    print("---- Optimal Parameter -----\n")
    for p in best.items():
        print(" - {}\t:\t{}".format(p[0], p[1]))
