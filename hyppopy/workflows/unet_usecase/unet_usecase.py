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

import os
import pickle

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from networks.RecursiveUNet import UNet

import hyppopy.solverfactory as sfac
from .unet_uscase_utils import *


def unet_usecase(args):
    print("Execute UNet UseCase...")
    data_dir = args.data
    preprocessed_dir = os.path.join(args.data, 'preprocessed')
    solver_plugin = args.plugin
    config_file = args.config
    print(f"input data directory: {data_dir}")
    print(f"use plugin: {solver_plugin}")
    print(f"config file: {config_file}")

    factory = sfac.SolverFactory.instance()
    solver = factory.get_solver(solver_plugin)
    solver.read_parameter(config_file)

    if preprocess_data(data_dir):
        create_splits(output_dir=data_dir, image_dir=preprocessed_dir)

    with open(os.path.join(data_dir, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)

    tr_keys = splits[solver.settings.fold]['train']
    val_keys = splits[solver.settings.fold]['val']
    test_keys = splits[solver.settings.fold]['test']

    def loss_function(patch_size, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data_loader = NumpyDataSet(data_dir,
                                         target_size=patch_size,
                                         batch_size=batch_size,
                                         keys=tr_keys)
        val_data_loader = NumpyDataSet(data_dir,
                                       target_size=patch_size,
                                       batch_size=batch_size,
                                       mode="val",
                                       do_reshuffle=False)
        model = UNet(num_classes=solver.settings.num_classes, in_channels=solver.settings.in_channels)
        model.to(device)

        dice_loss = SoftDiceLoss(batch_dice=True)
        ce_loss = torch.nn.CrossEntropyLoss()
        node_optimizer = optim.Adam(model.parameters(), lr=solver.settings.learning_rate)
        scheduler = ReduceLROnPlateau(node_optimizer, 'min')

        model.train()

        data = None
        batch_counter = 0
        for data_batch in train_data_loader:

            node_optimizer.zero_grad()

            data = data_batch['data'][0].float().to(device)
            target = data_batch['seg'][0].long().to(device)

            pred = model(data)
            pred_softmax = F.softmax(pred, dim=1)

            loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
            loss.backward()
            node_optimizer.step()
            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        model.eval()

        data = None
        loss_list = []

        with torch.no_grad():
            for data_batch in val_data_loader:
                data = data_batch['data'][0].float().to(device)
                target = data_batch['seg'][0].long().to(device)

                pred = model(data)
                pred_softmax = F.softmax(pred)

                loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        scheduler.step(np.mean(loss_list))

    data = []

    # solver.set_data(data)
    # solver.read_parameter(config_file)
    # solver.set_loss_function(loss_function)
    # solver.run()
    # solver.get_results()






