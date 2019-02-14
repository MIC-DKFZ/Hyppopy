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
import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import torch.optim as optim
import torch.nn.functional as F
from .networks.RecursiveUNet import UNet
from .loss_functions.dice_loss import SoftDiceLoss
from sklearn.model_selection import cross_val_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .datasets.two_dim.NumpyDataLoader import NumpyDataSet

from hyppopy.projectmanager import ProjectManager
from hyppopy.workflows.workflowbase import WorkflowBase
from hyppopy.workflows.dataloader.unetloader import UnetDataLoader


class unet_usecase(WorkflowBase):

    def setup(self):
        dl = UnetDataLoader()
        dl.start(data_path=ProjectManager.data_path,
                 data_name=ProjectManager.data_name,
                 image_dir=ProjectManager.image_dir,
                 labels_dir=ProjectManager.labels_dir,
                 split_dir=ProjectManager.split_dir,
                 output_dir=ProjectManager.data_path,
                 num_classes=ProjectManager.num_classes)
        self.solver.set_data(dl.data)

    def blackbox_function(self, data, params):
        if "batch_size" in params.keys():
            params["batch_size"] = int(round(params["batch_size"]))
        if "batch_size" in params.keys():
            params["batch_size"] = int(round(params["batch_size"]))
        if "n_epochs" in params.keys():
            params["n_epochs"] = int(round(params["n_epochs"]))

        batch_size = 8
        patch_size = 64

        tr_keys = data[ProjectManager.fold]['train']
        val_keys = data[ProjectManager.fold]['val']

        data_dir = os.path.join(ProjectManager.data_path, *(ProjectManager.data_name, ProjectManager.preprocessed_dir))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data_loader = NumpyDataSet(data_dir,
                                         target_size=patch_size,
                                         batch_size=batch_size,
                                         keys=tr_keys)
        val_data_loader = NumpyDataSet(data_dir,
                                        target_size=patch_size,
                                        batch_size=batch_size,
                                        keys=val_keys,
                                        mode="val",
                                        do_reshuffle=False)

        model = UNet(num_classes=ProjectManager.num_classes,
                     in_channels=ProjectManager.in_channels)
        model.to(device)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax für DICE Loss!
        ce_loss = torch.nn.CrossEntropyLoss()  # Kein Softmax für CE Loss -> ist in torch schon mit drin!

        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        losses = []
        print("n_epochs {}".format(params['n_epochs']))
        for epoch in range(params["n_epochs"]):
            #### Train ####
            model.train()
            data = None
            batch_counter = 0
            for data_batch in train_data_loader:
                optimizer.zero_grad()

                # Shape of data_batch = [1, b, c, w, h]
                # Desired shape = [b, c, w, h]
                # Move data and target to the GPU
                data = data_batch['data'][0].float().to(device)
                target = data_batch['seg'][0].long().to(device)

                pred = model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
                loss.backward()
                optimizer.step()
                batch_counter += 1
            ###############

            #### Validate ####
            model.eval()
            data = None
            loss_list = []
            with torch.no_grad():
                for data_batch in val_data_loader:
                    data = data_batch['data'][0].float().to(device)
                    target = data_batch['seg'][0].long().to(device)

                    pred = model(data)
                    pred_softmax = F.softmax(pred)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                    loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
                    loss_list.append(loss.item())

            assert data is not None, 'data is None. Please check if your dataloader works properly'
            scheduler.step(np.mean(loss_list))
            losses.append(np.mean(loss_list))
            ##################

        return np.mean(losses)
