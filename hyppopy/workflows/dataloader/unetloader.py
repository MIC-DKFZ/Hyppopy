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
from collections import defaultdict
from hyppopy.workflows.dataloader.dataloaderbase import DataLoaderBase


class UnetDataLoaderBase(DataLoaderBase):

    def read(self, **kwargs):
        pass

    def subfiles(self, folder, join=True, prefix=None, suffix=None, sort=True):
        if join:
            l = os.path.join
        else:
            l = lambda x, y: y
        res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
               and (prefix is None or i.startswith(prefix))
               and (suffix is None or i.endswith(suffix))]
        if sort:
            res.sort()
        return res

    def preprocess(self, **kwargs):
        image_dir = os.path.join(kwargs['root_dir'], kwargs['image_dir'])
        label_dir = os.path.join(kwargs['root_dir'], kwargs['labels_dir'])
        output_dir = os.path.join(kwargs['root_dir'], kwargs['output_dir'])
        classes = kwargs['classes']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('Created' + output_dir + '...')

        class_stats = defaultdict(int)
        total = 0

        nii_files = self.subfiles(image_dir, suffix=".nii.gz", join=False)

        for i in range(0, len(nii_files)):
            if nii_files[i].startswith("._"):
                nii_files[i] = nii_files[i][2:]

        for f in nii_files:
            image, _ = load(os.path.join(image_dir, f))
            label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))

            print(f)

            for i in range(classes):
                class_stats[i] += np.sum(label == i)
                total += np.sum(label == i)

            image = (image - image.min()) / (image.max() - image.min())

            image = reshape(image, append_value=0, new_shape=(64, 64, 64))
            label = reshape(label, append_value=0, new_shape=(64, 64, 64))

            result = np.stack((image, label))

            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
            print(f)

        print(total)
        for i in range(classes):
            print(class_stats[i], class_stats[i] / total)
