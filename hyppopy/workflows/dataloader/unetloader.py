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
import numpy as np
from medpy.io import load
from collections import defaultdict
from .dataloaderbase import DataLoaderBase


class UnetDataLoader(DataLoaderBase):

    def read(self, **kwargs):
        # preprocess data if not already done
        root_dir = os.path.join(kwargs['data_path'], kwargs['data_name'])
        split_dir = os.path.join(kwargs['data_path'], kwargs['split_dir'])
        preproc_dir = os.path.join(root_dir, 'preprocessed')
        if not os.path.isdir(preproc_dir):
            self.preprocess_data(root=root_dir,
                                 image_dir=kwargs['image_dir'],
                                 labels_dir=kwargs['labels_dir'],
                                 output_dir=preproc_dir,
                                 classes=kwargs['num_classes'])
            self.data = self.create_splits(output_dir=split_dir, image_dir=preproc_dir)
        else:
            with open(os.path.join(split_dir, "splits.pkl"), 'rb') as f:
                self.data = pickle.load(f)

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

    def reshape(self, orig_img, append_value=-1024, new_shape=(512, 512, 512)):
        reshaped_image = np.zeros(new_shape)
        reshaped_image[...] = append_value
        x_offset = 0
        y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
        z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

        reshaped_image[x_offset:orig_img.shape[0] + x_offset, y_offset:orig_img.shape[1] + y_offset,
        z_offset:orig_img.shape[2] + z_offset] = orig_img

        return reshaped_image

    def preprocess_data(self, root, image_dir, labels_dir, output_dir, classes):
        image_dir = os.path.join(root, image_dir)
        label_dir = os.path.join(root, labels_dir)
        output_dir = os.path.join(root, output_dir)
        classes = classes

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

            image = self.reshape(image, append_value=0, new_shape=(64, 64, 64))
            label = self.reshape(label, append_value=0, new_shape=(64, 64, 64))

            result = np.stack((image, label))

            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
            print(f)

        print(total)
        for i in range(classes):
            print(class_stats[i], class_stats[i] / total)

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

    def create_splits(self, output_dir, image_dir):
        npy_files = self.subfiles(image_dir, suffix=".npy", join=False)

        trainset_size = len(npy_files)*50//100
        valset_size = len(npy_files)*25//100
        testset_size = len(npy_files)*25//100

        splits = []
        for split in range(0, 5):
            image_list = npy_files.copy()
            trainset = []
            valset = []
            testset = []
            for i in range(0, trainset_size):
                patient = np.random.choice(image_list)
                image_list.remove(patient)
                trainset.append(patient[:-4])
            for i in range(0, valset_size):
                patient = np.random.choice(image_list)
                image_list.remove(patient)
                valset.append(patient[:-4])
            for i in range(0, testset_size):
                patient = np.random.choice(image_list)
                image_list.remove(patient)
                testset.append(patient[:-4])
            split_dict = dict()
            split_dict['train'] = trainset
            split_dict['val'] = valset
            split_dict['test'] = testset

            splits.append(split_dict)

        with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
            pickle.dump(splits, f)
        return splits





