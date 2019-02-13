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
import pickle
import fnmatch
import numpy as np
from torch import nn
from medpy.io import load
from collections import defaultdict
from abc import ABCMeta, abstractmethod


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            input = input.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:]  # this is the case when use_bg=False
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(tp.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights
    result = (- ((2 * tp + smooth_in_nom) / (2 * tp + fp + fn + smooth)) * weights).mean()
    return result


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth)) * weights).mean()  #TODO: Was ist weights and er Stelle?
    return result


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True, background_weight=1, rebalance_weights=None):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        if not do_bg:
            assert background_weight == 1, "if there is no bg, then set background weight to 1 you dummy"
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        self.smooth_in_nom = smooth_in_nom
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None
        if not smooth_in_nom:
            self.nom_smooth = 0
        else:
            self.nom_smooth = smooth

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        shp_y = y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively
        y_onehot = torch.zeros(shp_x)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError("nah son")
            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        else:
            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                      background_weight=self.background_weight,
                                      rebalance_weights=self.rebalance_weights)
        return l


def load_dataset(base_dir, pattern='*.npy', slice_offset=5, keys=None):
    fls = []
    files_len = []
    slices_ax = []

    for root, dirs, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename[:-4] in keys:
                npy_file = os.path.join(root, filename)
                numpy_array = np.load(npy_file, mmap_mode="r")

                fls.append(npy_file)
                files_len.append(numpy_array.shape[1])

                slices_ax.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])

                i += 1

    return fls, files_len, slices_ax,


class SlimDataLoaderBase(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        """
        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.
        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()
        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!
        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 seed=None, file_pattern='*.npy', label_slice=1, input_slice=(0,), keys=None):

        self.files, self.file_len, self.slices = load_dataset(base_dir=base_dir, pattern=file_pattern, slice_offset=0, keys=keys, )
        super(NumpyDataLoader, self).__init__(self.slices, batch_size, num_batches)

        self.batch_size = batch_size

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.slice_idxs = list(range(0, len(self.slices)))

        self.data_len = len(self.slices)

        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.np_data = np.asarray(self.slices)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.slice_idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):
        slice_idxs = self.slice_idxs
        data_len = len(self.slices)
        np_data = self.np_data

        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = slice_idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr = np_data[idxs]

        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):
        data = []
        fnames = []
        slice_idxs = []
        labels = []

        for slice in open_array:
            fn_name = self.files[slice[0]]

            numpy_array = np.load(fn_name, mmap_mode="r")

            numpy_slice = numpy_array[ :, slice[1], ]
            data.append(numpy_slice[None, self.input_slice[0]])   # 'None' keeps the dimension

            if self.label_slice is not None:
                labels.append(numpy_slice[None, self.label_slice[0]])   # 'None' keeps the dimension

            fnames.append(self.files[slice[0]])
            slice_idxs.append(slice[1])

        ret_dict = {'data': np.asarray(data), 'fnames': fnames, 'slice_idxs': slice_idxs}
        if self.label_slice is not None:
            ret_dict['seg'] = np.asarray(labels)

        return ret_dict


class NumpyDataSet(object):
    """
    TODO
    """
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None, num_processes=8, num_cached_per_queue=8 * 4, target_size=128,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), do_reshuffle=True, keys=None):

        data_loader = NumpyDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                      input_slice=input_slice, label_slice=label_slice, keys=keys)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)



def reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0] + x_offset, y_offset:orig_img.shape[1] + y_offset,
    z_offset:orig_img.shape[2] + z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
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


def preprocess_data(root_dir):
    print("preprocess data...")
    image_dir = os.path.join(root_dir, 'imagesTr')
    print(f"image_dir: {image_dir}")
    label_dir = os.path.join(root_dir, 'labelsTr')
    print(f"label_dir: {label_dir}")
    output_dir = os.path.join(root_dir, 'preprocessed')
    print(f"output_dir: {output_dir} ... ", end="")
    classes = 3

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("created!")
    else:
        print("found!\npreprocessed data already available, aborted preprocessing!")
        return False

    print("start preprocessing ... ", end="")
    class_stats = defaultdict(int)
    total = 0

    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    for i in range(0, len(nii_files)):
        if nii_files[i].startswith("._"):
            nii_files[i] = nii_files[i][2:]

    for i, f in enumerate(nii_files):
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))

        for i in range(classes):
            class_stats[i] += np.sum(label == i)
            total += np.sum(label == i)

        image = (image - image.min()) / (image.max() - image.min())

        image = reshape(image, append_value=0, new_shape=(64, 64, 64))
        label = reshape(label, append_value=0, new_shape=(64, 64, 64))

        result = np.stack((image, label))

        np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
    print("finished!")
    return True


def create_splits(output_dir, image_dir):
    print("creating splits ... ", end="")
    npy_files = subfiles(image_dir, suffix=".npy", join=False)

    trainset_size = len(npy_files) * 50 // 100
    valset_size = len(npy_files) * 25 // 100
    testset_size = len(npy_files) * 25 // 100

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
    print("finished!")
