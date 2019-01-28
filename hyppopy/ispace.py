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

import abc
import dpath
import dpath.util

import logging
LOG = logging.getLogger('hyppopy')


class ISpace(object, metaclass=abc.ABCMeta):
    _data = {}

    @abc.abstractmethod
    def convert(self, *args, **kwargs):
        raise NotImplementedError('users must define convert to use this base class')

    def clear(self):
        LOG.debug("clear()")
        self._data.clear()

    def get_section(self, path):
        if isinstance(path, str):
            try:
                return dpath.util.get(self._data, path)
            except Exception as e:
                LOG.error(f"path mismatch exception: {e}")
                raise IOError("path mismatch exception")
        elif isinstance(path, list) or isinstance(path, tuple):
            try:
                return dpath.util.get(self._data, "/".join(path))
            except Exception as e:
                LOG.error(f"path list content exception: {e}")
                raise IOError("path list content exception")
        else:
            LOG.error("unknown path type")
            raise IOError("unknown path type")

    def set(self, data):
        LOG.debug(f"set({data})")
        self._data = data

    def add_section(self, name, section=None):
        LOG.debug(f"add_section({name}, {section})")
        if section is None:
            self._data[name] = {}

    def add_entry(self, name, value, section=None):
        LOG.debug(f"add_entry({name}, {value}, {section})")

    def read_json(self, filename):
        LOG.debug(f"read_json({filename})")

    def write_json(self, filename):
        LOG.debug(f"write_json({filename})")

    def read_xml(self, filename):
        LOG.debug(f"read_xml({filename})")

    def write_xml(self, filename):
        LOG.debug(f"write_xml({filename})")

    def read(self, filename):
        LOG.debug(f"read({filename})")

    def write(self, filename):
        LOG.debug(f"write({filename})")
