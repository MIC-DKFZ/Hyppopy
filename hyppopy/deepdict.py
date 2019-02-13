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
import re
import json
import types
import pprint
import xmltodict
from dicttoxml import dicttoxml
from collections import OrderedDict

import logging
LOG = logging.getLogger('hyppopy')

from hyppopy.globals import DEEPDICT_XML_ROOT

def convert_ordered2std_dict(obj):
    """
    Helper function converting an OrderedDict into a standard lib dict.
    :param obj: [OrderedDict]
    """
    for key, value in obj.items():
        if isinstance(value, OrderedDict):
            obj[key] = dict(obj[key])
            convert_ordered2std_dict(obj[key])


def check_dir_existance(dirname):
    """
    Helper function to check if a directory exists, creating it if not.
    :param dirname: [str] full path of the directory to check
    """
    if not os.path.exists(dirname):
        os.mkdir(dirname)


class DeepDict(object):
    """
    The DeepDict class represents a nested dictionary with additional functionality compared to a standard
    lib dict. The data can be accessed and changed vie a pathlike access and dumped or read to .json/.xml files.

    Initializing instances using defaults creates an empty DeepDict. Using in_data enables to initialize the
    object instance with data, where in_data can be a dict, or a filepath to a json or xml file. Using path sep
    the appearance of path passing can be changed, a default data access via path would look like my_dd['target/section/path'] with path_sep='.' like so my_dd['target.section.path']

    :param in_data: [dict] or [str], input dict or filename
    :param path_sep: [str] path separator character
    """
    _data = None
    _sep = "/"

    def __init__(self, in_data=None, path_sep="/"):
        self.clear()
        self._sep = path_sep
        LOG.debug(f"path separator is: {self._sep}")
        if in_data is not None:
            if isinstance(in_data, str):
                self.from_file(in_data)
            elif isinstance(in_data, dict):
                self.data = in_data

    def __str__(self):
        """
        Enables print output for class instances, printing the instance data dict using pretty print
        :return: [str]
        """
        return pprint.pformat(self.data)

    def __eq__(self, other):
        """
        Overloads the == operator comparing the instance data dictionaries for equality
        :param other: [DeepDict] rhs
        :return: [bool]
        """
        return self.data == other.data

    def __getitem__(self, path):
        """
        Overloads the return of the [] operator for data access. This enables access the DeepDict instance like so:
        my_dd['target/section/path'] or my_dd[['target','section','path']]
        :param path: [str] or [list(str)], the path to the target data structure level/content
        :return: [object]
        """
        return DeepDict.get_from_path(self.data, path, self.sep)

    def __setitem__(self, path, value=None):
        """
        Overloads the setter for the [] operator for data assignment.
        :param path: [str] or [list(str)], the path to the target data structure level/content
        :param value: [object] rhs assignment object
        """
        if isinstance(path, str):
            path = path.split(self.sep)
        if not isinstance(path, list) or isinstance(path, tuple):
            raise IOError("Input Error, expect list[str] type for path")
        if len(path) < 1:
            raise IOError("Input Error, missing section strings")

        if not path[0] in self._data.keys():
            if value is not None and len(path) == 1:
                self._data[path[0]] = value
            else:
                self._data[path[0]] = {}

        tmp = self._data[path[0]]
        path.pop(0)
        while True:
            if len(path) == 0:
                break
            if path[0] not in tmp.keys():
                if value is not None and len(path) == 1:
                    tmp[path[0]] = value
                else:
                    tmp[path[0]] = {}
                    tmp = tmp[path[0]]
            else:
                tmp = tmp[path[0]]
            path.pop(0)

    def __len__(self):
        return len(self._data)

    def items(self):
        return self.data.items()

    def clear(self):
        """
        clears the instance data
        """
        LOG.debug("clear()")
        self._data = {}

    def from_file(self, fname):
        """
        Loads data from file. Currently implemented .json and .xml file reader.
        :param fname: [str] filename
        """
        if not isinstance(fname, str):
            raise IOError("Input Error, expect str type for fname")
        if fname.endswith(".json"):
            self.read_json(fname)
        elif fname.endswith(".xml"):
            self.read_xml(fname)
        else:
            LOG.error("Unknown filetype, expect [.json, .xml]")
            raise NotImplementedError("Unknown filetype, expect [.json, .xml]")

    def read_json(self, fname):
        """
        Read json file
        :param fname: [str] input filename
        """
        if not isinstance(fname, str):
            raise IOError("Input Error, expect str type for fname")
        if not os.path.isfile(fname):
            raise IOError(f"File {fname} not found!")
        LOG.debug(f"read_json({fname})")
        try:
            with open(fname, "r") as read_file:
                self._data = json.load(read_file)
            DeepDict.value_traverse(self.data, callback=DeepDict.parse_type)
        except Exception as e:
            LOG.error(f"Error while reading json file {fname} or while converting types")
            raise IOError("Error while reading json file {fname} or while converting types")

    def read_xml(self, fname):
        """
        Read xml file
        :param fname: [str] input filename
        """
        if not isinstance(fname, str):
            raise IOError("Input Error, expect str type for fname")
        if not os.path.isfile(fname):
            raise IOError(f"File {fname} not found!")
        LOG.debug(f"read_xml({fname})")
        try:
            with open(fname, "r") as read_file:
                xml = "".join(read_file.readlines())
                self._data = xmltodict.parse(xml, attr_prefix='')
            DeepDict.value_traverse(self.data, callback=DeepDict.parse_type)
        except Exception as e:
            msg = f"Error while reading xml file {fname} or while converting types"
            LOG.error(msg)
            raise IOError(msg)

        # if written with DeepDict, the xml contains a root node called
        # deepdict which should beremoved for consistency reasons
        if DEEPDICT_XML_ROOT in self._data.keys():
            self._data = self._data[DEEPDICT_XML_ROOT]
        self._data = dict(self.data)
        # convert the orderes dict structure to a default dict for consistency reasons
        convert_ordered2std_dict(self.data)

    def to_file(self, fname):
        """
        Write to file, type is determined by checking the filename ending.
        Currently implemented is writing to json and to xml.
        :param fname: [str] filename
        """
        if not isinstance(fname, str):
            raise IOError("Input Error, expect str type for fname")
        if fname.endswith(".json"):
            self.write_json(fname)
        elif fname.endswith(".xml"):
            self.write_xml(fname)
        else:
            LOG.error(f"Unknown filetype, expect [.json, .xml]")
            raise NotImplementedError("Unknown filetype, expect [.json, .xml]")

    def write_json(self, fname):
        """
        Dump data to json file.
        :param fname:  [str] filename
        """
        if not isinstance(fname, str):
            raise IOError("Input Error, expect str type for fname")
        check_dir_existance(os.path.dirname(fname))
        try:
            LOG.debug(f"write_json({fname})")
            with open(fname, "w") as write_file:
                json.dump(self.data, write_file)
        except Exception as e:
            LOG.error(f"Failed dumping to json file: {fname}")
            raise e

    def write_xml(self, fname):
        """
        Dump data to json file.
        :param fname:  [str] filename
        """
        if not isinstance(fname, str):
            raise IOError("Input Error, expect str type for fname")
        check_dir_existance(os.path.dirname(fname))
        xml = dicttoxml(self.data, custom_root=DEEPDICT_XML_ROOT, attr_type=False)
        LOG.debug(f"write_xml({fname})")
        try:
            with open(fname, "w") as write_file:
                write_file.write(xml.decode("utf-8"))
        except Exception as e:
            LOG.error(f"Failed dumping to xml file: {fname}")
            raise e

    def has_section(self, section):
        return DeepDict.has_key(self.data, section)

    @staticmethod
    def get_from_path(data, path, sep="/"):
        """
        Implements a nested dict access via a path like string like so path='target/section/path'
        which is equivalent to my_dict['target']['section']['path'].
        :param data: [dict] input dictionary
        :param path: [str] pathlike string
        :param sep: [str] path separator, default='/'
        :return: [object]
        """
        if not isinstance(data, dict):
            LOG.error("Input Error, expect dict type for data")
            raise IOError("Input Error, expect dict type for data")
        if isinstance(path, str):
            path = path.split(sep)
        if not isinstance(path, list) or isinstance(path, tuple):
            LOG.error(f"Input Error, expect list[str] type for path: {path}")
            raise IOError("Input Error, expect list[str] type for path")
        if not DeepDict.has_key(data, path[-1]):
            LOG.error(f"Input Error, section {path[-1]} does not exist in dictionary")
            raise IOError(f"Input Error, section {path[-1]} does not exist in dictionary")
        try:
            for k in path:
                data = data[k]
        except Exception as e:
            LOG.error(f"Failed retrieving data from path {path} due to {e}")
            raise LookupError(f"Failed retrieving data from path {path} due to {e}")
        return data

    @staticmethod
    def has_key(data, section, already_found=False):
        """
        Checks if input dictionary has a key called section. The already_found parameter
        is for internal recursion checks.
        :param data: [dict] input dictionary
        :param section: [str] key string to search for
        :param already_found: recursion criteria check
        :return: [bool] section found
        """
        if not isinstance(data, dict):
            LOG.error("Input Error, expect dict type for obj")
            raise IOError("Input Error, expect dict type for obj")
        if not isinstance(section, str):
            LOG.error(f"Input Error, expect dict type for obj {section}")
            raise IOError(f"Input Error, expect dict type for obj {section}")
        if already_found:
            return True
        found = False
        for key, value in data.items():
            if key == section:
                found = True
            if isinstance(value, dict):
                found = DeepDict.has_key(data[key], section, found)
        return found

    @staticmethod
    def value_traverse(data, callback=None):
        """
        Dictionary filter function, walks through the input dict (obj) calling the callback function for each value.
        The callback function return is assigned the the corresponding dict value.
        :param data: [dict] input dictionary
        :param callback:
        """
        if not isinstance(data, dict):
            LOG.error("Input Error, expect dict type for obj")
            raise IOError("Input Error, expect dict type for obj")
        if not isinstance(callback, types.FunctionType):
            LOG.error("Input Error, expect function type for callback")
            raise IOError("Input Error, expect function type for callback")
        for key, value in data.items():
            if isinstance(value, dict):
                DeepDict.value_traverse(data[key], callback)
            else:
                data[key] = callback(value)

    def transfer_attrs(self, cls, target_section):
        def set(item):
            setattr(cls, item[0], item[1])
        DeepDict.sectionconstraint_item_traverse(self.data, target_section, callback=set, section=None)

    @staticmethod
    def sectionconstraint_item_traverse(data, target_section, callback=None, section=None):
        """
        Dictionary filter function, walks through the input dict (obj) calling the callback function for each item.
        The callback function then is called with the key value pair as tuple input but only for the target section.
        :param data: [dict] input dictionary
        :param callback:
        """
        if not isinstance(data, dict):
            LOG.error("Input Error, expect dict type for obj")
            raise IOError("Input Error, expect dict type for obj")
        if not isinstance(callback, types.FunctionType):
            LOG.error("Input Error, expect function type for callback")
            raise IOError("Input Error, expect function type for callback")
        for key, value in data.items():
            if isinstance(value, dict):
                DeepDict.sectionconstraint_item_traverse(data[key], target_section, callback, key)
            else:
                if target_section == section:
                    callback((key, value))

    @staticmethod
    def item_traverse(data, callback=None):
        """
        Dictionary filter function, walks through the input dict (obj) calling the callback function for each item.
        The callback function then is called with the key value pair as tuple input.
        :param data: [dict] input dictionary
        :param callback:
        """
        if not isinstance(data, dict):
            LOG.error("Input Error, expect dict type for obj")
            raise IOError("Input Error, expect dict type for obj")
        if not isinstance(callback, types.FunctionType):
            LOG.error("Input Error, expect function type for callback")
            raise IOError("Input Error, expect function type for callback")
        for key, value in data.items():
            if isinstance(value, dict):
                DeepDict.value_traverse(data[key], callback)
            else:
                callback((key, value))

    @staticmethod
    def parse_type(string):
        """
        Type convert input string to float, int, list, tuple or string
        :param string: [str] input string
        :return: [T] converted output
        """
        try:
            a = float(string)
            try:
                b = int(string)
            except ValueError:
                return float(string)
            if a == b:
                return b
            return a
        except ValueError:
            if string.startswith("[") and string.endswith("]"):
                string = re.sub(' ', '', string)
                elements = string[1:-1].split(",")
                li = []
                for e in elements:
                    li.append(DeepDict.parse_type(e))
                return li
            elif string.startswith("(") and string.endswith(")"):
                elements = string[1:-1].split(",")
                li = []
                for e in elements:
                    li.append(DeepDict.parse_type(e))
                return tuple(li)
            return string

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, dict):
            LOG.error(f"Input Error, expect dict type for value, but got {type(value)}")
            raise IOError(f"Input Error, expect dict type for value, but got {type(value)}")
        self.clear()
        self._data = value

    @property
    def sep(self):
        return self._sep

    @sep.setter
    def sep(self, value):
        if not isinstance(value, str):
            LOG.error(f"Input Error, expect str type for value, but got {type(value)}")
            raise IOError(f"Input Error, expect str type for value, but got {type(value)}")
        self._sep = value
