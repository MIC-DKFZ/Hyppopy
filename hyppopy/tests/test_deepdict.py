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
import unittest

from hyppopy.deepdict.deepdict import DeepDict


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class DeepDictTestSuite(unittest.TestCase):

    def setUp(self):
            self.test_data = {
                'widget': {
                    'debug': 'on',
                    'image': {'alignment': 'center',
                              'hOffset': 250,
                              'name': 'sun1',
                              'src': 'Images/Sun.png',
                              'vOffset': 250},
                    'text': {'alignment': 'center',
                             'data': 'Click Here',
                             'hOffset': 250,
                             'name': 'text1',
                             'onMouseUp': 'sun1.opacity = (sun1.opacity / 100) * 90;',
                             'size': 36,
                             'style': 'bold',
                             'vOffset': 100},
                    'window': {'height': 500,
                               'name': 'main_window',
                               'title': 'Sample Konfabulator Widget',
                               'width': 500}
                }
            }

            self.test_data2 = {"test": {
                                "section": {
                                    "var1": 100,
                                    "var2": 200
                                }
            }}

    def test_fileIO(self):
        dd_json = DeepDict(os.path.join(DATA_PATH, 'test_json.json'))
        dd_xml = DeepDict(os.path.join(DATA_PATH, 'test_xml.xml'))
        dd_dict = DeepDict(self.test_data)

        self.assertTrue(list(self.test_data.keys())[0] == list(dd_json.data.keys())[0])
        self.assertTrue(list(self.test_data.keys())[0] == list(dd_xml.data.keys())[0])
        self.assertTrue(list(self.test_data.keys())[0] == list(dd_dict.data.keys())[0])
        for key in self.test_data['widget'].keys():
            self.assertTrue(self.test_data['widget'][key] == dd_json.data['widget'][key])
            self.assertTrue(self.test_data['widget'][key] == dd_xml.data['widget'][key])
            self.assertTrue(self.test_data['widget'][key] == dd_dict.data['widget'][key])
        for key in self.test_data['widget'].keys():
            if key == 'debug':
                self.assertTrue(dd_json.data['widget']["debug"] == "on")
                self.assertTrue(dd_xml.data['widget']["debug"] == "on")
                self.assertTrue(dd_dict.data['widget']["debug"] == "on")
            else:
                for key2, value2 in self.test_data['widget'][key].items():
                    self.assertTrue(value2 == dd_json.data['widget'][key][key2])
                    self.assertTrue(value2 == dd_xml.data['widget'][key][key2])
                    self.assertTrue(value2 == dd_dict.data['widget'][key][key2])

        dd_dict.to_file(os.path.join(DATA_PATH, 'write_to_json_test.json'))
        dd_dict.to_file(os.path.join(DATA_PATH, 'write_to_xml_test.xml'))
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, 'write_to_json_test.json')))
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, 'write_to_xml_test.xml')))
        dd_json = DeepDict(os.path.join(DATA_PATH, 'write_to_json_test.json'))
        dd_xml = DeepDict(os.path.join(DATA_PATH, 'write_to_xml_test.xml'))
        self.assertTrue(dd_json == dd_dict)
        self.assertTrue(dd_xml == dd_dict)
        try:
            os.remove(os.path.join(DATA_PATH, 'write_to_json_test.json'))
            os.remove(os.path.join(DATA_PATH, 'write_to_xml_test.xml'))
        except Exception as e:
            print(e)
            print("Warning: Failed to delete temporary data during tests!")

    def test_has_section(self):
        dd = DeepDict(self.test_data)
        self.assertTrue(DeepDict.has_section(dd.data, 'hOffset'))
        self.assertTrue(DeepDict.has_section(dd.data, 'window'))
        self.assertTrue(DeepDict.has_section(dd.data, 'widget'))
        self.assertFalse(DeepDict.has_section(dd.data, 'notasection'))

    def test_data_access(self):
        dd = DeepDict(self.test_data)
        self.assertEqual(dd['widget/window/height'], 500)
        self.assertEqual(dd['widget/image/name'], 'sun1')
        self.assertTrue(isinstance(dd['widget/window'], dict))
        self.assertEqual(len(dd['widget/window']), 4)

        dd = DeepDict(path_sep=".")
        dd.data = self.test_data
        self.assertEqual(dd['widget.window.height'], 500)
        self.assertEqual(dd['widget.image.name'], 'sun1')
        self.assertTrue(isinstance(dd['widget.window'], dict))
        self.assertEqual(len(dd['widget.window']), 4)

    def test_data_adding(self):
        dd = DeepDict()
        dd["test/section/var1"] = 100
        dd["test/section/var2"] = 200
        self.assertTrue(dd.data == self.test_data2)

        dd = DeepDict()
        dd["test"] = {}
        dd["test/section"] = {}
        dd["test/section/var1"] = 100
        dd["test/section/var2"] = 200
        self.assertTrue(dd.data == self.test_data2)

    def test_sample_space(self):
        dd = DeepDict(os.path.join(DATA_PATH, 'test_paramset.json'))
        self.assertEqual(len(dd[['parameter', 'activation', 'data']]), 4)
        self.assertEqual(dd['parameter/activation/data'], ['ReLU', 'tanh', 'sigm', 'ELU'])
        self.assertTrue(isinstance(dd['parameter/activation/data'], list))
        self.assertTrue(isinstance(dd['parameter/activation/data'][0], str))
        self.assertEqual(dd['parameter/layerdepth/data'], [3, 20])
        self.assertTrue(isinstance(dd['parameter/layerdepth/data'], list))
        self.assertTrue(isinstance(dd['parameter/layerdepth/data'][0], int))
        self.assertTrue(isinstance(dd['parameter/learningrate/data'][0], float))
        self.assertEqual(dd['parameter/learningrate/data'][0], 1e-5)
        self.assertEqual(dd['parameter/learningrate/data'][1], 10.0)

    def test_len(self):
        dd = DeepDict(os.path.join(DATA_PATH, 'test_paramset.json'))
        self.assertEqual(len(dd), 1)

if __name__ == '__main__':
    unittest.main()
