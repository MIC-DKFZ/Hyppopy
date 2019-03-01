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

import unittest
import numpy as np

from hyppopy.plugins.gridsearch_settings_plugin import gridsearch_SettingsParticle
from hyppopy.plugins.gridsearch_settings_plugin import gridsearch_Settings


class ProjectManagerTestSuite(unittest.TestCase):

    def setUp(self):
        self.hp = {
            'UniformFloat': {
                'domain': 'uniform',
                'data': [0, 1, 10],
                'type': 'float',
            },
            'UniformInt': {
                'domain': 'uniform',
                'data': [0, 7, 10],
                'type': 'int',
            },
            'NormalFloat': {
                'domain': 'normal',
                'data': [0, 1, 10],
                'type': 'float',
            },
            'NormalInt': {
                'domain': 'normal',
                'data': [0, 10, 10],
                'type': 'int',
            },
            'LogFloat': {
                'domain': 'loguniform',
                'data': [0.01, np.e, 10],
                'type': 'float',
            },
            'LogFloat': {
                'domain': 'loguniform',
                'data': [0.01, np.e, 10],
                'type': 'float',
            },
            'LogInt': {
                'domain': 'loguniform',
                'data': [0, 1000000, 10],
                'type': 'int',
            },
            'CategoricalStr': {
                'domain': 'categorical',
                'data': ['a', 'b'],
                'type': 'str',
            },
            'CategoricalInt': {
                'domain': 'categorical',
                'data': [0, 1],
                'type': 'int',
            }
        }


        self.truth = {
            'UniformFloat': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'UniformInt': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'NormalFloat': [0.0, 0.2592443381276233, 0.3673134565097225, 0.4251586871937128, 0.4649150940720099, 0.5,
               0.5350849059279901, 0.5748413128062873, 0.6326865434902775, 0.7407556618723767, 1.0],
            'NormalInt': [0, 3, 4, 5, 6, 7, 10],
            'LogFloat': [0.010000000000000004, 0.017515778645640943, 0.030680250156309114, 0.053738847053080116,
                        0.0941277749653705, 0.16487212707001322, 0.28878636825943366, 0.5058318102310787,
                        0.8860038019931427, 1.551904647490817, 2.7182818284590575],
            'LogInt': [0, 2, 1259, 1000000],
            'CategoricalStr': ['a', 'b'],
            'CategoricalInt': [0, 1]
        }

    def test_gridsearch_settings(self):
        gss = gridsearch_Settings()
        gss.set_hyperparameter(self.hp)
        res = gss.get_hyperparameter()
        self.assertTrue('CategoricalInt' in res.keys())
        self.assertTrue(len(res) == 1)
        self.assertTrue(0 in res['CategoricalInt'].keys())
        self.assertTrue(1 in res['CategoricalInt'].keys())
        self.assertTrue(len(res['CategoricalInt']) == 2)
        self.assertTrue('a' in res['CategoricalInt'][0]['CategoricalStr'].keys())
        self.assertTrue('b' in res['CategoricalInt'][0]['CategoricalStr'].keys())
        self.assertTrue(len(res['CategoricalInt'][0]['CategoricalStr']) == 2)
        self.assertTrue('a' in res['CategoricalInt'][1]['CategoricalStr'].keys())
        self.assertTrue('b' in res['CategoricalInt'][1]['CategoricalStr'].keys())
        self.assertTrue(len(res['CategoricalInt'][1]['CategoricalStr']) == 2)

        def check_truth(input_dict):
            for key, value in self.truth.items():
                if not key.startswith('Categorical'):
                    self.assertTrue(key in input_dict.keys())
                    if key == 'LogFloat':
                        a=0
                    if key == 'LogInt':
                        a=0
                    for n, v in enumerate(self.truth[key]):
                        self.assertAlmostEqual(v, input_dict[key][n])

        check_truth(res['CategoricalInt'][0]['CategoricalStr']['a'])
        check_truth(res['CategoricalInt'][1]['CategoricalStr']['a'])
        check_truth(res['CategoricalInt'][0]['CategoricalStr']['b'])
        check_truth(res['CategoricalInt'][1]['CategoricalStr']['b'])

    def test_gridsearch_particle(self):
        for name, data in self.hp.items():
            gsp = gridsearch_SettingsParticle(name=name,
                                              domain=data['domain'],
                                              dtype=data['type'],
                                              data=data['data'])
            data = gsp.get()
            for n in range(len(self.truth[name])):
                self.assertAlmostEqual(data[n], self.truth[name][n])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
