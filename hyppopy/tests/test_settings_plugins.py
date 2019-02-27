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
                'data': [-5, 5, 10],
                'type': 'float',
            },
            'LogFloat': {
                'domain': 'loguniform',
                'data': [-5, 5, 10],
                'type': 'float',
            },
            'LogInt': {
                'domain': 'loguniform',
                'data': [0, 6, 10],
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
            'LogFloat': [0.006737946999085467, 0.01831563888873418, 0.049787068367863944, 0.1353352832366127, 0.36787944117144233,
               1.0, 2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766],
            'LogInt': [1, 2, 3, 6, 11, 20, 37, 67, 122, 221, 403],
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
