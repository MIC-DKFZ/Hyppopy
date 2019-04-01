import os
from hyppopy.globals import ROOT


def create_readmesnippeds():
    fname = os.path.join(ROOT, "README.md")
    f = open(fname, "r")
    codes = []
    snipped = None

    for line in f.readlines():
        if snipped is not None:
            snipped.append("\t\t{}".format(line))
        if line.startswith("```"):
            if line.startswith("```python"):
                snipped = []
            else:
                if snipped is not None:
                    snipped.pop(-1)
                    codes.append(snipped)
                    snipped = None




    for n, snipped in enumerate(codes):
        f = open(os.path.join(ROOT, *("hyppopy", "tests", "test_snipped_{}.py".format(str(n).zfill(3)))), "w")
        test_code = "# DKFZ\n"
        test_code += "#\n"
        test_code += "#\n"
        test_code += "# Copyright (c) German Cancer Research Center,\n"
        test_code += "# Division of Medical Image Computing.\n"
        test_code += "# All rights reserved.\n"
        test_code += "#\n"
        test_code += "# This software is distributed WITHOUT ANY WARRANTY; without\n"
        test_code += "# even the implied warranty of MERCHANTABILITY or FITNESS FOR\n"
        test_code += "# A PARTICULAR PURPOSE.\n"
        test_code += "#\n"
        test_code += "# See LICENSE\n\n"
        test_code += "import os\n"
        test_code += "import unittest\n\n"
        test_code += "class ReadmeSnipped_{}TestSuite(unittest.TestCase):\n\n".format(str(n).zfill(3))
        test_code += "\tdef test_scripts(self):\n\n"
        snipped.insert(0, test_code)
        snipped.append("\t\tself.assertTrue(True)\n")
        f.writelines(snipped)
        f.close()

create_readmesnippeds()
