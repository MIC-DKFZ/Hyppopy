# this small script runs the unittest on our package and creates xml output.
# The xml output can be parsed by jenkins to create nicely formatted test output


if __name__ == '__main__':
    import xmlrunner
    import unittest
    runner = xmlrunner.XMLTestRunner('python_tests_xml')
    runner.run(unittest.TestLoader().discover('hyppopy'))
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
