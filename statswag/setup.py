import os
from os.path import join

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('statswag', parent_package, top_path)

    # submodules which have their own setup.py
    config.add_subpackage('datasets')
    config.add_subpackage('estimators')

    # submodules without their own setup.py
    # meaning you would have to manually add subfolders if they existed
    config.add_subpackage('utils')
    config.add_subpackage('metrics')
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
