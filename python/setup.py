from setuptools import setup, find_packages

setup(
    name='e2vid',
    version='0.1',
    packages=['e2vid', 'e2vid/base', 'e2vid/model', 'e2vid/options', 'e2vid/utils'],
    package_dir={'':'reconstruction'}
)
