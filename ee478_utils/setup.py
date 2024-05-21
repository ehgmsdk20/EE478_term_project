from setuptools import find_packages
from distutils.core import setup

setup(
    name='ee478_utils',
    version='1.0.0',
    author='Urban Robotics Lab',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Utilities for EE478 final project',
    install_requires=['isaacgym']
)