# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:25:51 2024

@author: cdarc
"""

from setuptools import setup, find_packages

setup(
    name='less_squares',
    version='0.1.4',
    description='A fast O(nm) iterative least squares solution',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Christopher D\'Arcy',
    author_email='chris@scigood.com',
    url='https://github.com/Christopher-DArcy/less_squares',
    license='GNU General Public License v3 (GPLv3)',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),  # Automatically finds packages
    python_requires='>=3.8',  # Restrict to Python 3.8 only
    install_requires=[
        'numpy',  # Add numpy as a dependency
    ],
    extras_require={
        'test': [
            'unittest; python_version >= "3.8"',
        ],
    },
)
