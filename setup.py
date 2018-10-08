#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:08:08 2018

@author: Kyunghan
"""
from setuptools import setup, find_packages, Extension

setup(
    name="pysim",
    packages=find_packages(),
    version="1.0.0",
    description="python simulation environment",
    author='Kyunghan Min',
    author_email='kyunghah.min@gmail.com',
    keywords=['EV', 'Prediction', 'Control'],
    install_requires=['matplotlib'],
    entry_points={'console_scripts': ['pysim = pysim.__main__:main']},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Spyder',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.5',
    ]
)
