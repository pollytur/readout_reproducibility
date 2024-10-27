#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="sensorium",
    version="0.0",
    description="readout reproducibility publication code, based on sensorium and divisive normalization",
    author="pollytur",
    packages=find_packages(exclude=[]),
    install_requires=[
#     change neuralpredictors and nnfabrik installations to the ones from git
#     add datajoint and git installations
        "neuralpredictors @ git+https://github.com/sinzlab/neuralpredictors.git@5e86a8222db48b6eab271e8ff672fa67b29012f8",
        "nnfabrik @ git+https://github.com/sinzlab/nnfabrik.git@29f22bc95841897d734532c02b77423e602ba21f",
        "scikit-image>=0.19.1",
        "lipstick",
        "numpy>=1.22.0",
        "datajoint",
        "GitPython"
    ],
)
