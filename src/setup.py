#!/usr/bin/env python
"""Library setup script."""

import os
from setuptools import find_packages, setup

setup(
    name=os.environ["PKG_NAME"],
    version=os.environ["PKG_VERSION"],
    description="My Python library project",
    author="Physical AI QA Demo",
    packages=find_packages(exclude=["contrib", "docs", "test"]),
    # Please specify your dependencies in conda_recipe/meta.yaml instead.
    install_requires=[],
)