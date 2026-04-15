#!/usr/bin/env python
"""Library setup script."""

import os
from setuptools import find_packages, setup

setup(
    name=os.environ.get("PKG_NAME", "qa-cell-edge-agent"),
    version=os.environ.get("PKG_VERSION", "0.1.0"),
    description="Jetson Nano edge agent for the Physical AI QA Cell",
    author="Physical AI QA Demo",
    packages=find_packages(exclude=["contrib", "docs", "test", "scripts"]),
    python_requires=">=3.8",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "qa-cell-agent=qa_cell_edge_agent.main:main",
        ],
    },
)
