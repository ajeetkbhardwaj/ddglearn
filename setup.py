"""Setup shim for legacy tooling.

All package metadata lives in pyproject.toml (PEP 621).
This file exists only so `pip install -e .` and older build
frontends that require setup.py keep working.
"""
from setuptools import setup

setup()
