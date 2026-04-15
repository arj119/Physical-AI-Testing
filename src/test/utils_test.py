# Unit tests are disabled by default. Please refer to the README.md for instructions on how to enable them.
# import pytest
from myproject.utils import add_one


def test_add_one():
    assert add_one(1) == 2
