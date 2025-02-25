"""Test composition"""

import pytest

from meshed.composition import line_with_dag


def test_line_with_dag():
    """
    Very simple test of a basic usage of line_with_dag
    """

    def f(x):
        return x + 1

    def g(y):
        return 2 * y

    d = line_with_dag(f, g)
    assert d(2) == 6
