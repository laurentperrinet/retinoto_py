#!/usr/bin/env python
import pytest

"""Tests for `retinoto_py` package."""

from retinoto_py import get_preprocess
from retinoto_py import Params, get_idx_to_label


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyfeldroy/cookiecutter-pypackage')
    args = Params()
    preprocess = get_preprocess(args)





def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""

    args = Params()
    idx_to_label = get_idx_to_label(args)
    assert idx_to_label[0] == 'tench'