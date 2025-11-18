"""Test suite for the Category Encoder"""

import pytest

from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def category_instance():
    """Fixture to create a Category encoder instance for tests"""


def test_category_initialization():
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    assert isinstance(e, CategoryEncoder)


def test_encode_us():
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    a = SDR([1, 12])
    e.encode("US", a)
    assert a == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_unknown_category():
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    a = SDR([1, 12])
    e.encode("NA", a)
    assert a == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_encode_es():
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    a = SDR([1, 12])
    e.encode("ES", a)
    assert a == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_with_width_one():
    categories = ["cat1", "cat2", "cat3", "cat4", "cat5"]
    """Note: I think since width is 1, each category is 1 bit and there is the first bit that is the unknown category."""
    expected = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    parameters = CategoryParameters(w=1, categoryList=categories, forced=True)
    e = CategoryEncoder(parameters=parameters)
    i = 0
    for cat in categories:
        a = SDR([1, 6])
        e.encode(cat, a)
        assert a == expected[i]
        i = i + 1
