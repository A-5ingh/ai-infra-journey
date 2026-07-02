import pytest, numpy
from math_utils import *

@pytest.fixture
def m1():
    return np.array([[1, 2], [1, 1]])

@pytest.fixture
def m2():
    return np.array([[3, 4], [2, 2]])


def test_dot_product():
    actual = dot_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
    expected = 32
    
    assert actual == expected

def test_vector_norm(m1):
    actual = vector_norm(m1)
    expected = 2.6457513110645907
    
    assert actual == expected

def test_matrix_multuply(m1, m2):
    actual = matrix_multiply(m1, m2)
    expected = np.array([[7, 8], [5, 6]])
    
    np.testing.assert_array_equal(actual, expected)
    
    
# Day 3
def test_mean_by_row(m1, m2):
    actual = mean_by_row(m1)
    expected = np.array([1.5, 1.])
    
    np.testing.assert_array_equal(actual, expected)
    
    actual = mean_by_row(m2)
    expected = np.array([3.5, 2.])
    
    np.testing.assert_array_equal(actual, expected)
    
def test_normalize_columns(m2):
    actual = normalize_columns(m2)
    expected = np.array([[1., 1.], [-1., -1.]])
    
    np.testing.assert_array_equal(actual, expected)
    
def test_pairwise_l2_distances(m1):
    actual = pairwise_l2_distances(m1)
    expected = np.array([[0, 1.], [1., 0]])
    
    np.testing.assert_array_equal(actual, expected)