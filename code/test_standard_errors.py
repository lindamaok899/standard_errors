import pytest
import pickle
from numpy.testing import assert_array_almost_equal as aaae
import standard_errors as se
from itertools import product


def get_expected_covariance(model, cov_method):
    """Load expected covariance matrix.

    Args:
        model (str): one of ['logit', 'probit']
        cov_method (str): one of ['jacobian', 'hessian', 'sandwich']

    Returns:
        expected_covariance

    """
    with open('test_fixtures/{}_{}.pickle'.format(model, cov_method), 'rb') as f:
        expected_cov = pickle.load(f)
    return expected_cov


def get_input(model, input_types):
    """Load the inputs.

    Args:
        model (str): one of ['logit', 'probit']
        input_types (list): can contain the elements 'jacobian' and 'hessian'

    Returns:
        inputs (dict): The inputs for the covariance function

    """
    inputs = {}
    for typ in input_types:
        with open('test_fixtures/{}_{}_matrix.pickle'.format(model, typ), 'rb') as f:
            input_matrix = pickle.load(f)
        inputs[typ] = input_matrix
    return inputs


models = ['probit', 'logit']
methods = ['jacobian', 'hessian', 'sandwich']
test_cases = list(product(models, methods))


@pytest.mark.parametrize('model, method', test_cases)
def test_cov_function(model, method):
    expected = get_expected_covariance(model, method)

    if method in ['jacobian', 'hessian']:
        input_types = [method]
    elif method == 'sandwich':
        input_types = ['jacobian', 'hessian']

    inputs = get_input(model, input_types)

    calculated = getattr(se, 'cov_{}'.format(method))(**inputs)

    aaae(calculated, expected)


