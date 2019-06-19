import pytest
import pickle
from numpy.testing import assert_array_almost_equal as aaae
from standard_errors import cov_hessian, cov_jacobian, cov_sandwich


@pytest.fixture
def setup_covariance_matrices():

    output_dict = {}
    with open("test_fixtures/probit_hessian_matrix.pickle", "rb") as f:
        output_dict["probit_hessian_matrix"] = pickle.load(f)

    with open("test_fixtures/probit_jacobian_matrix.pickle", "rb") as f:
        output_dict["probit_jacobian_matrix"] = pickle.load(f)

    with open("test_fixtures/logit_hessian_matrix.pickle", "rb") as f:
        output_dict["logit_hessian_matrix"] = pickle.load(f)

    with open("test_fixtures/logit_jacobian_matrix.pickle", "rb") as f:
        output_dict["logit_jacobian_matrix"] = pickle.load(f)

    return output_dict


@pytest.fixture
def expected_covariance_matrices():

    output_dict = {}
    with open("test_fixtures/probit_jacobian.pickle", "rb") as f:
        output_dict["probit_jacobian_cov"] = pickle.load(f)

    with open("test_fixtures/probit_hessian.pickle", "rb") as f:
        output_dict["probit_hessian_cov"] = pickle.load(f)

    with open("test_fixtures/probit_sandwich.pickle", "rb") as f:
        output_dict["probit_sandwich_cov"] = pickle.load(f)

    with open("test_fixtures/logit_jacobian.pickle", "rb") as f:
        output_dict["logit_jacobian_cov"] = pickle.load(f)

    with open("test_fixtures/logit_hessian.pickle", "rb") as f:
        output_dict["logit_hessian_cov"] = pickle.load(f)

    with open("test_fixtures/logit_sandwich.pickle", "rb") as f:
        output_dict["logit_sandwich_cov"] = pickle.load(f)

    return output_dict


def test_cov_hessian_with_probit(
    setup_covariance_matrices, expected_covariance_matrices
):
    inputs, outputs = setup_covariance_matrices, expected_covariance_matrices
    covariance_matrix = cov_hessian(inputs["probit_hessian_matrix"])
    aaae(covariance_matrix, outputs["probit_hessian_cov"])


def test_cov_hessian_with_logit(
    setup_covariance_matrices, expected_covariance_matrices
):
    inputs, outputs = setup_covariance_matrices, expected_covariance_matrices
    covariance_matrix = cov_hessian(inputs["logit_hessian_matrix"])
    aaae(covariance_matrix, outputs["logit_hessian_cov"])


def test_cov_jacobian_with_probit(
    setup_covariance_matrices, expected_covariance_matrices
):
    inputs, outputs = setup_covariance_matrices, expected_covariance_matrices
    covariance_matrix = cov_jacobian(inputs["probit_jacobian_matrix"])
    aaae(covariance_matrix, outputs["probit_jacobian_cov"])


def test_cov_jacobian_with_logit(
    setup_covariance_matrices, expected_covariance_matrices
):
    inputs, outputs = setup_covariance_matrices, expected_covariance_matrices
    covariance_matrix = cov_jacobian(inputs["logit_jacobian_matrix"])
    aaae(covariance_matrix, outputs["logit_jacobian_cov"])


def test_cov_sandwich_with_probit(
    setup_covariance_matrices, expected_covariance_matrices
):
    inputs, outputs = setup_covariance_matrices, expected_covariance_matrices
    covariance_matrix = cov_sandwich(
        inputs["probit_jacobian_matrix"], inputs["probit_hessian_matrix"]
    )
    aaae(covariance_matrix, outputs["probit_sandwich_cov"])


def test_cov_sandwich_with_logit(
    setup_covariance_matrices, expected_covariance_matrices
):
    inputs, outputs = setup_covariance_matrices, expected_covariance_matrices
    covariance_matrix = cov_sandwich(
        inputs["logit_jacobian_matrix"], inputs["logit_hessian_matrix"]
    )
    aaae(covariance_matrix, outputs["logit_sandwich_cov"])
