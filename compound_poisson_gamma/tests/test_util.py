from typing import TypedDict

import pytest

from compound_poisson_gamma.util import (
    _TEST_DATA_1,
    _TEST_DATA_2,
    _get_model_params,
    _TestDataDict,
)


class _ModelParamsDict(TypedDict):
    lam: float
    alpha: float
    beta: float


@pytest.mark.parametrize(
    ("test_data", "expected_model_params"),
    [
        (_TEST_DATA_1, _ModelParamsDict(lam=1.43, alpha=2.33, beta=3.33)),
        (_TEST_DATA_2, _ModelParamsDict(lam=3, alpha=99, beta=98.91)),
    ],
)
def test_to_model_params(
    test_data: _TestDataDict, expected_model_params: _ModelParamsDict
) -> None:
    lam, alpha, beta = _get_model_params(**test_data["model_params"])
    assert lam == pytest.approx(expected_model_params["lam"], rel=1e-2)
    assert alpha == pytest.approx(expected_model_params["alpha"], rel=1e-2)
    assert beta == pytest.approx(expected_model_params["beta"], rel=1e-2)
