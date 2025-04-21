from typing import Final, TypedDict, overload

import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.typing import ArrayLike

_FOUR_SIGMA_P: Final = 0.999936657516334


class _TestModelParamsDict(TypedDict):
    mu: float
    sigma: float
    theta: float


class _TestDataDict(TypedDict):
    model_params: _TestModelParamsDict
    num_samples: int


_TEST_DATA_1: Final[_TestDataDict] = {
    "model_params": {"mu": 1.0, "sigma": 1.0, "theta": 1.3},
    "num_samples": 200,
}
_TEST_DATA_2: Final[_TestDataDict] = {
    "model_params": {"mu": 3.0, "sigma": 1.0, "theta": 1.01},
    "num_samples": 1000,
}


@overload
def _poisson_icdf(lam: float, value: float) -> float: ...


@overload
def _poisson_icdf(lam: float, value: ArrayLike) -> ArrayLike: ...


def _poisson_icdf(lam: float, value: float | ArrayLike) -> float | ArrayLike:
    """Normal Asymptotic Approximation of the Poisson distribution inverse CDF.

    From:
    Michael B. Giles "Algorithm 955: approximation of the inverse
    Poisson cumulative distribution function"
    http://dx.doi.org/10.1145/2699466
    """
    w = ndtri(value)
    q_n1 = lam + jnp.sqrt(lam) * w + (1 / 3) + ((1 / 6) * w**2)
    q_n2 = q_n1 + (1 / jnp.sqrt(lam)) * ((-w / 36) - (1 / 72) * w**3)
    return q_n2 + (1 / lam) * ((-8 / 405) + (7 / 810) * w**2 + (1 / 270) * w**4)


@overload
def _get_lambda(
    mu: float,
    sigma: float,
    theta: float,
) -> float: ...


@overload
def _get_lambda(
    mu: ArrayLike,
    sigma: ArrayLike,
    theta: ArrayLike,
) -> ArrayLike: ...


def _get_lambda(
    mu: float | ArrayLike,
    sigma: float | ArrayLike,
    theta: float | ArrayLike,
) -> float | ArrayLike:
    return (1 / sigma**2) * (mu ** (2 - theta)) / (2 - theta)


def _get_alpha(theta: ArrayLike) -> ArrayLike:
    return (2 - theta) / (theta - 1)


@overload
def _get_beta(
    mu: float,
    sigma: float,
    theta: float,
) -> float: ...


@overload
def _get_beta(
    mu: ArrayLike,
    sigma: ArrayLike,
    theta: ArrayLike,
) -> ArrayLike: ...


def _get_beta(
    mu: float | ArrayLike,
    sigma: float | ArrayLike,
    theta: float | ArrayLike,
) -> float | ArrayLike:
    return (1 / sigma**2) * (mu ** (1 - theta)) / (theta - 1)


@overload
def _get_mu(
    lam: float,
    alpha: float,
    beta: float,
) -> float: ...


@overload
def _get_mu(
    lam: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> ArrayLike: ...


def _get_mu(
    lam: float | ArrayLike,
    alpha: float | ArrayLike,
    beta: float | ArrayLike,
) -> float | ArrayLike:
    return lam * alpha / beta


@overload
def _get_sigma(
    lam: float,
    alpha: float,
    beta: float,
) -> float: ...


@overload
def _get_sigma(
    lam: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> ArrayLike: ...


def _get_sigma(
    lam: float | ArrayLike,
    alpha: float | ArrayLike,
    beta: float | ArrayLike,
) -> float | ArrayLike:
    theta = _get_theta(alpha)
    return ((lam ** (1 - theta)) * (alpha / beta) ** (2 - theta)) / (2 - theta)


@overload
def _get_theta(alpha: float) -> float: ...


@overload
def _get_theta(alpha: ArrayLike) -> ArrayLike: ...


def _get_theta(alpha: float | ArrayLike) -> float | ArrayLike:
    return (alpha + 2) / (alpha + 1)


def _get_tweedie_params(
    lam: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    mu = _get_mu(lam, alpha, beta)
    sigma = _get_sigma(lam, alpha, beta)
    lam = _get_theta(alpha)
    return mu, sigma, lam


@overload
def _get_model_params(
    mu: float,
    sigma: float,
    theta: float,
) -> tuple[float, float, float]: ...


@overload
def _get_model_params(
    mu: ArrayLike,
    sigma: ArrayLike,
    theta: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...


def _get_model_params(
    mu: float | ArrayLike,
    sigma: float | ArrayLike,
    theta: float | ArrayLike,
) -> tuple[float | ArrayLike, float | ArrayLike, float | ArrayLike]:
    lam = _get_lambda(mu, sigma, theta)
    alpha = _get_alpha(theta)
    beta = _get_beta(mu, sigma, theta)
    return lam, alpha, beta
