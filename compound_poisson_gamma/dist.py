import warnings
from functools import partial
from typing import ClassVar, Final

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax, random
from jax.scipy.special import logsumexp
from jax.scipy.stats import gamma, poisson
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key

from .util import (
    _FOUR_SIGMA_P,
    _get_model_params,
    _get_mu,
    _get_sigma,
    _get_theta,
    _poisson_icdf,
)

_MAX_M: Final = 30


def _check_lam(lam: float) -> None:
    max_m = _poisson_icdf(lam, _FOUR_SIGMA_P)
    if max_m > _MAX_M:
        msg = (
            "`lam` (the `rate` parameter on the underlying Poisson distribution) "
            "is large enough to violate the 'small count' assumption used in "
            "`.log_prob`. You may run into sampling issues."
        )
        warnings.warn(msg, stacklevel=1)


class CompoundPoissonGamma(dist.Distribution):
    arg_constraints: ClassVar = {
        "lam": constraints.nonnegative,
        "alpha": constraints.positive,
        "beta": constraints.positive,
    }
    support = constraints.nonnegative
    pytree_data_fields = ("lam", "alpha", "beta", "_poisson", "_build_gamma")

    def __init__(
        self,
        lam: float,
        alpha: float,
        beta: float,
        *,
        validate_args: bool | None = None,
    ) -> None:
        batch_shape = lax.broadcast_shapes(
            jnp.shape(lam),
            jnp.shape(alpha),
            jnp.shape(beta),
        )

        self.lam, self.alpha, self.beta = promote_shapes(lam, alpha, beta)

        self._poisson = dist.Poisson(rate=lam, validate_args=validate_args)
        self._build_gamma = partial(
            dist.Gamma,
            rate=self.beta,
            validate_args=validate_args,
        )

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @classmethod
    def from_tweedie_params(
        cls,
        mu: float,
        sigma: float,
        theta: float,
        *,
        validate_args: bool | None = None,
    ) -> Self:
        lam, alpha, beta = _get_model_params(mu, sigma, theta)
        return cls(lam, alpha, beta, validate_args=validate_args)

    @property
    def mu(self) -> float:
        return _get_mu(lam=self.lam, alpha=self.alpha, beta=self.beta)

    @property
    def sigma(self) -> float:
        return _get_sigma(lam=self.lam, alpha=self.alpha, beta=self.beta)

    @property
    def theta(self) -> float:
        return _get_theta(alpha=self.alpha)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma**2 * self.mu**self.theta

    def sample_with_intermediates(
        self, key: ArrayLike, sample_shape: tuple[int, ...] = ()
    ) -> tuple[ArrayLike, list[ArrayLike]]:
        assert is_prng_key(key)
        key, key_gamma = random.split(key)

        m = self._poisson.sample(key, sample_shape)
        x = self._build_gamma(m * self.alpha).sample(key_gamma)

        return jnp.where(m == 0, 0.0, x), [m]

    def sample(self, key: ArrayLike, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        x, *_ = self.sample_with_intermediates(key, sample_shape=sample_shape)
        return x

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        m = jnp.arange(1, _MAX_M)
        # Must mask zero elements from the Gamma Log-PDF function and replace with
        # safe values. Otherwise we will run into nan gradients as per
        # https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
        nonzero_mask = jnp.asarray(value) > 0
        nonzero_logprob = logsumexp(
            poisson.logpmf(m, self.lam)[:, jnp.newaxis]
            + gamma.logpdf(
                jnp.where(nonzero_mask, value, 1.0),
                a=(m * self.alpha)[:, jnp.newaxis],
                scale=1 / self.beta,
            ),
            axis=0,
        )
        return jnp.where(nonzero_mask, nonzero_logprob, -self.lam)


def _example_compound_poisson_gamma_model(
    mu_prior: float = 5.0,
    phi_prior: float = 5.0,
    y: ArrayLike | None = None,
) -> None:
    mu = numpyro.sample("mu", dist.HalfCauchy(mu_prior))
    phi = numpyro.sample("phi", dist.HalfCauchy(phi_prior))
    sigma = numpyro.deterministic("sigma", jnp.sqrt(phi))
    theta = numpyro.sample("theta", dist.Uniform(1.0, 2.0))

    numpyro.sample(
        "obs",
        CompoundPoissonGamma.from_tweedie_params(mu, sigma, theta),
        obs=y,
    )
