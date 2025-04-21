# Compound Poisson Gamma Regression

[![Run Unit Tests](https://github.com/billyvinning/compound-poisson-gamma.numpyro/actions/workflows/test.yaml/badge.svg)](https://github.com/billyvinning/compound-poisson-gamma.numpyro/actions/workflows/test.yaml)

Parameter estimation for the Compound Poisson Gamma distribution in [NumPyro](https://num.pyro.ai).

## What is the Compound Poisson Gamma Distribution?

The Compound Poisson Gamma Distribution is a Tweedie distribution with Tweedie power parameter $1 < p < 2$. Put explicitly:

$$ N \sim \text{Poisson}(\lambda) $$
$$ X_i \sim \Gamma(\alpha, \beta) $$
$$ Y = \sum^N_i X_i $$

The resulting distribution has seen applications in the actuarial sciences for modelling insurance claims and in meteorology for modelling monthly rainfall.

## How do I install this package?

The package can be installed like so:

```console
git clone https://github.com/billyvinning/compound-poisson-gamma.numpyro
cd compound-poisson-gamma.numpyro
pip install .
```

## How do I use this package?

This package implements a `numpyro.distributions.Distribution` class for the Compound Poisson Gamma distribution, allowing for simple Bayesian inference via MCMC methods such as HMC/NUTS.

Supposing that you wanted to perform some simple parameter estimation given some observed data, define a NumPyro model in the usual fashion, using `CompoundPoissonGamma` as the outcome distribution.

```python
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.typing import ArrayLike
from compound_poisson_gamma import CompoundPoissonGamma

def model(
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
```

Then perform inference in the usual way:

```python
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

y = ... # Your observed data.
rng_key = PRNGKey(1)

kernel = NUTS(model, target_accept_prob=0.90)
mcmc = MCMC(
    kernel,
    num_warmup=250,
    num_samples=500,
    num_chains=4,
    chain_method="parallel"
)
mcmc.warmup(rng_key, y=y)
mcmc.run(rng_key, y=y)

posterior_parameter_samples = mcmc.get_samples()
```

## References

- [Tweedie分布のパラメータを推定する (Stan Implementation)](https://statmodeling.hatenablog.com/entry/tweedie-distribution)
- [Compound Poisson Gamma distribution](https://en.wikipedia.org/wiki/Compound_Poisson_distribution#Discrete_compound_Poisson_distribution)
