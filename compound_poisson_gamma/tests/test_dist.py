import jax.numpy as jnp
import numpyro
import pytest
from jax.random import PRNGKey
from numpyro.diagnostics import hpdi
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Trace_ELBO,
)
from numpyro.infer.autoguide import AutoNormal

from compound_poisson_gamma.dist import (
    CompoundPoissonGamma,
    _example_compound_poisson_gamma_model,
    _example_multioutput_compound_poisson_gamma_model,
)
from compound_poisson_gamma.util import (
    _TEST_DATA_1,
    _TEST_DATA_2,
    _TestDataDict,
)


@pytest.mark.parametrize("test_data", [_TEST_DATA_1, _TEST_DATA_2])
def test_mcmc_parameter_estimates(test_data: _TestDataDict) -> None:
    numpyro.set_host_device_count(4)

    data_rng_key = PRNGKey(0)
    dist = CompoundPoissonGamma.from_tweedie_params(**test_data["model_params"])
    y = dist.sample(data_rng_key, (test_data["num_samples"],))

    sampling_rng_key = PRNGKey(1)
    kernel = NUTS(_example_compound_poisson_gamma_model, target_accept_prob=0.90)
    mcmc = MCMC(
        kernel, num_warmup=250, num_samples=500, num_chains=4, chain_method="parallel"
    )
    mcmc.warmup(sampling_rng_key, y=y)
    mcmc.run(sampling_rng_key, y=y)

    assert not mcmc.get_extra_fields()["diverging"].any()

    posterior_samples = mcmc.get_samples()

    mu = test_data["model_params"]["mu"]
    sigma = test_data["model_params"]["sigma"]
    theta = test_data["model_params"]["theta"]

    mu_lower, mu_upper = hpdi(posterior_samples["mu"])
    sigma_lower, sigma_upper = hpdi(posterior_samples["sigma"])
    theta_lower, theta_upper = hpdi(posterior_samples["theta"])

    assert mu_lower < mu < mu_upper
    assert sigma_lower < sigma < sigma_upper
    assert theta_lower < theta < theta_upper


def test_multioutput_mcmc_parameter_estimates() -> None:
    numpyro.set_host_device_count(4)

    data_rng_key = PRNGKey(0)
    dist_1 = CompoundPoissonGamma.from_tweedie_params(**_TEST_DATA_1["model_params"])
    y_1 = dist_1.sample(data_rng_key, (_TEST_DATA_1["num_samples"],))
    dist_2 = CompoundPoissonGamma.from_tweedie_params(**_TEST_DATA_2["model_params"])
    y_2 = dist_2.sample(data_rng_key, (_TEST_DATA_1["num_samples"],))

    y = jnp.vstack([y_1, y_2]).transpose()

    assert y.shape == (_TEST_DATA_1["num_samples"], 2)

    sampling_rng_key = PRNGKey(1)
    kernel = NUTS(
        _example_multioutput_compound_poisson_gamma_model, target_accept_prob=0.90
    )
    mcmc = MCMC(
        kernel, num_warmup=250, num_samples=500, num_chains=4, chain_method="parallel"
    )
    mcmc.warmup(sampling_rng_key, n=2, y=y)
    mcmc.run(sampling_rng_key, n=2, y=y)

    assert not mcmc.get_extra_fields()["diverging"].any()

    posterior_samples = mcmc.get_samples()

    mu_1 = _TEST_DATA_1["model_params"]["mu"]
    sigma_1 = _TEST_DATA_1["model_params"]["sigma"]
    theta_1 = _TEST_DATA_1["model_params"]["theta"]

    mu_2 = _TEST_DATA_2["model_params"]["mu"]
    sigma_2 = _TEST_DATA_2["model_params"]["sigma"]
    theta_2 = _TEST_DATA_2["model_params"]["theta"]

    mu_1_lower, mu_1_upper = hpdi(posterior_samples["mu"][:, 0])
    sigma_1_lower, sigma_1_upper = hpdi(posterior_samples["sigma"][:, 0])
    theta_1_lower, theta_1_upper = hpdi(posterior_samples["theta"][:, 0])

    assert mu_1_lower < mu_1 < mu_1_upper
    assert sigma_1_lower < sigma_1 < sigma_1_upper
    assert theta_1_lower < theta_1 < theta_1_upper

    mu_2_lower, mu_2_upper = hpdi(posterior_samples["mu"][:, 1])
    sigma_2_lower, sigma_2_upper = hpdi(posterior_samples["sigma"][:, 1])
    theta_2_lower, theta_2_upper = hpdi(posterior_samples["theta"][:, 1])

    assert mu_2_lower < mu_2 < mu_2_upper
    assert sigma_2_lower < sigma_2 < sigma_2_upper
    assert theta_2_lower < theta_2 < theta_2_upper


@pytest.mark.skip
@pytest.mark.parametrize("test_data", [_TEST_DATA_1, _TEST_DATA_2])
def test_svi_parameter_estimates(test_data: _TestDataDict) -> None:
    numpyro.set_host_device_count(4)

    data_rng_key = PRNGKey(0)
    dist = CompoundPoissonGamma.from_tweedie_params(**test_data["model_params"])
    y = dist.sample(data_rng_key, (test_data["num_samples"],))

    sampling_rng_key = PRNGKey(1)
    guide = AutoNormal(_example_compound_poisson_gamma_model)
    loss = Trace_ELBO()
    optimiser = numpyro.optim.Adam(step_size=1e-4)

    svi = SVI(
        _example_compound_poisson_gamma_model, guide=guide, optim=optimiser, loss=loss
    )
    svi.run(
        sampling_rng_key,
        y=y,
        num_steps=50_000,
    )
