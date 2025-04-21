def test_package_level_imports() -> None:
    import compound_poisson_gamma

    assert set(compound_poisson_gamma.__all__) == {"CompoundPoissonGamma"}
