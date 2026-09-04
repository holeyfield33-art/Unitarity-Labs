"""Package import sanity check."""


def test_package_imports():
    import unitarity_labs

    assert unitarity_labs.__version__


def test_subpackages_import():
    import unitarity_labs.ecc  # noqa: F401
    import unitarity_labs.instrumentation  # noqa: F401
    import unitarity_labs.provenance  # noqa: F401
    import unitarity_labs.statistics  # noqa: F401
