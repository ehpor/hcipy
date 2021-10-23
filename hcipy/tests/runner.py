import os

def run_tests_cli(args):
    '''Run unit tests using command-line arguments.

    Parameters
    ----------
    args : iterable
        The arguments that will be directly passed to `pytest.main()`.

    Returns
    -------
    pytest.ExitCode
        The result from the unit tests. This is the result of the call
        to `pytest.main()`. A non-zero value indicates an error.
    '''
    import pytest

    file_location = os.path.dirname(__file__)
    return pytest.main([file_location] + list(args))

def run_tests(run_slow=False):
    '''Run unit tests.

    Parameters
    ----------
    run_slow : boolean
        Whether to also run longer tests. These tests do more in-depth
        analysis but are more time-consuming, and are therefore not run
        by default. Default: False.

    Returns
    -------
    pytest.ExitCode
        The result from the unit tests. This is the result of the call
        to `pytest.main()`. A non-zero value indicates an error.
    '''
    args = ()

    if run_slow == True:
        args += ('--runslow',)

    return run_tests_cli(args)
