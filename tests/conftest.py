import pytest
import hcipy

def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', help='run slow tests')

def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption('--runslow'):
        pytest.skip('Need --runslow option to run.')

def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: marks tests as slow')

@pytest.fixture(scope='session', autouse=True)
def disable_user_config():
    hcipy.Configuration().reset(enable_user_overrides=False)
    yield
    hcipy.Configuration().reset()

def get_backend(backend_name):
    if backend_name == 'numpy':
        import numpy as np
        return np
    elif backend_name == 'cupy':
        try:
            import cupy as cp
            return cp
        except ImportError:
            pytest.skip(f'{backend_name} not available')
    elif backend_name == 'torch':
        try:
            import torch
            return torch
        except ImportError:
            pytest.skip(f'{backend_name} not available')
    elif backend_name == 'jax':
        try:
            import jax
            return jax
        except ImportError:
            pytest.skip(f'{backend_name} not available')
    else:
        pytest.skip(f'{backend_name} not available')


@pytest.fixture(params=['numpy', 'cupy', 'jax', 'torch'])
def xp(request):
    return get_backend(request.param)
