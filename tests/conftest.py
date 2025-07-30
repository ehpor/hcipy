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
