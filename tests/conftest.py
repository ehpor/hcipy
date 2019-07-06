import pytest

def pytest_addoption(parser):
	parser.addoption('--runslow', action='store_true', help='run slow tests')

def pytest_runtest_setup(item):
	if 'slow' in item.keywords and not item.config.getoption('--runslow'):
		pytest.skip('Need --runslow option to run.')

def pytest_configure(config):
	config.addinivalue_line('markers', 'slow: marks tests as slow')
