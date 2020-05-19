from hcipy import *

def test_config():
    Configuration()['test_config'] = 1
    assert Configuration()['test_config'] == 1

    Configuration().test_config = 2
    assert Configuration()['test_config'] == 2

    Configuration().update({'test_section': {'test_value': 3}})
    assert Configuration().test_section.test_value == 3
