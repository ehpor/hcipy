from hcipy import *

def test_config():
    old_value = Configuration().core.use_new_style_fields
    new_value = not old_value

    Configuration().core.use_new_style_fields = new_value

    assert Configuration().core.use_new_style_fields == new_value

    Configuration().reset()

    assert Configuration().core.use_new_style_fields == old_value
