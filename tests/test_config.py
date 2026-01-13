from hcipy import Configuration
import pytest

def test_config():
    old_value = Configuration().core.use_new_style_fields
    new_value = not old_value

    Configuration().core.use_new_style_fields = new_value

    assert Configuration().core.use_new_style_fields == new_value
    assert Configuration()["core"]["use_new_style_fields"] == new_value

    Configuration().reset()

    assert Configuration().core.use_new_style_fields == old_value
    assert Configuration()["core"]["use_new_style_fields"] == old_value

    with pytest.raises(TypeError):
        Configuration().non_existent_variable_name = 0

    with pytest.raises(TypeError):
        Configuration().core = False

    with pytest.raises(TypeError):
        Configuration().core.use_new_style_fields = {'a': 0}
