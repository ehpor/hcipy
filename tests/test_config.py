from hcipy import Configuration
from hcipy.config import on_config_change
import pytest
from pydantic import ValidationError

@pytest.fixture(autouse=True)
def reset_config():
    Configuration().reset(enable_user_overrides=False)
    yield
    Configuration().reset()

def test_config():
    old_value = Configuration().use_array_api
    new_value = not old_value

    Configuration().use_array_api = new_value

    assert Configuration().use_array_api == new_value

    Configuration().reset()

    assert Configuration().use_array_api == old_value

    Configuration().fft_method = ['scipy', 'numpy']
    assert Configuration().fft_method == ['scipy', 'numpy']

    Configuration().cmap_psf = 'magma'
    assert Configuration().cmap_psf == 'magma'

    Configuration().reset()
    assert Configuration().fft_method == ['mkl', 'scipy', 'fftw', 'numpy']
    assert Configuration().cmap_psf == 'inferno'

def test_on_config_change():
    calls = []

    @on_config_change
    def record(configuration):
        calls.append(configuration)

    # Assignment fires the callbacks with the root Configuration instance.
    Configuration().use_array_api = True
    assert calls
    assert all(call is Configuration() for call in calls)
    assert calls[-1].use_array_api is True

    Configuration().cmap_psf = 'viridis'
    assert calls[-1].cmap_psf == 'viridis'

    # No-op assignments also fire.
    n = len(calls)
    Configuration().cmap_psf = 'viridis'
    assert len(calls) == n + 1

    # reset() fires once, with the reloaded Configuration instance.
    Configuration().reset()
    assert len(calls) == n + 2
    assert calls[-1].use_array_api is False

def test_assignment_errors():
    with pytest.raises(ValidationError):
        Configuration().non_existent_variable_name = 0

    with pytest.raises(ValidationError):
        Configuration().use_array_api = {'a': 0}

    with pytest.raises(ValidationError):
        Configuration().fft_method = 'scipy'
