from hcipy import Configuration
from hcipy.config import on_config_change
from hcipy.config.migrate import _cli
import pytest
from pydantic import ValidationError
import sys
import yaml

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

def test_migrate_config_file():
    from hcipy.config import migrate_config_file

    legacy = {
        'fourier': {
            'fft': {
                'emulate_fftshifts': True,
                'method': ['scipy', 'numpy'],
                'execution_time_prediction_coefficients': {'a': 1.0, 'b': 2.0, 'c': -3.0},
            },
            'mft': {'precompute_matrices': False},
        },
        'plotting': {'psf_colormap': 'magma'},
        'core': {'use_new_style_fields': True},
    }

    migrated = migrate_config_file(legacy)

    assert migrated == {
        'fft_emulate_fftshifts': True,
        'fft_method': ['scipy', 'numpy'],
        'fft_runtime_coeffs': [1.0, 2.0, -3.0],
        'mft_precompute_matrices': False,
        'cmap_psf': 'magma',
        'use_array_api': True,
    }

    # A configuration in the current format is left untouched.
    assert migrate_config_file({'fft_method': ['scipy']}) is None

    # Non-dict input is left untouched.
    assert migrate_config_file('fourier: ...') is None
    assert migrate_config_file(None) is None

def test_migrate_config_cli(tmp_path, monkeypatch):
    config_file = tmp_path / 'hcipy_config.yaml'
    config_file.write_text('fourier:\n  fft:\n    method: [scipy]\ncore:\n  use_new_style_fields: true\n')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['hcipy_migrate_config'])
    _cli()

    assert yaml.safe_load(config_file.read_text()) == {
        'fft_method': ['scipy'],
        'use_array_api': True,
    }
    assert config_file.with_suffix('.yaml.bak').exists()
