import argparse
import os
import shutil
import warnings
import yaml
from pathlib import Path

__all__ = ['migrate_config_file']

_LEGACY_TO_FLAT = {
    'fourier.fft.emulate_fftshifts': 'fft_emulate_fftshifts',
    'fourier.fft.method': 'fft_method',
    'fourier.fft.execution_time_prediction_coefficients': 'fft_runtime_coeffs',
    'fourier.mft.precompute_matrices': 'mft_precompute_matrices',
    'fourier.mft.allocate_intermediate': 'mft_allocate_intermediate',
    'fourier.mft.execution_time_prediction_coefficients': 'mft_runtime_coeffs',
    'fourier.nft.precompute_matrices': 'nft_precompute_matrices',
    'fourier.zfft.execution_time_prediction_coefficients': 'zfft_runtime_coeffs',
    'plotting.ffmpeg_path': 'ffmpeg_path',
    'plotting.psf_colormap': 'cmap_psf',
    'plotting.pupil_phase_colormap': 'cmap_pupil_phase',
    'core.use_new_style_fields': 'use_array_api',
}

_COEFFICIENT_KEYS = (
    'fourier.fft.execution_time_prediction_coefficients',
    'fourier.mft.execution_time_prediction_coefficients',
    'fourier.zfft.execution_time_prediction_coefficients',
)

_LEGACY_SECTIONS = ('fourier', 'plotting', 'core')

def _flatten(mapping, prefix=''):
    '''Flatten a nested legacy configuration into dotted key paths.'''
    for key, value in mapping.items():
        path = key if not prefix else prefix + '.' + key

        if isinstance(value, dict) and any(p.startswith(path + '.') for p in _LEGACY_TO_FLAT):
            yield from _flatten(value, path)
        else:
            yield path, value

def migrate_config_file(config_data):
    '''Migrate a legacy HCIPy configuration to the current flat format.

    This function maps the legacy keys to their flat counterparts, and
    converts the execution time prediction coefficients from ``{a, b, c}``
    dictionaries to ``[a, b, c]`` lists.

    Parameters
    ----------
    config_data : dict
        The legacy configuration, as a dictionary.

    Returns
    -------
    dict or None
        The migrated flat configuration, or None if the input was already in
        the flat format.
    '''
    if not isinstance(config_data, dict):
        return None

    if not any(key in config_data for key in _LEGACY_SECTIONS):
        return None

    migrated = {}
    ignored = []

    for dotted_key, value in _flatten(config_data):
        if dotted_key in _LEGACY_TO_FLAT:
            if dotted_key in _COEFFICIENT_KEYS and isinstance(value, dict):
                value = [value['a'], value['b'], value['c']]

            migrated[_LEGACY_TO_FLAT[dotted_key]] = value
        else:
            ignored.append(dotted_key)

    if ignored:
        warnings.warn('Ignoring unknown configuration keys: ' + ', '.join(ignored))

    return migrated

def _cli():
    '''A command-line interface for migrating configuration files.
    '''
    parser = argparse.ArgumentParser(description='Migrate HCIPy configuration files to the current format.')
    parser.add_argument('path', nargs='?', type=str, default=None,
                        help='The configuration file to migrate. If not given, the default configuration files '
                             '(./hcipy_config.yaml and ~/.hcipy/hcipy_config.yaml) are migrated.')
    args = parser.parse_args()

    if args.path is not None:
        paths = [Path(args.path)]
    else:
        paths = [Path('./hcipy_config.yaml'), Path(os.path.expanduser('~/.hcipy/hcipy_config.yaml'))]

    migrated_any = False

    for path in paths:
        if not path.exists():
            print(f'{path}: not found, skipping')
            continue

        migrated = migrate_config_file(yaml.safe_load(path.read_text()))

        if migrated is None:
            print(f'{path}: already in the current format')
        else:
            shutil.copy2(path, path.with_suffix(path.suffix + '.bak'))
            path.write_text(yaml.dump(migrated))
            migrated_any = True
            print(f'{path}: migrated {len(migrated)} keys (backup written to {path}.bak)')

    if not migrated_any:
        print('Nothing to migrate.')
