from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Callable, ClassVar

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import InitSettingsSource

__all__ = [
    'Configuration',
    'on_config_change',
]

_callbacks: list[Callable[['Configuration'], None]] = []

def on_config_change(func):
    '''Decorator that registers `func` to be called whenever (any part of) the
    configuration is assigned a value or reloaded by `Configuration().reset()`.
    '''
    _callbacks.append(func)
    return func

class YAMLConfigSource(InitSettingsSource):
    '''A settings source that reads a (possibly missing) YAML file.'''
    def __init__(self, settings_cls, path):
        try:
            file_data = yaml.safe_load(Path(path).read_text())
        except FileNotFoundError:
            file_data = None

        if not isinstance(file_data, dict):
            file_data = {}

        super().__init__(settings_cls, file_data)

class Configuration(BaseSettings):
    '''The global configuration object, implemented as a singleton.

    Use `Configuration()` to access the single configuration instance and
    `Configuration().reset()` to (re-)load the configuration from the
    configuration files and environment variables.
    '''
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # FFT
    fft_emulate_fftshifts: bool = True
    fft_method: list[str] = ['mkl', 'scipy', 'fftw', 'numpy']
    fft_runtime_coeffs: tuple[float, float, float] = (3.746, 0.881, -4.749)

    # MFT
    mft_precompute_matrices: bool = True
    mft_allocate_intermediate: bool = True
    mft_runtime_coeffs: tuple[float, float, float] = (1.667, 0.716, -4.624)

    # NFT
    nft_precompute_matrices: bool = False

    # ZFFT
    zfft_runtime_coeffs: tuple[float, float, float] = (4.093, 0.847, -2.973)

    # Plotting
    ffmpeg_path: str | None = None
    cmap_psf: str = 'inferno'
    cmap_pupil_phase: str = 'RdBu'

    # Core
    use_array_api: bool = False

    # Internal class management
    _instance: ClassVar[Configuration | None] = None
    _initialized: ClassVar[bool] = False
    _enable_user_overrides: ClassVar[bool] = True

    model_config = SettingsConfigDict(
        env_prefix='HCIPY_',
        extra='forbid',
        validate_assignment=True,
    )

    @model_validator(mode='after')
    def _notify(self):
        for callback in _callbacks:
            callback(self)

        return self

    def __init__(self):
        if not Configuration._initialized:
            Configuration._initialized = True
            super().__init__()

    def reset(self, enable_user_overrides=True):
        '''Reset the configuration to the default configuration.

        This re-reads the configuration files and environment variables. If
        `enable_user_overrides` is False, the environment variables and the
        user-specific configuration files (`./hcipy_config.yaml` and
        `~/.hcipy/hcipy_config.yaml`) are skipped, and only the defaults are
        used.

        Parameters
        ----------
        enable_user_overrides : bool
            Whether to enable overrides of the config by the user-specific
            configuration files and environment variables. The default is True.
        '''
        Configuration._enable_user_overrides = enable_user_overrides
        BaseSettings.__init__(self)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        sources = [init_settings]

        if cls._enable_user_overrides:
            sources.append(env_settings)
            sources.append(YAMLConfigSource(settings_cls, Path('./hcipy_config.yaml')))
            sources.append(YAMLConfigSource(settings_cls, Path(os.path.expanduser('~/.hcipy/hcipy_config.yaml'))))

        return tuple(sources)
