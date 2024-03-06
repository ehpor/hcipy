import os
import yaml
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

class _ConfigurationItem(object):
    def __init__(self, val):
        self._val = val

    def __getitem__(self, key):
        val = self._val[key]

        if isinstance(val, dict):
            return _ConfigurationItem(val)
        else:
            return val

    def __setitem__(self, key, value):
        self._val[key] = value

    def __contains__(self, key):
        return key in self._val

    def __getattr__(self, name):
        if name == '_val':
            return getattr(self, name)
        else:
            return self.__getitem__(name)

    def __setattr__(self, name, value):
        if name == '_val':
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __str__(self):
        return str(self._val)

    def clear(self):
        self._val.clear()

    def update(self, b):
        for key in b.keys():
            if key in self:
                if isinstance(b[key], dict) and isinstance(self._val[key], dict):
                    self[key].update(b[key])
                else:
                    self[key] = b[key]
            else:
                self[key] = b[key]

class Configuration(object):
    '''A configuration object that describes the current configuration status of the package.
    '''
    def __init__(self):
        if Configuration._config is None:
            self.reset()

    _config = None

    def __getitem__(self, key):
        '''Get the value for the key.

        Parameters
        ----------
        key : string
            The configuration key.
        '''
        return Configuration._config[key]

    def __setitem__(self, key, value):
        '''Set the value for the key.

        Parameters
        ----------
        key : string
            The configuration key.
        value : anything
            The value to set this to.
        '''
        Configuration._config[key] = value

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def __str__(self):
        return str(Configuration._config)

    def reset(self):
        '''Reset the configuration to the default configuration.

        This default configuration consists of the default parameters in `hcipy/data/default_config.yaml`, which
        can be overridden by a configuration file in `~/.hcipy/hcipy_config.yaml`. This can in turn be overridden
        by a configuration file named `hcipy_config.yaml` located in the current working directory.
        '''
        Configuration._config = _ConfigurationItem({})

        default_config = files('hcipy.config').joinpath('default_config.yaml')
        user_config = Path(os.path.expanduser('~/.hcipy/hcipy_config.yaml'))
        current_working_directory = Path('./hcipy_config.yaml')

        paths = [default_config, user_config, current_working_directory]

        for path in paths:
            try:
                contents = path.read_text()
                new_config = yaml.safe_load(contents)

                self.update(new_config)
            except FileNotFoundError:
                # If a configuration file is not found, just ignore.
                pass

    def update(self, b):
        '''Update the configuration with the configuration `b`, as a dictionary.

        Parameters
        ----------
        b : dict
            A dictionary containing the values to update in the configuration.
        '''
        Configuration._config.update(b)
