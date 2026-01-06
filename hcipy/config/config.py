import os
import yaml
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

class _ConfigurationItem:
    def __init__(self, mapping=None):
        if mapping:
            self.update(mapping, allow_creation=True)

    def __setattr__(self, name, value):
        self.update({name: value})

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def update(self, mapping, allow_creation=False):
        '''Update the configuration with the configuration `mapping`, as a dictionary.

        Parameters
        ----------
        mapping : dict
            A dictionary containing the values to update in the configuration.
        allow_creation : bool
            Whether to allow creation of new fields during the update. Default: False.
        '''
        for key, val in mapping.items():
            # Check whether the key is already set before.
            if key not in self.__dict__:
                if not allow_creation:
                    raise TypeError("Not allowed to set a non-existent attribute.")

                if isinstance(val, dict):
                    val = _ConfigurationItem(val)

                self.__dict__[key] = val

                continue

            # Update the existing item either recursively or as a value.
            field = getattr(self, key)
            if isinstance(field, _ConfigurationItem):
                if not isinstance(val, dict):
                    raise TypeError(f"Not allowed to set the attribute {key} with a non-dict type.")

                field.update(val)
            else:
                if isinstance(val, dict):
                    raise TypeError(f"Not allowed to set the attribute {key} with a dict type.")

                self.__dict__[key] = val

    def __repr__(self):
        return repr(self.__dict__)

class Configuration(_ConfigurationItem):
    '''A configuration object that describes the current configuration status of the package.
    '''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()

        return cls._instance

    def __init__(self):
        super().__init__(mapping=None)

    def reset(self, enable_user_overrides=True):
        '''Reset the configuration to the default configuration.

        This default configuration consists of the default parameters in `hcipy/data/default_config.yaml`, which
        can be overridden by a configuration file in `~/.hcipy/hcipy_config.yaml`. This can in turn be overridden
        by a configuration file named `hcipy_config.yaml` located in the current working directory.

        Parameters
        ----------
        enable_user_overrides : bool
            Whether to enable overrides of the config by the user-specific configuration files.
            The default is True.
        '''
        self.__dict__.clear()

        default_config = files('hcipy.config').joinpath('default_config.yaml')
        user_config = Path(os.path.expanduser('~/.hcipy/hcipy_config.yaml'))
        current_working_directory = Path('./hcipy_config.yaml')

        paths = [default_config]

        if enable_user_overrides:
            paths.extend([user_config, current_working_directory])

        for path in paths:
            try:
                contents = path.read_text()
                new_config = yaml.safe_load(contents)

                self.update(new_config, allow_creation=True)
            except FileNotFoundError:
                # If a configuration file is not found, just ignore.
                pass
