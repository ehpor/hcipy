try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Try backported to PY<37 importlib_metadata.
    from importlib_metadata import version, PackageNotFoundError

def get_version():
    if get_version._version is None:
        try:
            get_version._version = version('hcipy')
        except PackageNotFoundError:
            # package is not installed
            pass

    return get_version._version

get_version._version = None
