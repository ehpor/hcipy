import functools
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml
import difflib

from tqdm import tqdm

from .formulas import (
    make_cauchy_index,
    make_exotic_index,
    make_gases_index,
    make_herzberger_index,
    make_polynomial_index,
    make_retro_index,
    make_rii_index,
    make_sellmeier2_index,
    make_sellmeier_index
)
from .material import RefractiveIndexMaterial

_YAML_LOADER = getattr(yaml, 'CSafeLoader', None) or yaml.SafeLoader

__all__ = [
    'AmbiguousMaterialError',
    'MaterialCatalog',
    'download_rii_database',
    'get_material',
    'search_material'
]

DEFAULT_DATABASE_PATH = Path.home() / '.hcipy' / 'rii_database'
EXCLUDED_SHELVES = ('3d', 'popular_glass')
DATABASE_URL = 'https://github.com/polyanskiy/refractiveindex.info-database/releases/download/v2026-05-24/rii-database-2026-05-24.zip'


class AmbiguousMaterialError(ValueError):
    '''Raised when a material name resolves to multiple pages.

    The candidates are available through the ``candidates`` attribute, as a
    list of the page_info dictionaries of all matching pages.
    '''
    def __init__(self, message, candidates=None):
        super().__init__(message)
        self.candidates = candidates


_FORMULAS = {
    1: make_sellmeier_index,
    2: make_sellmeier2_index,
    3: make_polynomial_index,
    4: make_rii_index,
    5: make_cauchy_index,
    6: make_gases_index,
    7: make_herzberger_index,
    8: make_retro_index,
    9: make_exotic_index
}

def _make_formula_refractive_index(formula_id, coefficients):
    '''Create a refractive index function from a formula id (1-9) and coefficients.'''
    if formula_id not in _FORMULAS:
        raise ValueError(f'Unknown dispersion formula {formula_id!r}.')
    return _FORMULAS[formula_id](coefficients)


def _parse_tabulated(data_str):
    rows = data_str.strip().splitlines()
    wavelength = []
    col1 = []
    col2 = []
    for row in rows:
        parts = row.split()
        if not parts or parts[0].startswith('#'):
            continue
        wavelength.append(float(parts[0]))
        col1.append(float(parts[1]))
        if len(parts) > 2:
            col2.append(float(parts[2]))
    wavelength = np.array(wavelength)
    col1 = np.array(col1)
    col2 = np.array(col2) if col2 else None
    return wavelength, col1, col2


class _PageSpec:
    __slots__ = ('shelf', 'book', 'page', 'filepath')

    def __init__(self, shelf, book, page, filepath):
        self.shelf = shelf
        self.book = book
        self.page = page
        self.filepath = filepath

    def load(self):
        with open(self.filepath, encoding='utf-8') as f:
            doc = yaml.load(f, Loader=_YAML_LOADER)

        n_grid = None
        k_grid = None
        formula = None
        wavelength_range = None

        for data in doc.get('DATA', []):
            parts = data['type'].split()
            category = parts[0]
            subtype = parts[1] if len(parts) > 1 else None

            if category == 'formula':
                fid = int(subtype)
                coefficients = [float(c) for c in data['coefficients'].split()]
                rng = data.get('range', data.get('wavelength_range'))
                if rng is not None:
                    lo, hi = (float(x) for x in rng.split())
                    wavelength_range = (lo * 1e-6, hi * 1e-6)
                formula = (fid, coefficients)
            elif category == 'tabulated':
                wl, c1, c2 = _parse_tabulated(data['data'])
                wl = wl * 1e-6
                if subtype == 'n':
                    n_grid = (wl, c1)
                elif subtype == 'k':
                    k_grid = (wl, c1)
                elif subtype == 'nk':
                    n_grid = (wl, c1)
                    k_grid = (wl, c2)

        if formula is not None:
            fid, coefficients = formula
            try:
                n_function = _make_formula_refractive_index(fid, coefficients)
            except ValueError as e:
                raise ValueError(f'{self.shelf}/{self.book}/{self.page}: {e}') from e
        elif n_grid is not None:
            n_wavelengths, n_values = n_grid
            def n_function(wavelength, _x=n_wavelengths, _y=n_values):
                return np.interp(wavelength, _x, _y)
        else:
            n_function = None

        if k_grid is not None:
            k_wavelengths, k_values = k_grid
            def k_function(wavelength, _x=k_wavelengths, _y=k_values):
                return np.interp(wavelength, _x, _y)
        else:
            k_function = None

        return RefractiveIndexMaterial(
            self.page,
            n_function,
            k_function,
            shelf=self.shelf,
            book=self.book,
            wavelength_range=wavelength_range
        )


class MaterialCatalog:
    '''A catalog of materials from the refractiveindex.info database.

    Parameters
    ----------
    path : str or Path, optional
        The path of the refractiveindex.info database. Defaults to
        ~/.hcipy/rii_database. When the path is omitted, the database is
        downloaded automatically on first use.

    Notes
    -----
    The '3d' and 'popular_glass' shelves are excluded, as they contain no
    data that is not already present in the other shelves.
    '''
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else DEFAULT_DATABASE_PATH

        catalog_file = self.path / 'catalog-nk.yml'
        if not catalog_file.exists():
            if path is None:
                download_rii_database(self.path)
            else:
                raise FileNotFoundError(
                    f'The refractiveindex.info database was not found at {self.path}. '
                    'Download it with hcipy.materials.download_rii_database().')

        self._index = {}
        self._pages = []

        with open(catalog_file, encoding='utf-8') as f:
            catalog = yaml.load(f, Loader=_YAML_LOADER)

        for shelf_entry in catalog:
            if 'SHELF' not in shelf_entry:
                continue

            shelf = shelf_entry['SHELF']
            if shelf in EXCLUDED_SHELVES:
                continue

            for book_entry in shelf_entry.get('content', []):
                if 'BOOK' not in book_entry:
                    continue

                book = book_entry['BOOK']
                for page_entry in book_entry.get('content', []):
                    if 'PAGE' not in page_entry:
                        continue

                    page = page_entry['PAGE']
                    data_path = page_entry.get('data')
                    if data_path is None:
                        continue

                    filepath = self.path / 'data' / Path(data_path)
                    if not filepath.exists():
                        continue

                    spec = _PageSpec(shelf, book, page, filepath)
                    self._pages.append(spec)
                    self._index.setdefault(page, []).append(spec)

    @staticmethod
    def _matches(spec, shelf, book, page):
        if shelf is not None and spec.shelf != shelf:
            return False

        if book is not None and spec.book != book:
            return False

        if page is not None and spec.page != page:
            return False

        return True

    def get(self, page, /, *, shelf=None, book=None):
        '''Get the unique material matching the given page name.

        The page must match a page name exactly. The search can be narrowed
        with the shelf and book keywords. Raises a KeyError when no material
        matches, and an AmbiguousMaterialError when multiple pages match.

        Parameters
        ----------
        page : str
            The name of the page.
        shelf : str, optional
            Narrow the search to the given shelf.
        book : str, optional
            Narrow the search to the given book.

        Returns
        -------
        RefractiveIndexMaterial
            The material.
        '''
        candidates = self._index.get(page, [])
        candidates = [c for c in candidates if self._matches(c, shelf, book, None)]

        if not candidates:
            suggestions = difflib.get_close_matches(page, self._index.keys(), n=3)

            if len(suggestions) > 0:
                suggestion_names = [repr(s) for s in suggestions]
                raise KeyError(f'No material named {page!r} found in the catalog. Did you mean {" or ".join(suggestion_names)}?')

            raise KeyError(f'No material named {page!r} found in the catalog.')

        if len(candidates) > 1:
            raise AmbiguousMaterialError(
                f'The material named {page!r} matches {len(candidates)} pages. Narrow the search '
                'with the shelf or book keywords.',
                candidates=[c.load().page_info for c in candidates])

        return candidates[0].load()

    def search(self, page=None, /, *, shelf=None, book=None):
        '''Get all materials matching the given criteria.

        The page is an exact-case substring match over the page and book
        names. The search can be narrowed with the shelf and book keywords.

        Parameters
        ----------
        page : str, optional
            A substring of the page or book name.
        shelf : str, optional
            Narrow the search to the given shelf.
        book : str, optional
            Narrow the search to the given book.

        Returns
        -------
        list of RefractiveIndexMaterial
            The materials matching the criteria.
        '''
        results = []

        results = []

        for spec in self._pages:
            if not self._matches(spec, shelf, book, None):
                continue

            if page is not None:
                if page not in spec.page and page not in spec.book:
                    continue
            try:
                results.append(spec.load())
            except (ValueError, KeyError, IndexError, TypeError):
                continue

        return results

    def __getitem__(self, page):
        '''Get the unique material matching the given page.'''
        return self.get(page)

    def __contains__(self, page):
        '''Check whether the catalog contains a material with the given page.'''
        return page in self._index

    def __len__(self):
        '''Get the number of materials in the catalog.'''
        return len(self._pages)

    def __repr__(self):
        return f'MaterialCatalog({str(self.path)!r}, {len(self)} pages)'


@functools.cache
def _default_catalog():
    return MaterialCatalog()


def get_material(page, /, *, shelf=None, book=None):
    '''Get the unique material matching the given page from the default catalog.

    Parameters
    ----------
    page : str
        The name of the page.
    shelf : str, optional
        Narrow the search to the given shelf.
    book : str, optional
        Narrow the search to the given book.

    Returns
    -------
    RefractiveIndexMaterial
        The material.
    '''
    return _default_catalog().get(page, shelf=shelf, book=book)


def search_material(page=None, /, *, shelf=None, book=None):
    '''Get all materials matching the given criteria from the default catalog.

    Parameters
    ----------
    page : str, optional
        A substring of the page or book name.
    shelf : str, optional
        Narrow the search to the given shelf.
    book : str, optional
        Narrow the search to the given book.

    Returns
    -------
    list of RefractiveIndexMaterial
        The materials matching the criteria.
    '''
    return _default_catalog().search(page, shelf=shelf, book=book)


def download_rii_database(path=None):
    '''Download (or re-download) the refractiveindex.info database.

    Parameters
    ----------
    path : str or Path, optional
        The directory in which to store the database. Defaults to
        ~/.hcipy/rii_database.

    Returns
    -------
    Path
        The path of the downloaded database.
    '''
    if path is None:
        path = DEFAULT_DATABASE_PATH

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive_path = tmp / 'rii.zip'

        request = urllib.request.Request(DATABASE_URL)

        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get('Content-Length', 0))

            with tqdm(total=total, unit='B', unit_scale=True, desc='Downloading refractiveindex.info database') as pbar:
                with open(archive_path, 'wb') as f:
                    while True:
                        chunk = response.read(1024 * 256)

                        if not chunk:
                            break

                        f.write(chunk)
                        pbar.update(len(chunk))

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp / 'staging')

        database_dir = tmp / 'staging' / 'database'
        if not database_dir.exists():
            raise ValueError(f'No database directory found in {DATABASE_URL}.')

        for entry in database_dir.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, path / entry.name, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, path / entry.name)

    return path
