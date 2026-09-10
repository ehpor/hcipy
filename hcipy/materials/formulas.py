import numpy as np

__all__ = [
    'make_sellmeier_index',
    'make_sellmeier2_index',
    'make_polynomial_index',
    'make_rii_index',
    'make_cauchy_index',
    'make_gases_index',
    'make_herzberger_index',
    'make_retro_index',
    'make_exotic_index'
]


def _check_pairs(coefficients, formula):
    n = len(coefficients)

    if n < 3 or n % 2 == 0:
        raise ValueError(
            f'The {formula} formula requires the constant C0 followed by coefficient pairs, '
            f'got {n} coefficients.')


def _check_count(coefficients, n, formula):
    if len(coefficients) != n:
        raise ValueError(f'The {formula} formula requires exactly {n} coefficients, got {len(coefficients)}.')


def make_sellmeier_index(coefficients):
    r'''The Sellmeier dispersion formula for the refractive index of materials.

    The dispersion relation of the Sellmeier formula is [Sellmeier1871]_,

    .. math:: n^2 = 1 + C_0 + \sum_{i}{\frac{K_i \lambda^2}{\lambda^2 - L_i^2}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the coefficients do not consist of C0 followed by coefficient pairs.

    .. [Sellmeier1871] W. Sellmeier 1871, Zur Erklärung der abnormen Farbenfolge im
        Spectrum einiger Substanzen, Annalen der Physik und Chemie. 143, 272–282 (1871)
    '''
    _check_pairs(coefficients, 'Sellmeier')

    def refractive_index(wavelength):
        wl = wavelength * 1e6
        nsq = 1.0 + coefficients[0]

        for i in range(1, len(coefficients) - 1, 2):
            nsq = nsq + coefficients[i] * wl**2 / (wl**2 - coefficients[i + 1]**2)

        return np.sqrt(nsq)

    return refractive_index


def make_sellmeier2_index(coefficients):
    r'''The Sellmeier-2 dispersion formula for the refractive index of materials.

    In this form the resonance terms are given in um**2 directly. The dispersion
    relation of the Sellmeier-2 formula is [SchottTIE29]_,

    .. math:: n^2 = 1 + C_0 + \sum_{i}{\frac{K_i \lambda^2}{\lambda^2 - L_i}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the coefficients do not consist of C0 followed by coefficient pairs.

    .. [SchottTIE29] Schott AG 2023, Refractive Index and Dispersion, Technical
        Information Advanced Optics TIE-29, Mainz, Germany (2023)
    '''
    _check_pairs(coefficients, 'Sellmeier-2')

    def refractive_index(wavelength):
        wl = wavelength * 1e6
        nsq = 1.0 + coefficients[0]

        for i in range(1, len(coefficients) - 1, 2):
            nsq = nsq + coefficients[i] * wl**2 / (wl**2 - coefficients[i + 1])

        return np.sqrt(nsq)

    return refractive_index


def make_polynomial_index(coefficients):
    r'''The polynomial dispersion formula for the refractive index of materials.

    The dispersion relation of the polynomial formula is [Polyanskiy2024]_,

    .. math:: n^2 = \sum_{i}{C_i \lambda^{k_i}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula. Coefficients come in
        (coefficient, exponent) pairs after the leading C0 term.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the coefficients do not consist of C0 followed by coefficient pairs.

    .. [Polyanskiy2024] M. N. Polyanskiy 2024, Refractiveindex.info database of optical
        constants, Scientific Data. 11, 94 (2024)
    '''
    _check_pairs(coefficients, 'polynomial')

    def refractive_index(wavelength):
        wl = wavelength * 1e6
        nsq = coefficients[0]

        for i in range(1, len(coefficients) - 1, 2):
            nsq = nsq + coefficients[i] * wl**coefficients[i + 1]

        return np.sqrt(nsq)

    return refractive_index


def make_rii_index(coefficients):
    r'''The RefractiveIndex.INFO dispersion formula for the refractive index of materials.

    The dispersion relation of the RefractiveIndex.INFO formula is [Polyanskiy2024]_,

    .. math:: n^2 = C_0 + \sum_{i}{\frac{C_i \lambda^{C_{i+1}}}{\lambda^2 - C_{i+2}^{C_{i+3}}}} + \sum_{j}{C_j \lambda^{C_{j+1}}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the coefficients do not consist of C0, four coefficients per
        resonance term, followed by coefficient pairs.

    .. [Polyanskiy2024] M. N. Polyanskiy 2024, Refractiveindex.info database of optical
        constants, Scientific Data. 11, 94 (2024)
    '''
    n = len(coefficients)

    if n < 9 or (n - 9) % 2 != 0:
        raise ValueError(
            f'The RefractiveIndex.INFO formula requires at least 9 coefficients, followed by '
            f'coefficient pairs, got {n}.')

    def refractive_index(wavelength):
        wl = wavelength * 1e6
        nsq = coefficients[0]

        for i in range(1, 9, 4):
            nsq = nsq + coefficients[i] * wl**coefficients[i + 1] / (wl**2 - coefficients[i + 2]**coefficients[i + 3])

        for i in range(9, len(coefficients) - 1, 2):
            nsq = nsq + coefficients[i] * wl**coefficients[i + 1]

        return np.sqrt(nsq)

    return refractive_index


def make_cauchy_index(coefficients):
    r'''The Cauchy dispersion formula for the refractive index of materials.

    The dispersion relation of the Cauchy formula is [Cauchy1836]_,

    .. math:: n = \sum_{i}{C_i \lambda^{k_i}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula. Coefficients come in
        (coefficient, exponent) pairs after the leading C0 term.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the coefficients do not consist of C0 followed by coefficient pairs.

    .. [Cauchy1836] A.-L. Cauchy 1836, Mémoire sur la dispersion de la lumière, J. G.
        Calve, Prague (1836)
    '''
    _check_pairs(coefficients, 'Cauchy')

    def refractive_index(wavelength):
        wl = wavelength * 1e6
        n = coefficients[0]

        for i in range(1, len(coefficients) - 1, 2):
            n = n + coefficients[i] * wl**coefficients[i + 1]

        return n

    return refractive_index


def make_gases_index(coefficients):
    r'''The gases dispersion formula for the refractive index of materials.

    The dispersion relation of the gases formula is [PeckReeder1972]_,

    .. math:: n = 1 + C_0 + \sum_{i}{\frac{C_i}{C_{i+1} - \lambda^{-2}}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the coefficients do not consist of C0 followed by coefficient pairs.

    .. [PeckReeder1972] E. R. Peck and K. Reeder 1972, Dispersion of air, J. Opt. Soc.
        Am. 62, 958–962 (1972)
    '''
    _check_pairs(coefficients, 'gases')

    def refractive_index(wavelength):
        wl = wavelength * 1e6
        n = 1.0 + coefficients[0]

        for i in range(1, len(coefficients) - 1, 2):
            n = n + coefficients[i] / (coefficients[i + 1] - wl**(-2))

        return n

    return refractive_index


def make_herzberger_index(coefficients):
    r'''The Herzberger dispersion formula for the refractive index of materials.

    The dispersion relation of the Herzberger formula is [HerzbergerSalzberg1962]_,

    .. math:: n = C_0 + \frac{C_1}{\lambda^2 - 0.028} + \frac{C_2}{(\lambda^2 - 0.028)^2} + \sum_{i \geq 3}{C_i \lambda^{2(i-2)}}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the number of coefficients is not 5.

    .. [HerzbergerSalzberg1962] M. Herzberger and C. D. Salzberg 1962, Refractive indices
        of infrared optical materials and color correction of infrared lenses, J. Opt.
        Soc. Am. 52, 420–427 (1962)
    '''
    _check_count(coefficients, 5, 'Herzberger')

    def refractive_index(wavelength):
        wl = wavelength * 1e6

        n = coefficients[0]
        n = n + coefficients[1] / (wl**2 - 0.028)
        n = n + coefficients[2] / (wl**2 - 0.028)**2

        for i in range(3, len(coefficients)):
            n = n + coefficients[i] * wl**(2 * (i - 2))

        return n

    return refractive_index


def make_retro_index(coefficients):
    r'''The retro dispersion formula for the refractive index of materials.

    The dispersion relation of the retro formula is [Polyanskiy2024]_,

    .. math:: n = \sqrt{\frac{2T + 1}{1 - T}}, \quad T = C_0 + \frac{C_1 \lambda^2}{\lambda^2 - C_2} + C_3 \lambda^2

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the number of coefficients is not 4.

    .. [Polyanskiy2024] M. N. Polyanskiy 2024, Refractiveindex.info database of optical
        constants, Scientific Data. 11, 94 (2024)
    '''
    _check_count(coefficients, 4, 'retro')

    def refractive_index(wavelength):
        wl = wavelength * 1e6

        T = coefficients[0] + coefficients[1] * wl**2 / (wl**2 - coefficients[2]) + coefficients[3] * wl**2

        return np.sqrt((2 * T + 1) / (1 - T))

    return refractive_index


def make_exotic_index(coefficients):
    r'''The exotic dispersion formula for the refractive index of materials.

    The dispersion relation of the exotic formula is [Rosker1985]_,

    .. math:: n^2 = C_0 + \frac{C_1}{\lambda^2 - C_2} + \frac{C_3 (\lambda - C_4)}{(\lambda - C_4)^2 + C_5}

    Parameters
    ----------
    coefficients : array_like
        The coefficients C0, C1, ... of the formula.

    Returns
    -------
    function
        The refractive index as a function of wavelength in meters.

    Raises
    ------
    ValueError
        If the number of coefficients is not 6.

    .. [Rosker1985] M. J. Rosker, K. Cheng, and C. L. Tang 1985, Practical urea optical
        parametric oscillator for tunable generation throughout the visible and
        near-infrared, IEEE J. Quantum Electron. 21, 1600–1606 (1985)
    '''
    _check_count(coefficients, 6, 'exotic')

    def refractive_index(wavelength):
        wl = wavelength * 1e6

        return np.sqrt(
            coefficients[0] + coefficients[1] / (wl**2 - coefficients[2]) + coefficients[3] * (wl - coefficients[4]) / ((wl - coefficients[4])**2 + coefficients[5])
        )

    return refractive_index
