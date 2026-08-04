import numpy as np
import pytest

from hcipy import get_fringe_visibility


@pytest.mark.parametrize(
    'intensity, expected',
    [
        ([0, 1, 0, 1], 1.0),
        ([1, 3, 1, 3], 0.5),
        ([2, 2, 2, 2], 0.0),
    ]
)
def test_get_fringe_visibility(intensity, expected):
    visibility = get_fringe_visibility(intensity)

    assert visibility == pytest.approx(expected)


def test_get_fringe_visibility_is_scaling_invariant():
    intensity = np.array([1, 2, 5, 3], dtype=float)

    visibility = get_fringe_visibility(intensity)
    scaled_visibility = get_fringe_visibility(10 * intensity)

    assert scaled_visibility == pytest.approx(visibility)


@pytest.mark.parametrize(
    'intensity, error_message',
    [
        ([], 'must not be empty'),
        ([0, 0, 0], 'undefined'),
        ([-1, 1, 2], 'must be non-negative'),
        ([1, np.nan, 2], 'must be finite'),
        ([1, np.inf, 2], 'must be finite'),
    ]
)
def test_get_fringe_visibility_rejects_invalid_input(
    intensity,
    error_message
):
    with pytest.raises(ValueError, match=error_message):
        get_fringe_visibility(intensity)
