import numpy as np
import pytest

from hcipy import DeformableMirror, make_gaussian_influence_functions, make_pupil_grid


@pytest.fixture
def random_deformable_mirror():
    grid = make_pupil_grid(8)
    influence_functions = make_gaussian_influence_functions(grid, 2, 0.5)
    return DeformableMirror(influence_functions)


@pytest.mark.parametrize('seed', [42, [42, 123], np.random.SeedSequence(42)])
def test_deformable_mirror_random_seed(random_deformable_mirror, seed):
    dm = random_deformable_mirror
    dm.random(1e-7, seed=seed)
    actuators = dm.actuators.copy()
    surface = dm.surface.copy()

    dm.random(1e-7, seed=seed)
    np.testing.assert_array_equal(dm.actuators, actuators)
    np.testing.assert_array_equal(dm.surface, surface)

    dm.random(1e-7, seed=43)
    assert not np.array_equal(dm.actuators, actuators)
    assert not np.array_equal(dm.surface, surface)


@pytest.mark.parametrize('seed_factory', [np.random.default_rng, np.random.PCG64])
def test_deformable_mirror_random_stream(random_deformable_mirror, seed_factory):
    dm = random_deformable_mirror
    seed = seed_factory(42)
    reference = np.random.default_rng(seed_factory(42))

    for _ in range(2):
        dm.random(1e-7, seed=seed)
        expected = reference.standard_normal(dm.num_actuators) * 1e-7
        np.testing.assert_array_equal(dm.actuators, expected)


def test_deformable_mirror_random_amplitudes(random_deformable_mirror):
    dm = random_deformable_mirror
    dm.random(1, seed=42)
    unit_actuators = dm.actuators.copy()
    amplitudes = np.array([0, 1e-8, 2e-8, 3e-8])

    dm.random(amplitudes, seed=42)
    np.testing.assert_array_equal(dm.actuators, unit_actuators * amplitudes)
    np.testing.assert_array_equal(dm.surface, dm.influence_functions.linear_combination(dm.actuators))

    dm.random(0, seed=42)
    np.testing.assert_array_equal(dm.actuators, np.zeros(dm.num_actuators))
    np.testing.assert_array_equal(dm.surface, np.zeros(dm.input_grid.size))


@pytest.mark.parametrize('explicit_none', [False, True])
def test_deformable_mirror_random_legacy_stream(random_deformable_mirror, explicit_none):
    dm = random_deformable_mirror
    state = np.random.get_state()
    try:
        np.random.seed(42)
        expected = np.random.randn(dm.num_actuators) * 1e-7
        next_draw = np.random.randn()
        np.random.seed(42)
        if explicit_none:
            dm.random(1e-7, seed=None)
        else:
            dm.random(1e-7)
        np.testing.assert_array_equal(dm.actuators, expected)
        assert np.random.randn() == next_draw
    finally:
        np.random.set_state(state)


def test_deformable_mirror_random_seed_preserves_global_stream(random_deformable_mirror):
    state = np.random.get_state()
    try:
        expected = np.random.randn(4)
        np.random.set_state(state)
        random_deformable_mirror.random(1e-7, seed=42)
        np.testing.assert_array_equal(np.random.randn(4), expected)
    finally:
        np.random.set_state(state)
