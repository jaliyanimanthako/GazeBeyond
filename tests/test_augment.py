import numpy as np

from egospan.data.augment import SpanAugment, shift_zero_fill


def test_shift_zero_fill():
    a = np.zeros((4, 4, 4), int)
    a[0, 0, 0] = 1
    b = shift_zero_fill(a, (2, 1, 0))
    assert b[2, 1, 0] == 1 and b.sum() == 1
    c = shift_zero_fill(a, (-1, 0, 0))
    assert c.sum() == 0


def test_vertical_axis_never_flipped_or_swapped():
    for vertical in (1, 2):
        aug = SpanAugment(vertical_axis=vertical, max_shift=0, seed=0)
        a = np.zeros((2, 6, 6, 6), bool)
        top = [slice(None)] * 4
        top[1 + vertical] = 5
        a[tuple(top)] = True                                    # the whole top layer along the vertical axis
        for _ in range(30):
            (b,) = aug(a)
            assert b.sum() == a.sum()
            assert np.take(b, 5, axis=1 + vertical).all()


def test_same_transform_on_all_grids():
    aug = SpanAugment(vertical_axis=2, max_shift=2, seed=1)
    rng = np.random.default_rng(0)
    x = rng.random((3, 8, 8, 8)) > 0.7
    for _ in range(20):
        xa, ya = aug(x, x[:1].copy())
        assert (xa[:1] == ya).all()
