"""Training augmentation (Appendix A.2): axis permutation and flipping excluding upside-down orientations, plus random
translation of up to 2 voxels. The same transform is applied to every grid of a sample (inputs and target).

Permutations and flips act only on the two horizontal axes, so the vertical axis (2 in the gravity-aligned world
frame, 1 in the CPF frame) is never swapped or flipped. That leaves the 8 symmetries of the square. Translation shifts
all three axes by an integer in [-max_shift, max_shift] and fills with zeros.
"""
import numpy as np


def shift_zero_fill(a, shift):
    """Shift the last three axes of `a` by integer offsets, filling with zeros (no wrap-around)."""
    out = np.zeros_like(a)
    src, dst = [], []
    for n, s in zip(a.shape[-3:], shift):
        s = int(s)
        src.append(slice(0, n - s) if s >= 0 else slice(-s, n))
        dst.append(slice(s, n) if s >= 0 else slice(0, n + s))
    out[(Ellipsis, *dst)] = a[(Ellipsis, *src)]
    return out


def transform_vectors(v, perm, flips, vector_slices=(), pseudo_slices=()):
    """Apply the grid's axis permutation / flips to per-step vectors in the grid frame, e.g. gaze directions.
    Directions transform with M, rotation vectors (pseudovectors) with det(M) * M, where M maps old to new axes."""
    M = np.eye(3)[list(perm)]
    for f in flips:
        M[f] *= -1
    v = np.array(v, copy=True)
    for a, b in vector_slices:
        v[..., a:b] = v[..., a:b] @ M.T
    for a, b in pseudo_slices:
        v[..., a:b] = np.linalg.det(M) * (v[..., a:b] @ M.T)
    return v


class SpanAugment:
    def __init__(self, vertical_axis=2, max_shift=2, permute=True, flip=True, seed=None, fixed_axes=()):
        """fixed_axes: further axes that must not be permuted or flipped (e.g. the forward axis of a front-half
        crop, whose grid is not cubic and whose back/front must not swap)."""
        assert vertical_axis in (0, 1, 2)
        self.vertical = vertical_axis
        self.horizontal = [a for a in (0, 1, 2) if a != vertical_axis and a not in fixed_axes]
        self.max_shift, self.permute, self.flip = max_shift, permute, flip
        self.rng = np.random.default_rng(seed)

    def sample(self):
        perm = [0, 1, 2]
        if self.permute and len(self.horizontal) == 2 and self.rng.random() < 0.5:
            h0, h1 = self.horizontal
            perm[h0], perm[h1] = h1, h0
        flips = [a for a in self.horizontal if self.flip and self.rng.random() < 0.5]
        shift = self.rng.integers(-self.max_shift, self.max_shift + 1, 3) if self.max_shift else np.zeros(3, int)
        return perm, flips, shift

    @staticmethod
    def apply(a, perm, flips, shift):
        """a: (..., X, Y, Z)."""
        lead = a.ndim - 3
        a = np.transpose(a, list(range(lead)) + [lead + p for p in perm])
        if flips:
            a = np.flip(a, [lead + f for f in flips])
        if np.any(shift):
            a = shift_zero_fill(a, shift)
        return np.ascontiguousarray(a)

    def __call__(self, *grids):
        params = self.sample()
        return tuple(self.apply(g, *params) for g in grids)


class MirrorAugment:
    """Spherical grids: the only exact symmetry left is the left-right mirror (azimuth -> -azimuth, i.e. x -> -x),
    because the bins are uneven in angle (no shifts or 90 deg turns) and up / down and front / back differ."""

    def __init__(self, seed=None, p=0.5):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return [0, 1, 2], ([0] if self.rng.random() < self.p else []), np.zeros(3, int)

    apply = staticmethod(SpanAugment.apply)
