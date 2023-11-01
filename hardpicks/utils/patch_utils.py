"""Defines a nice little utility class to hold and convert raw tuples of patch coordinates."""

import typing


class PatchCoord:
    """Holds the N-dimensional coordinates of a patch.

    This utility class is meant to simplify the conversion of top-left/bottom-right coordinates into
    shape and center attributes.
    """

    def __init__(
        self,
        top_left: typing.Iterable[int],
        *args,  # should always remain empty! the parent needs to specify either shape or br corner!
        bottom_right: typing.Optional[typing.Iterable[int]] = None,
        shape: typing.Optional[typing.Iterable[int]] = None,
    ):
        """Constructs a patch coordinate object from the specified corners, or top-left + shape."""
        assert len(args) == 0, "all args should be specified with keywords after the first one!"
        assert bottom_right is not None or shape is not None, "need to specify BR corner or size!"
        assert bottom_right is None or shape is None, "cannot specify both BR corner AND size!"
        self._top_left = tuple(top_left)
        self._ndim = len(self._top_left)
        assert self._ndim > 0, "invalid 0-dim patch!"
        if bottom_right is not None:
            bottom_right = tuple(bottom_right)
            assert len(bottom_right) == self._ndim, "dimension count mismatch!"
            shape = tuple([bottom_right[d] - self._top_left[d] for d in range(self._ndim)])
            assert all([s >= 0 for s in shape]), "patch shape should never have negative dims!"
        else:
            shape = tuple(shape)
            assert len(shape) == self._ndim, "dimension count mismatch!"
            assert all([s >= 0 for s in shape]), "patch shape should never have negative dims!"
            bottom_right = tuple([self._top_left[d] + shape[d] for d in range(self._ndim)])
        self._shape = shape
        self._bottom_right = bottom_right

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the patch's coordinates."""
        return self._ndim

    @property
    def dimrange(self) -> typing.Iterable[int]:
        """Returns an iterator over the dimension range of the patch's coordinates."""
        return range(self._ndim)

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Returns the shape of the patch, which should be strictly positive along all dimensions."""
        return self._shape

    @property
    def size(self) -> int:
        """Returns the number of elements inside the patch."""
        size = 1
        for s in self._shape:
            size *= s
        return size

    @property
    def is_empty(self) -> bool:
        """Returns whether the patch is empty (i.e. whether it contains any element or not)."""
        return any([s == 0 for s in self._shape])

    @property
    def tl(self) -> typing.Tuple[int, ...]:
        """Returns the coordinates of the 'top-left' (closest-to-the-origin) corner of the patch."""
        return self._top_left

    @property
    def top_left(self) -> typing.Tuple[int, ...]:
        """Returns the coordinates of the 'top-left' (closest-to-the-origin) corner of the patch."""
        return self._top_left

    @property
    def br(self) -> typing.Tuple[int, ...]:
        """Returns the coordinates of the 'bottom-right' (farthest-from-the-origin) corner of the patch."""
        return self._bottom_right

    @property
    def bottom_right(self) -> typing.Tuple[int, ...]:
        """Returns the coordinates of the 'bottom-right' (farthest-from-the-origin) corner of the patch."""
        return self._bottom_right

    @property
    def center_real(self) -> typing.Tuple[float, ...]:
        """Returns the real-valued coordinates of the patch center."""
        return tuple([
            (self._bottom_right[d] + self._top_left[d]) / 2
            for d in range(self._ndim)
        ])

    @property
    def center(self) -> typing.Tuple[int, ...]:
        """Returns the floored integer coordinates of the patch center."""
        return tuple([
            int((self._bottom_right[d] + self._top_left[d]) / 2)
            for d in range(self._ndim)
        ])

    def center_roi(self, radius: int) -> typing.Tuple[slice, ...]:
        """Returns the slices tuple that corresponds to a central region of interest in the patch."""
        assert radius >= 0, "invalid center buffer radius (should be non-negative value!)"
        # note: this will NOT check to ensure that the buffered region remains smaller than the patch
        center = self.center
        return tuple([
            slice(center[d] - radius, center[d] + radius + 1)
            for d in range(self._ndim)
        ])

    def __eq__(self, other) -> bool:
        """Returns whether the specified object has the same coordinates as the current object."""
        assert isinstance(other, PatchCoord), "unsupported equality check with non-patch-coord obj"
        return self._top_left == other._top_left and self._bottom_right == other._bottom_right

    def __contains__(self, item) -> bool:
        """Returns whether the specified object is entirely contained in the current patch region."""
        if isinstance(item, (tuple, list)):  # this means 'item' is a single point
            assert len(item) == self._ndim, "invalid coordinates provided to contain check!"
            assert all([isinstance(c, (int, float)) for c in item]), "invalid coord scalar type"
            return all([
                self._top_left[d] <= item[d] < self._bottom_right[d]
                for d in range(self._ndim)
            ])
        elif isinstance(item, PatchCoord):  # here, we'll check for a rect-in-rect relationship
            assert item._ndim == self._ndim, "mismatch coordinate dimension count in contain check!"
            return all([
                self._top_left[d] <= item._top_left[d] and self._bottom_right[d] >= item._bottom_right[d]
                for d in range(self._ndim)
            ])
        else:
            raise AssertionError("unexpected object type")

    def intersects(self, item) -> bool:
        """Returns whether the specified object intersects the current patch region."""
        if isinstance(item, (tuple, list)):
            return item in self
        elif isinstance(item, PatchCoord):
            assert item._ndim == self._ndim, "mismatch coordinate dimension count in contain check!"
            return not any([
                item._bottom_right[d] <= self._top_left[d] or self._bottom_right[d] <= item._top_left[d]
                for d in range(self._ndim)
            ])
        else:
            raise AssertionError("unexpected object type")

    def intersection(self, item) -> typing.Optional["PatchCoord"]:
        """Returns the intersection between the current patch region and the provided object."""
        # note: if there is no intersection between the two patches, we'll return `None`
        if isinstance(item, PatchCoord):
            assert item._ndim == self._ndim, "mismatch coordinate dimension count in intersection!"
            br = tuple([min(item._bottom_right[d], self._bottom_right[d]) for d in range(self._ndim)])
            tl = tuple([max(item._top_left[d], self._top_left[d]) for d in range(self._ndim)])
            if any([tl[d] >= br[d] for d in range(self._ndim)]):
                return None
            return PatchCoord(top_left=tl, bottom_right=br)
        else:  # for now, we only handle patchcoord-like objects for this function...
            raise AssertionError("unexpected object type")

    @property
    def slice(self) -> typing.Tuple[slice, ...]:
        """Returns the slices tuple that corresponds to the per-dim axis ranges for this patch."""
        # note: be careful when indexing with negative indices, some libraries do different things!
        return tuple([slice(self._top_left[d], self.bottom_right[d]) for d in range(self._ndim)])

    def __repr__(self) -> str:
        """Returns a pretty/printable version of the patch coordinates."""
        return f"(tl={self._top_left}, br={self._bottom_right})"
