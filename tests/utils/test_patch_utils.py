import pytest

import hardpicks.utils.patch_utils as patch_utils


def test_patch_coord():
    coords = patch_utils.PatchCoord((0, 1), bottom_right=(5, 7))
    assert coords.tl == coords.top_left
    assert coords.br == coords.bottom_right
    assert coords.ndim == 2
    assert coords.tl == (0, 1)
    assert coords.br == (5, 7)
    assert coords.shape == (5, 6)
    assert coords.size == 30
    assert tuple(coords.dimrange) == tuple(range(2))
    assert coords.center_real == (2.5, 4)
    assert coords.center == (2, 4)
    coords2 = patch_utils.PatchCoord((0, 1), shape=(5, 6))
    assert coords == coords2
    assert coords2.tl == coords2.top_left
    assert coords2.br == coords2.bottom_right
    assert coords2.ndim == coords.ndim
    assert coords2.tl == coords.tl
    assert coords2.br == coords.br
    assert coords2.shape == coords.shape
    assert coords2.size == coords.size
    assert tuple(coords2.dimrange) == tuple(coords.dimrange)
    assert coords2.center_real == coords.center_real
    assert coords2.center == coords.center
    coords3 = patch_utils.PatchCoord((1, 1), shape=(5, 6))
    assert coords != coords3
    with pytest.raises(AssertionError):
        _ = patch_utils.PatchCoord((0, 0), (5, 5))
    with pytest.raises(AssertionError):
        _ = patch_utils.PatchCoord(top_left=(0, 0), bottom_right=(5, 5), shape=(5, 5))


def test_patch_coord_contains_and_intersects():
    rect1 = patch_utils.PatchCoord((5, 10), shape=(10, 10))  # 10x10 rect between (5,10) and (15,20)
    assert (0, 0) not in rect1
    assert (5, 5) not in rect1
    assert (5, 9) not in rect1
    assert (5, 10) in rect1
    assert (5, 11) in rect1
    assert (5, 19) in rect1
    assert (5, 20) not in rect1
    assert (10, 15) in rect1
    assert (14, 19) in rect1
    assert (15, 20) not in rect1
    assert sum([(y, x) in rect1 for y in range(30) for x in range(30)]) == rect1.size
    assert rect1.intersects(rect1) and rect1 in rect1
    rect2 = patch_utils.PatchCoord((5, 10), shape=(3, 3))
    assert rect2.intersects(rect1) and rect2 in rect1
    assert rect2.intersection(rect1) == rect2
    assert rect1.intersection(rect2) == rect2
    rect3 = patch_utils.PatchCoord((10, 10), shape=(10, 10))
    assert rect3.intersects(rect1) and rect3 not in rect1
    exp_inters = patch_utils.PatchCoord((10, 10), shape=(5, 10))
    assert rect3.intersection(rect1) == exp_inters
    assert rect1.intersection(rect3) == exp_inters
    rect4 = patch_utils.PatchCoord((8, 8), shape=(4, 4))
    exp_inters = patch_utils.PatchCoord((8, 10), shape=(4, 2))
    assert rect4.intersection(rect1) == exp_inters
    assert rect1.intersection(rect4) == exp_inters
    rect5 = patch_utils.PatchCoord((15, 20), shape=(100, 100))
    assert rect5.intersection(rect1) is None
    rect6 = patch_utils.PatchCoord((5, 10), shape=(0, 0))
    assert rect6.intersection(rect1) is None
