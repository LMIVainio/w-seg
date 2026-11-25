import numpy as np

from w_seg_napari import _tissue_mask as tm


def _blobs(h=200, w=400, boxes=((20, 20, 180, 150), (20, 250, 180, 380))):
    prob = np.zeros((h, w), dtype=np.float32)
    for y0, x0, y1, x1 in boxes:
        prob[y0:y1, x0:x1] = 0.95
    return prob


def test_prepare_input_range_and_shape():
    rng = np.random.default_rng(0)
    img = (rng.random((300, 700, 3)) * 255).astype(np.uint8)
    x = tm.prepare_input(img, size=64)
    assert x.shape == (64, 64, 3) and x.dtype == np.float32
    assert x.min() >= -1.0 and x.max() <= 1.0
    # 16 bit and float images give the same picture
    x16 = tm.prepare_input(img.astype(np.uint16) * 257, size=64)
    xf = tm.prepare_input(img.astype(np.float32) / 255.0, size=64)
    assert np.abs(x - x16).max() < 0.02 and np.abs(x - xf).max() < 0.02


def test_make_target_closes_gaps_and_fills_holes():
    wall = np.zeros((2000, 3000), dtype=np.uint8)
    # a tissue made of grid lines, 60 px apart, in one corner
    for y in range(200, 1400, 60):
        wall[y : y + 4, 300:2000] = 1
    for x in range(300, 2000, 60):
        wall[200:1400, x : x + 4] = 1
    target = tm.make_target(wall, size=500, close=4)
    assert target.shape == (500, 500)
    # inside the grid everything is tissue, far away nothing is
    assert (
        target[100:150, 100:200].all()
        or target[150:300, 150:400].mean() > 0.95
    )
    assert not target[400:, 400:].any()
    # a speck is removed
    wall2 = np.zeros((2000, 3000), dtype=np.uint8)
    wall2[10:14, 10:14] = 1
    assert not tm.make_target(wall2, size=500, close=0).any()


def test_lookup_fraction_and_tile_skipping():
    look = tm.TissueLookup(_blobs(), (2000, 4000), threshold=0.5)
    assert look.fraction(0, 0, 2000, 4000) > 0.2
    assert look.fraction(300, 300, 1500, 1400) == 1.0  # inside the left root
    assert look.fraction(0, 1600, 2000, 2400) == 0.0  # between the roots
    assert look.keep_tile(500, 500, 256)
    assert not look.keep_tile(500, 1700, 256)


def test_outside_cells_uses_the_inside_fraction():
    look = tm.TissueLookup(_blobs(), (200, 400), threshold=0.5)
    labels = np.zeros((200, 400), dtype=np.int32)
    labels[50:70, 50:70] = 1  # inside the left root
    labels[50:70, 180:200] = 2  # in the gap
    labels[50:70, 140:160] = 3  # half in, half out
    ids, frac = look.outside_cells(labels, min_inside=0.5)
    assert ids == [2] or ids == [2, 3]
    assert frac[1] == 1.0 and frac[2] == 0.0
    assert 0.2 < frac[3] < 0.8
    ids, _ = look.outside_cells(labels, min_inside=0.9)
    assert 3 in ids and 1 not in ids


def test_split_roots_one_box_per_piece_left_to_right():
    look = tm.TissueLookup(_blobs(), (2000, 4000), threshold=0.5)
    boxes = tm.split_roots(look, pad=0.0)
    assert len(boxes) == 2
    assert boxes[0][1] < boxes[1][1]
    (y0, x0, y1, x1) = boxes[0]
    assert 150 < x0 < 250 and 1400 < x1 < 1560


def test_split_roots_joins_overlapping_and_ignores_specks():
    prob = _blobs(boxes=((20, 20, 180, 150), (20, 160, 180, 300)))
    look = tm.TissueLookup(prob, (200, 400), threshold=0.5)
    assert len(tm.split_roots(look, pad=0.05)) == 1  # padding makes them touch
    speck = _blobs(boxes=((20, 20, 180, 300), (5, 380, 8, 383)))
    assert (
        len(tm.split_roots(tm.TissueLookup(speck, (200, 400)), pad=0.0)) == 1
    )
    empty = tm.TissueLookup(np.zeros((50, 50)), (300, 300))
    assert tm.split_roots(empty) == [(0, 0, 300, 300)]
