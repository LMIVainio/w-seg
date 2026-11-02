import numpy as np

from w_seg_napari._geometry_fields import (
    clean_geometry,
    geometry_payloads,
    vector_glyphs,
)
from w_seg_napari._seeded_watershed import (
    label_seeds,
    map_seeds,
    seeded_edt_watershed,
)
from w_seg_napari._training.targets import compute_geometry


def _scene():
    """A dumbbell cell (two EDT maxima), a plain rectangle, a small square."""
    n = 120
    inst = np.zeros((n, n), np.int32)
    yy, xx = np.mgrid[:n, :n]
    for cx in (22, 58):  # two discs joined by a neck
        inst[(yy - 30) ** 2 + (xx - cx) ** 2 <= 14**2] = 1
    inst[26:35, 22:58] = 1  # neck, 9 px wide
    inst[66:110, 8:60] = 2  # rectangle
    inst[66:90, 72:100] = 3  # square
    return inst


def _fields(inst, pad=4):
    padded = np.pad(inst, pad)
    geo = compute_geometry(padded, inst.shape[0], pad)
    wall = (inst == 0).astype(np.float32)
    return wall, geo[..., 3], geo


def test_one_label_per_cell_even_for_two_peaks():
    inst = _scene()
    wall, skel, _ = _fields(inst)
    labels = seeded_edt_watershed(wall, skel, sigma=0.0, min_cell_size=10)
    assert len(np.unique(labels)) - 1 == 3
    # the dumbbell is ONE label
    assert len(np.unique(labels[inst == 1])) == 1
    # every label covers (almost) exactly one true cell
    for cell in (1, 2, 3):
        got = np.unique(labels[inst == cell])
        assert len(got) == 1 and got[0] > 0
    assert (labels[inst == 0] == 0).all()


def test_unseeded_cell_is_filled_or_left_out():
    inst = _scene()
    wall, skel, _ = _fields(inst)
    skel = skel.copy()
    skel[inst == 3] = 0  # the skeleton missed the small square
    kept = seeded_edt_watershed(wall, skel, sigma=0.0, fill_unseeded=True)
    dropped = seeded_edt_watershed(wall, skel, sigma=0.0, fill_unseeded=False)
    assert (kept[inst == 3] > 0).all()
    assert (dropped[inst == 3] == 0).all()
    assert len(np.unique(dropped)) - 1 == 2


def test_noise_pieces_and_threshold():
    inst = _scene()
    wall, skel, _ = _fields(inst)
    noisy = skel.copy()
    noisy[100, 70] = 1.0  # a lone pixel inside the background
    cell = wall < 0.35
    seeds = map_seeds(noisy, cell, 0.5, min_seed_px=5)
    assert seeds[100, 70] == 0  # outside cells and tiny
    assert seeds.max() == 3  # still one seed per cell
    assert map_seeds(np.zeros_like(skel), cell).max() == 0


def test_missing_wall_merges_seeds():
    """Documented weak spot: touching skeletons give one seed."""
    inst = np.zeros((100, 100), np.int32)
    inst[5:95, 5:48] = 1
    inst[5:95, 52:95] = 2
    wall, skel, _ = _fields(inst)
    skel = skel.copy()
    # An intact wall protects: a skeleton bridge across it is ignored ...
    skel[48:52, 20:80] = 1.0
    intact = seeded_edt_watershed(wall, skel, sigma=0.0)
    assert len(np.unique(intact)) - 1 == 2
    # ... but with a gap in the wall too, the two skeletons become one seed
    wall = wall.copy()
    wall[44:56, 46:54] = 0.0
    labels = seeded_edt_watershed(wall, skel, sigma=0.0)
    assert len(np.unique(labels)) - 1 == 1


def test_empty_inputs_and_shape_check():
    z = np.zeros((20, 20), np.float32)
    assert seeded_edt_watershed(np.ones_like(z), z).max() == 0
    assert seeded_edt_watershed(z, z, fill_unseeded=False).max() == 0
    try:
        seeded_edt_watershed(z, z[:10])
    except ValueError:
        pass
    else:
        raise AssertionError('shape mismatch not rejected')


def test_geometry_payloads_and_vectors():
    inst = _scene()
    _, _, geo_hwc = _fields(inst)
    geo = np.moveaxis(geo_hwc, -1, 0) * 1.3  # out of range on purpose
    clean = clean_geometry(geo)
    assert clean[0].max() <= 1 and clean[1:3].min() >= -1
    payloads = geometry_payloads(geo, 'img', vector_step=8)
    names = [p['name'] for p in payloads]
    assert names == [
        'geo_dist_img',
        'geo_skel_img',
        'geo_dir_img',
        'geo_vec_img',
    ]
    rgb = payloads[2]['data']
    assert rgb.shape == inst.shape + (3,) and rgb.min() >= 0 and rgb.max() <= 1
    vec = payloads[3]['data']
    assert vec.ndim == 3 and vec.shape[1:] == (2, 2) and len(vec) > 0
    # glyphs lie inside cells; lengths about 0.8 * step
    ys, xs = vec[:, 0, 0].astype(int), vec[:, 0, 1].astype(int)
    assert (inst[ys, xs] > 0).all()
    assert np.allclose(np.hypot(vec[:, 1, 0], vec[:, 1, 1]), 6.4, atol=0.1)
    # many pixels -> the grid is coarsened to respect max_vectors
    assert len(vector_glyphs(geo, step=1, max_vectors=50)) <= 50
    try:
        geometry_payloads(geo[:3], 'x')
    except ValueError:
        pass
    else:
        raise AssertionError('wrong channel count not rejected')


def test_edge_distance_as_seed_layer():
    """Edge distance works as seed map; a two-peak cell splits (documented)."""
    inst = _scene()
    wall, _, geo = _fields(inst)
    dist = geo[..., 0]
    labels = seeded_edt_watershed(
        wall, dist, sigma=0.0, seed_threshold=0.6, min_cell_size=10
    )
    # rectangle and square stay one cell each ...
    for cell in (2, 3):
        assert len(np.unique(labels[inst == cell])) == 1
    # ... the dumbbell has two distance peaks, hence two seeds
    assert len(np.unique(labels[inst == 1])) == 2
    # a high threshold keeps only the cell cores (smaller seeds, same cells)
    high = map_seeds(dist, wall < 0.35, 0.9, min_seed_px=1)
    low = map_seeds(dist, wall < 0.35, 0.5, min_seed_px=1)
    assert (high > 0).sum() < (low > 0).sum()


def test_labels_as_seed_layer_and_thinning():
    inst = _scene()
    wall, skel, _ = _fields(inst)
    # an existing label image as seeds: erode each cell to a core
    from scipy import ndimage as ndi

    cores = np.zeros_like(inst)
    for cell in (1, 2, 3):
        cores[ndi.binary_erosion(inst == cell, iterations=3)] = cell
    labels = seeded_edt_watershed(
        wall, seed_labels=cores, sigma=0.0, min_cell_size=10
    )
    assert len(np.unique(labels)) - 1 == 3
    for cell in (1, 2, 3):
        assert len(np.unique(labels[inst == cell])) == 1
    # label seeds that cross a wall are cut at the wall
    crossing = np.where(wall > 0.5, 0, 1)  # one label over everything
    cut = label_seeds(crossing * 7, wall < 0.35)
    assert cut.max() >= 3 and (cut[wall > 0.5] == 0).all()
    # thinning keeps one seed per piece
    thick = skel > 0.5
    thin = map_seeds(skel, wall < 0.35, 0.5, min_seed_px=1, thin=True)
    plain = map_seeds(skel, wall < 0.35, 0.5, min_seed_px=1, thin=False)
    assert thin.max() == plain.max()
    assert (thin > 0).sum() <= thick.sum()
    # no seed information at all is an error
    try:
        seeded_edt_watershed(wall)
    except ValueError:
        pass
    else:
        raise AssertionError('missing seeds not rejected')
