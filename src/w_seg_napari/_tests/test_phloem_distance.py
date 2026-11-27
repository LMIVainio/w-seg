import numpy as np
import pandas as pd
import pytest

from w_seg_napari._phloem_distance import (
    compare_with_manual,
    image_key,
    load_tables,
    measure_all,
    measure_image,
    two_poles,
)


def _root(rng, ph_dist=100.0, tilt=0.0, n=8, jitter=4.0, stray=False):
    """Synthetic diarch root: phloem on the x axis, xylem on the y axis,
    both rotated by `tilt` radians, plus other cells (class 1)."""
    rot = np.array(
        [[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]]
    )
    c0 = np.array([300.0, 300.0])
    pts, cls = [], []
    for sign in (-1, 1):
        pole = c0 + rot @ np.array([sign * ph_dist / 2, 0.0])
        pts += list(pole + rng.normal(0, jitter, (n, 2)))
        cls += [3] * n
        pole = c0 + rot @ np.array([0.0, sign * 40.0])
        pts += list(pole + rng.normal(0, jitter, (n, 2)))
        cls += [4] * n
    pts += list(c0 + rng.uniform(-150, 150, (60, 2)))
    cls += [1] * 60
    if stray:
        pts.append(np.array([500.0, 20.0]))
        cls.append(3)
    pts = np.array(pts)
    return pd.DataFrame(
        {'label_id': range(len(pts)), 'centroid-0': pts[:, 1],
         'centroid-1': pts[:, 0], 'predicted_class': cls}
    )  # fmt: skip


def test_distance_recovered():
    df = _root(np.random.default_rng(0), ph_dist=120.0)
    r = measure_image(df, phloem_classes=[3])
    assert r['distance_px'] == pytest.approx(120.0, abs=3.0)
    assert r['pole1_cells'] == r['pole2_cells'] == 8
    assert r['flags'] == ''


def test_class_given_as_text_or_float():
    df = _root(np.random.default_rng(1))
    df['predicted_class'] = df['predicted_class'].astype(float)
    assert measure_image(df, ['3'])['n_phloem_cells'] == 16
    df['predicted_class'] = df['predicted_class'].map(
        {1.0: 'other', 3.0: 'Phloem', 4.0: 'Xylem'}
    )
    assert measure_image(df, ['phloem'])['n_phloem_cells'] == 16


def test_xylem_axis_and_angle():
    df = _root(np.random.default_rng(2), ph_dist=100.0, tilt=0.5)
    r = measure_image(df, [3], xylem_classes=[4])
    assert r['distance_xylem_px'] == pytest.approx(100.0, abs=4.0)
    assert r['angle_deg'] == pytest.approx(90.0, abs=6.0)


def test_xylem_line_corrects_sideways_offset():
    # one phloem pole is displaced along the xylem axis: the plain centre
    # distance is too long, the perpendicular distance stays near the truth
    df = _root(np.random.default_rng(3), ph_dist=100.0, jitter=1.0)
    is_pole2 = (df['predicted_class'] == 3) & (df['centroid-1'] > 300)
    df.loc[is_pole2, 'centroid-0'] += 30.0
    r = measure_image(df, [3], xylem_classes=[4])
    assert r['distance_px'] > 103
    assert r['distance_xylem_px'] == pytest.approx(100.0, abs=3.0)


def test_trim_removes_stray_cell():
    df = _root(np.random.default_rng(4), ph_dist=100.0, stray=True)
    no_trim = measure_image(df, [3], trim=0)
    trimmed = measure_image(df, [3], trim=3.0)
    assert abs(trimmed['distance_px'] - 100) < abs(
        no_trim['distance_px'] - 100
    )
    assert trimmed['n_dropped'] >= 1


def test_too_few_cells_is_flagged_not_an_error():
    df = _root(np.random.default_rng(5))
    r = measure_image(df, [99])
    assert np.isnan(r['distance_px']) and 'phloem' in r['flags']


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        measure_image(pd.DataFrame({'a': [1]}), [3])


def test_two_poles_sorted_left_to_right():
    xy = np.r_[
        np.random.default_rng(0).normal([200, 50], 3, (10, 2)),
        np.random.default_rng(1).normal([20, 50], 3, (10, 2)),
    ]
    assert two_poles(xy)['centres'][0, 0] < two_poles(xy)['centres'][1, 0]


def test_batch_scales_and_loading(tmp_path):
    rng = np.random.default_rng(6)
    big = pd.concat(
        [
            _root(rng, 100.0).assign(source_file='a_20x.tif'),
            _root(rng, 200.0).assign(source_file='b_40x.tif'),
        ]
    )
    big.to_csv(tmp_path / 'all.csv', index=False)
    tables = load_tables(tmp_path / 'all.csv')
    assert set(tables) == {'a_20x.tif', 'b_40x.tif'}
    res, details = measure_all(
        tables, scales={'a_20x': 1.0, 'b_40x': 0.5}, phloem_classes=[3]
    )
    um = dict(zip(res['image'], res['distance_um']))
    assert um['a_20x.tif'] == pytest.approx(100, abs=4)
    assert um['b_40x.tif'] == pytest.approx(100, abs=6)  # 200 px at 0.5 um
    assert set(details) == set(tables)


def test_per_image_csv_folder_with_comment_header(tmp_path):
    df = _root(np.random.default_rng(7))
    for n in ('x1', 'x2'):
        with open(tmp_path / f'{n}_morphology.csv', 'w') as f:
            f.write(
                '# === METADATA ===\n# Source Layer: y\n# === DATA TABLE ===\n'
            )
            df.to_csv(f, index=False)
    assert set(load_tables(tmp_path)) == {'x1_morphology', 'x2_morphology'}


def test_compare_with_manual():
    res = pd.DataFrame(
        {
            'image': ['a.tif', 'b.tif', 'c.tif'],
            'distance_um': [100.0, 120.0, 90.0],
        }
    )
    manual = pd.DataFrame(
        {'Image': ['a', 'B_morphology.csv', 'zzz'], 'val': [101.0, 118.0, 5.0]}
    )
    merged, stats, unmatched = compare_with_manual(res, manual, 'Image', 'val')
    assert stats['n'] == 2
    assert stats['bias'] == pytest.approx(0.5)
    assert stats['mean_abs_error'] == pytest.approx(1.5)
    assert set(unmatched) == {'c', 'zzz'}
    assert image_key('Folder/B_morphology.csv') == 'b'


def _stepwise_root(rng, pppp_poles=2, pp_poles=2, se_poles=2):
    """Phloem-type cells: class 'pppp' on `pppp_poles` poles (0, 1 or 2),
    'pp' (primary phloem) on `pp_poles`, 'se' on `se_poles`; other cells."""
    c0 = np.array([300.0, 300.0])
    pts, cls = [], []
    for name, n_poles in (
        ('pppp', pppp_poles),
        ('pp', pp_poles),
        ('se', se_poles),
    ):
        for sign in (-1, 1)[:n_poles]:
            pole = c0 + np.array([sign * 60.0, 0.0])
            pts += list(pole + rng.normal(0, 3.0, (5, 2)))
            cls += [name] * 5
    pts += list(c0 + rng.uniform(-150, 150, (40, 2)))
    cls += ['other'] * 40
    pts = np.array(pts)
    return pd.DataFrame(
        {'label_id': range(len(pts)), 'centroid-0': pts[:, 1],
         'centroid-1': pts[:, 0], 'predicted_class': cls}
    )  # fmt: skip


def test_levels_use_first_level_when_clear():
    from w_seg_napari._phloem_distance import pooling_levels

    df = _stepwise_root(np.random.default_rng(1))
    r = measure_image(df, phloem_levels=pooling_levels(['pppp', 'pp', 'se']))
    assert r['phloem_level'] == 'pppp'
    assert r['distance_px'] == pytest.approx(120.0, abs=3.0)
    assert 'unclear' not in r['flags']


def test_levels_add_next_class_when_first_has_one_pole():
    from w_seg_napari._phloem_distance import pooling_levels

    # pppp found on one side only; adding pp (both sides) gives two poles
    df = _stepwise_root(np.random.default_rng(2), pppp_poles=1, se_poles=0)
    r = measure_image(df, phloem_levels=pooling_levels(['pppp', 'pp', 'se']))
    assert r['phloem_level'] == 'pppp+pp'
    assert r['distance_px'] == pytest.approx(120.0, abs=5.0)


def test_levels_unclear_when_nothing_clusters():
    from w_seg_napari._phloem_distance import pooling_levels

    df = _stepwise_root(
        np.random.default_rng(3), pppp_poles=1, pp_poles=1, se_poles=1
    )
    r = measure_image(df, phloem_levels=pooling_levels(['pppp', 'pp', 'se']))
    assert r['phloem_level'] == 'unclear'
    assert 'phloem poles unclear' in r['flags']
    assert 'pppp:' in r['phloem_levels_tried']


def test_overlay_plot_runs(tmp_path):
    from w_seg_napari._phloem_distance import plot_overlay

    df = _root(np.random.default_rng(4))
    res, det = measure_all({'a': df}, phloem_classes=[3], xylem_classes=[4])
    img = np.zeros((600, 600, 3), np.uint8)
    plot_overlay('a', res.iloc[0], det['a'], img, tmp_path / 'a.png')
    assert (tmp_path / 'a.png').stat().st_size > 1000


def test_class_prefix_and_unmatched():
    from w_seg_napari._phloem_distance import select_cells, unmatched_classes

    df = pd.DataFrame(
        {'c': ['10_primary_phloem_pole', '04_primary_phloem', '07_periderm']}
    )
    assert len(select_cells(df, 'c', ['10'])) == 1
    assert len(select_cells(df, 'c', ['04_primary_phloem'])) == 1
    assert unmatched_classes({'a': df}, 'c', ['10', '10_phloem pole']) == [
        '10_phloem pole'
    ]


def test_load_tables_ignores_cleanup_logs(tmp_path):
    cells = pd.DataFrame({'centroid-0': [1.0], 'centroid-1': [2.0]})
    cells.to_csv(tmp_path / 'a_morphology.csv', index=False)
    pd.DataFrame({'label_id': [1]}).to_csv(tmp_path / 'a_cleanup.csv')
    assert list(load_tables(tmp_path)) == ['a_morphology']


def test_manual_poles_replace_clustering():
    from w_seg_napari._phloem_distance import poles_from_vertices

    # napari vertices are (row, column); the line goes right to left
    poles = poles_from_vertices([[300, 360], [300, 240]])
    assert poles == ((360.0, 300.0), (240.0, 300.0))
    assert poles_from_vertices([[1, 1]]) is None
    assert poles_from_vertices([[5, 5], [5, 5]]) is None

    df = _root(np.random.default_rng(0), ph_dist=120.0)
    res, det = measure_all(
        {'a': df},
        manual={'a': poles},
        phloem_classes=[3],
        xylem_classes=[4],
    )
    r = res.iloc[0]
    assert r['phloem_level'] == 'manual'
    assert r['distance_px'] == pytest.approx(120.0)
    assert r['pole1_x'] == 240.0  # left pole first
    assert 'manual poles' in r['flags']
    assert r['distance_xylem_px'] == pytest.approx(120.0, abs=4.0)
    assert len(det['a']['_phloem_xy']) == 0


def test_manual_overlay_runs(tmp_path):
    from w_seg_napari._phloem_distance import plot_overlay

    df = _root(np.random.default_rng(1))
    res, det = measure_all(
        {'a': df},
        manual={'a': ((240.0, 300.0), (360.0, 300.0))},
        phloem_classes=[3],
        xylem_classes=[4],
    )
    plot_overlay(
        'a', res.iloc[0], det['a'], np.zeros((600, 600, 3), np.uint8),
        tmp_path / 'a.png',
    )  # fmt: skip
    assert (tmp_path / 'a.png').exists()
