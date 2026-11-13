import json

import numpy as np
import pandas as pd

from w_seg_napari import _batch_summary as bs


def _folder(tmp_path, name, images, spacing=0.5, model='m1'):
    out = tmp_path / name / 'wseg_batch'
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for k, n in enumerate(images):
        pd.DataFrame(
            {
                'source_file': f'img{k}.tif',
                'labels_source': 'corrected',
                'label_id': np.arange(1, n + 1),
                'centroid-0': rng.random(n),
                'area': rng.random(n) * 100,
                'predicted_class': rng.integers(1, 3, n),
                'flagged': rng.random(n) > 0.8,
            }
        ).to_csv(out / f'img{k}_morphology.csv', index=False)
    (out / bs.SETTINGS_NAME).write_text(
        json.dumps(
            {
                'morph_params': {'pixel_spacing': spacing},
                'model': {'name': model},
            }
        )
    )
    return out


def test_columns_are_split_into_measurements_and_categories(tmp_path):
    out = _folder(tmp_path, 'a', [20])
    cells = bs.read_group('a', out)
    numeric, cats = bs.split_columns(cells)
    assert numeric == ['area']
    assert set(cats) == {'predicted_class', 'flagged'}


def test_image_is_the_unit_and_groups_are_built_from_images(tmp_path):
    a = _folder(tmp_path, 'a', [20, 30, 25])
    b = _folder(tmp_path, 'b', [40, 40])
    res = bs.build_summary([('lineA', a), ('lineB', b)])
    assert len(res['cells']) == 20 + 30 + 25 + 80
    assert list(res['images']['n_cells']) == [20, 30, 25, 40, 40]
    g = res['groups']
    row = g[(g['group'] == 'lineA') & (g['measure'] == 'n_cells')].iloc[0]
    assert row['n_images'] == 3
    assert abs(row['mean'] - 25) < 1e-9
    assert abs(row['sem'] - np.std([20, 30, 25], ddof=1) / np.sqrt(3)) < 1e-9
    shares = res['images'].filter(like='share_predicted_class=')
    assert np.allclose(shares.sum(axis=1), 1.0)


def test_warnings_for_scale_and_model_differences(tmp_path):
    a = _folder(tmp_path, 'a', [10], spacing=0.5, model='m1')
    b = _folder(tmp_path, 'b', [10], spacing=0.65, model='m2')
    w = ' '.join(bs.build_summary([('a', a), ('b', b)])['warnings'])
    assert 'pixel sizes' in w and 'different models' in w


def test_density_uses_the_analysed_area(tmp_path):
    a = _folder(tmp_path, 'a', [100], spacing=1.0)
    res = bs.build_summary([('a', a)], area_of=lambda d, im: 1_000_000)
    assert abs(res['images']['cells_per_mm2'].iloc[0] - 100) < 1e-6


def test_write_and_empty_folder(tmp_path):
    a = _folder(tmp_path, 'a', [10, 12])
    empty = tmp_path / 'e'
    empty.mkdir()
    res = bs.build_summary([('a', a), ('e', empty)])
    assert any('no _morphology.csv' in w for w in res['warnings'])
    paths = bs.write_summary(res, tmp_path / 'summary')
    assert set(paths) == {'cells', 'images', 'groups'}
    assert (tmp_path / 'summary' / 'summary_info.json').is_file()
