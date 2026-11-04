import numpy as np
import pandas as pd
import pytest
from PIL import Image

from w_seg_napari._model_config import (
    DEFAULT_YOLO_CROP,
    attach_yolo,
    get_companions,
    get_yolo_crop,
    load_config,
    make_config,
    save_config,
)
from w_seg_napari._yolo_data import (
    build_dataset,
    cell_table,
    class_folder,
    split_cells,
    split_images,
)


def _grid(n=3, size=20, gap=2, seed=0):
    """n x n square cells; class ids 1/2 alternate by column."""
    side = n * (size + gap) + gap
    labels = np.zeros((side, side), np.int32)
    classes = np.zeros_like(labels)
    k = 1
    for i in range(n):
        for j in range(n):
            y, x = gap + i * (size + gap), gap + j * (size + gap)
            labels[y : y + size, x : x + size] = k
            classes[y : y + size, x : x + size] = 1 + (j % 2)
            k += 1
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (side, side, 3), dtype=np.uint8)
    return image, labels, classes


def fake_crop(
    bbox,
    image_np,
    centroids=None,
    axis_l=None,
    target_size=256,
    context_factor=3.0,
    min_crop_size=64,
    max_crop_size=384,
    **kw,
):
    cy, cx = centroids
    t = np.zeros((3, 8, 8), np.float32)
    t[0] = (cy % 255) / 255.0  # encodes the position so tests can check it
    t[1] = (cx % 255) / 255.0
    return t


def test_cell_table_and_majority_class():
    image, labels, classes = _grid()
    classes[labels == 1] = 0
    classes[labels == 1][:5] = 0
    # cell 2: mostly class 2, a few painted class 1 pixels -> class 2
    cell2 = np.argwhere(labels == 2)
    classes[tuple(cell2[:10].T)] = 1
    df = cell_table(labels, classes)
    assert len(df) == 9
    assert {'bbox-0', 'bbox-3', 'centroid-0', 'minor_axis_length'} <= set(
        df.columns
    )
    by = df.set_index('label_id')['class_id']
    assert by[1] == 0
    assert by[2] == 2
    assert by[3] == 1
    cov = cell_table(labels, classes, min_coverage=2.0)
    assert (cov['class_id'] == 0).all()


def test_class_folder_order():
    assert class_folder(1, 'Pavement cell') == '01_Pavement_cell'
    assert sorted([class_folder(10, 'a'), class_folder(2, 'b')])[0].startswith(
        '02'
    )


def test_split_images_covers_every_class_on_both_sides():
    counts = {'a': {1: 10, 2: 5}, 'b': {1: 8, 2: 6}, 'c': {1: 12}}
    val = split_images(counts, 0.3, seed=0)
    assert val is not None and 0 < len(val) < 3
    # class 2 only exists in a and b, so val must hold exactly one of them
    assert len(val & {'a', 'b'}) == 1


def test_split_images_none_when_impossible():
    assert split_images({'a': {1: 5}}) is None
    assert (
        split_images({'a': {1: 5}, 'b': {2: 5}}) is None
    )  # no image holds both


def test_split_cells_stratified_and_single_cell_class():
    df = pd.DataFrame({'class_id': [1] * 10 + [2] * 10 + [3]})
    v = split_cells(df, 0.3, seed=1)
    assert v[:10].sum() == 3 and v[10:20].sum() == 3
    assert not v[20]


def test_build_dataset_by_image(tmp_path):
    samples = []
    for i in range(3):
        image, labels, classes = _grid(seed=i)
        samples.append(
            {
                'name': f'img{i}',
                'image': image,
                'labels': labels,
                'classes': classes,
            }
        )
    info = build_dataset(
        samples,
        {1: 'a', 2: 'b'},
        tmp_path / 'ds',
        crop_fn=fake_crop,
        val_fraction=0.34,
    )
    assert info['split_mode'] == 'by image'
    assert len(info['val_images']) == 1
    train = list((tmp_path / 'ds' / 'train').rglob('*.png'))
    val = list((tmp_path / 'ds' / 'val').rglob('*.png'))
    assert len(train) + len(val) == 27 and len(val) == 9
    assert all(p.name.startswith(info['val_images'][0]) for p in val)
    assert Image.open(val[0]).size == (8, 8)
    assert info['crop'] == DEFAULT_YOLO_CROP
    assert {p.parent.name for p in train} == {'01_a', '02_b'}


def test_build_dataset_single_image_warns(tmp_path):
    image, labels, classes = _grid(n=4)
    info = build_dataset(
        [
            {
                'name': 'only',
                'image': image,
                'labels': labels,
                'classes': classes,
            }
        ],
        {1: 'a', 2: 'b'},
        tmp_path / 'ds',
        crop_fn=fake_crop,
    )
    assert info['split_mode'] == 'by cell'
    assert any('One image only' in w for w in info['warnings'])
    n_val = len(list((tmp_path / 'ds' / 'val').rglob('*.png')))
    assert 0 < n_val < 16


def test_build_dataset_skips_unlisted_classes_and_checks(tmp_path):
    image, labels, classes = _grid()
    info = build_dataset(
        [{'name': 'x', 'image': image, 'labels': labels, 'classes': classes}],
        {1: 'only1'},
        tmp_path / 'ds',
        crop_fn=fake_crop,
    )
    assert info['n_cells'] == 6  # classes 2 cells skipped
    with pytest.raises(ValueError):
        build_dataset(
            [
                {
                    'name': 'x',
                    'image': image,
                    'labels': labels,
                    'classes': classes * 0,
                }
            ],
            {1: 'a'},
            tmp_path / 'e',
            crop_fn=fake_crop,
        )
    with pytest.raises(ValueError):
        build_dataset(
            [
                {
                    'name': 'x',
                    'image': image,
                    'labels': labels,
                    'classes': classes,
                }
            ]
            * 2,
            {1: 'a'},
            tmp_path / 'f',
            crop_fn=fake_crop,
        )


def test_layer_scale_maps_label_pixels_to_image_pixels(tmp_path):
    # labels at half resolution of the image (scale 2): crop centres must be
    # in image pixels, i.e. twice the label-pixel centroid
    image, labels, classes = _grid(n=2, size=10)
    big = np.kron(image, np.ones((2, 2, 1), np.uint8))
    seen = []

    def spy(bbox, image_np, centroids=None, **kw):
        seen.append(centroids)
        return np.zeros((3, 4, 4), np.float32)

    build_dataset(
        [
            {
                'name': 'x',
                'image': big,
                'labels': labels,
                'classes': classes,
                'labels_scale': (2, 2),
                'image_scale': (1, 1),
            }
        ],
        {1: 'a', 2: 'b'},
        tmp_path / 'ds',
        crop_fn=spy,
    )
    cy = sorted(c[0] for c in seen)
    from skimage.measure import regionprops

    want = sorted(round(2 * p.centroid[0]) for p in regionprops(labels))
    assert cy == want


def test_attach_yolo_copies_and_edits_config(tmp_path):
    src = tmp_path / 'bundle'
    src.mkdir()
    (src / 'model.keras').write_bytes(b'weights')
    (src / 'old.pt').write_bytes(b'old')
    save_config(src / 'config.json', make_config(name='m'))
    pt = tmp_path / 'best.pt'
    pt.write_bytes(b'new')
    out = attach_yolo(
        src,
        pt,
        tmp_path / 'out',
        {
            'crop': {'context_factor': 2.0},
            'class_names': {'0': '01_a', '1': '02_b'},
        },
    )
    assert sorted(p.name for p in out.iterdir()) == [
        'best.pt',
        'config.json',
        'model.keras',
    ]
    cfg = load_config(out / 'config.json')
    assert get_companions(cfg, out)['yolo'] == 'best.pt'
    crop = get_yolo_crop(cfg)
    assert crop['context_factor'] == 2.0 and crop['target_size'] == 256
    assert cfg['yolo']['class_names']['1'] == '02_b'
    # the source bundle is untouched
    assert (src / 'old.pt').exists()
    assert get_yolo_crop(load_config(src / 'config.json')) == DEFAULT_YOLO_CROP
    with pytest.raises(FileExistsError):
        attach_yolo(src, pt, out)
    with pytest.raises(ValueError):
        attach_yolo(src, pt, src)
    with pytest.raises(FileNotFoundError):
        attach_yolo(tmp_path / 'nope', pt, tmp_path / 'o2')
