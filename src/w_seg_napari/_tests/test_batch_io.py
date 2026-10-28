import numpy as np
import pytest

from w_seg_napari._batch_io import (
    list_input_files,
    load_image,
    output_dir,
    session_path,
    to_model_input_range,
    to_rgb,
)


def test_to_rgb_layouts():
    assert to_rgb(np.zeros((5, 6))).shape == (5, 6, 3)
    assert to_rgb(np.zeros((5, 6, 3))).shape == (5, 6, 3)
    assert to_rgb(np.zeros((3, 5, 6))).shape == (5, 6, 3)  # channels first
    assert to_rgb(np.zeros((5, 6, 4))).shape == (5, 6, 3)  # alpha dropped
    assert to_rgb(np.zeros((1, 5, 6, 1))).shape == (5, 6, 3)  # squeezed
    with pytest.raises(ValueError):
        to_rgb(np.zeros((5, 6, 7)))
    with pytest.raises(ValueError):
        to_rgb(np.zeros((2, 3, 4, 5)))


def test_channels_first_values_are_kept():
    chw = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    hwc = to_rgb(chw)
    assert hwc[2, 3, 1] == chw[1, 2, 3]


def test_list_input_files_skips_results_and_other_files(tmp_path):
    for name in ['b.tif', 'a.PNG', 'x_session.h5', 'notes.txt', 'c.h5']:
        (tmp_path / name).write_text('x')
    (tmp_path / 'folder.tif').mkdir()  # a folder is not an image
    out = output_dir(tmp_path)
    out.mkdir()
    (out / 'a_session.h5').write_text('x')
    assert list_input_files(tmp_path) == ['a.PNG', 'b.tif', 'c.h5']


def test_session_path_uses_the_image_stem(tmp_path):
    assert session_path(tmp_path, 'img.001.tif').name == 'img.001_session.h5'


def test_load_tiff(tmp_path):
    import tifffile

    arr = (np.random.default_rng(0).random((8, 9, 3)) * 255).astype(np.uint8)
    tifffile.imwrite(tmp_path / 'a.tif', arr)
    out = load_image(tmp_path / 'a.tif')
    assert out.shape == (8, 9, 3) and np.array_equal(out, arr)


def test_model_input_range():
    assert to_model_input_range(np.full((2, 2, 3), 255, np.uint8)).max() == 1.0
    f = np.full((2, 2, 3), 7.0, np.float32)
    assert to_model_input_range(f).max() == 7.0
