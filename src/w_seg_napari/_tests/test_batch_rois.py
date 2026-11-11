import numpy as np

import w_seg_napari._batch_rois as br


def test_regions_round_trip_and_bad_files_read_as_empty(tmp_path):
    assert br.load_rois(tmp_path) == {}
    br.save_rois(tmp_path, {'b.tif': [1, 2, 9, 9], 'a.tif': [0, 0, 5, 5]})
    assert br.load_rois(tmp_path) == {
        'a.tif': [0, 0, 5, 5],
        'b.tif': [1, 2, 9, 9],
    }
    (tmp_path / br.FILENAME).write_text('not json')
    assert br.load_rois(tmp_path) == {}
    br.save_rois(tmp_path, {})
    assert not (tmp_path / br.FILENAME).exists()


def test_regions_are_clipped_and_whole_images_mean_no_region():
    shape = (20, 30, 3)
    assert br.normalize_roi([-5, -5, 10, 12], shape) == [0, 0, 10, 12]
    assert br.normalize_roi([5, 5, 100, 100], shape) == [5, 5, 20, 30]
    assert br.normalize_roi([12, 20, 4, 6], shape) == [4, 6, 12, 20]
    assert br.normalize_roi([0, 0, 20, 30], shape) is None
    assert br.normalize_roi([3, 3, 3, 9], shape) is None  # no area
    assert br.normalize_roi(None, shape) is None


def test_region_from_a_drawn_rectangle():
    verts = [[2.4, 3.1], [2.4, 11.2], [9.7, 11.2], [9.7, 3.1]]
    assert br.roi_from_vertices(verts, (20, 30)) == [2, 3, 10, 12]


def test_crop_keeps_channels_and_makes_a_copy():
    img = np.arange(20 * 30 * 3).reshape(20, 30, 3)
    out = br.crop_image(img, [2, 3, 10, 12])
    assert out.shape == (8, 9, 3) and (out == img[2:10, 3:12]).all()
    out[0, 0, 0] = -1
    assert img[2, 3, 0] != -1
    assert br.crop_image(img, None) is img
