import numpy as np
import pandas as pd

from w_seg_napari._geometry import (
    data_to_world,
    label_frame_to_image_pixels,
    spatial_shape,
    world_to_data,
    yx,
)


def test_world_data_roundtrip():
    idx = np.array([0, 5, 10])
    w = data_to_world(idx, scale=(2.0, 0.5), translate=(10.0, 3.0), axis=1)
    assert np.allclose(w, [3.0, 5.5, 8.0])
    assert np.allclose(
        world_to_data(w, scale=(2.0, 0.5), translate=(10.0, 3.0), axis=1), idx
    )


def test_yx_takes_last_two_axes():
    assert list(yx([1, 2, 3], 1.0)) == [2.0, 3.0]
    assert list(yx(None, 0.0)) == [0.0, 0.0]


def _one_cell_table(cy, cx, spacing=1.0):
    # a cell centred at (cy, cx) in label pixels, 10 x 10 pixels large
    return pd.DataFrame(
        {
            'bbox-0': [int(cy) - 5],
            'bbox-1': [int(cx) - 5],
            'bbox-2': [int(cy) + 5],
            'bbox-3': [int(cx) + 5],
            'centroid-0': [cy * spacing],
            'centroid-1': [cx * spacing],
            'minor_axis_length': [10 * spacing],
        }
    )


def test_cropped_labels_map_back_to_the_full_image():
    # Labels were cropped from the full image at row 30, column 20 (what the
    # ROI tool does: translate = crop origin). A cell at (15, 25) in the
    # cropped labels sits at (45, 45) in the full image.
    df = _one_cell_table(15, 25)
    out = label_frame_to_image_pixels(
        df, labels_translate=(30.0, 20.0), image_translate=(0.0, 0.0)
    )
    assert out['centroid-0'][0] == 45 and out['centroid-1'][0] == 45
    assert out['bbox-0'][0] == 40 and out['bbox-3'][0] == 50


def test_pixel_spacing_does_not_move_the_crop():
    # centroid and axis length are measured in micrometres, bbox in pixels
    df = _one_cell_table(15, 25, spacing=0.25)
    out = label_frame_to_image_pixels(df, pixel_spacing=0.25)
    assert np.isclose(out['centroid-0'][0], 15) and np.isclose(
        out['centroid-1'][0], 25
    )
    assert np.isclose(out['minor_axis_length'][0], 10)


def test_different_pixel_size_between_layers():
    # image has twice as many pixels per world unit as the labels
    df = _one_cell_table(10, 10)
    out = label_frame_to_image_pixels(
        df, labels_scale=(2.0, 2.0), image_scale=(1.0, 1.0)
    )
    assert np.isclose(out['centroid-0'][0], 20)
    assert np.isclose(out['minor_axis_length'][0], 20)


def test_same_layers_are_left_alone():
    df = _one_cell_table(15, 25)
    out = label_frame_to_image_pixels(df)
    assert np.allclose(out[df.columns].to_numpy(float), df.to_numpy(float))


def test_spatial_shape():
    assert spatial_shape(np.zeros((4, 5))) == (4, 5)
    assert spatial_shape(np.zeros((4, 5, 3))) == (4, 5)
    assert spatial_shape([1, 2]) is None
