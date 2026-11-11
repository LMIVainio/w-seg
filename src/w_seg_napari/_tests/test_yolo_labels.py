import numpy as np
import pandas as pd

from w_seg_napari._yolo_labels import yolo_class_names, yolo_label_map


def test_class_ids_start_at_one():
    assert yolo_class_names({0: 'xylem', 1: 'phloem'}) == {
        1: 'xylem',
        2: 'phloem',
    }
    assert yolo_class_names(['a', 'b']) == {1: 'a', 2: 'b'}
    assert yolo_class_names(None) == {}


def test_every_cell_gets_its_class():
    labels = np.zeros((4, 9), np.int32)
    labels[:, :3], labels[:, 3:6], labels[:, 6:] = 1, 2, 3
    df = pd.DataFrame(
        {
            'label_id': [1, 2, 3],
            'yolo_pred_tissue': ['xylem', 'phloem', None],
        }
    )
    out, names = yolo_label_map(labels, df, {0: 'xylem', 1: 'phloem'})
    assert names == {1: 'xylem', 2: 'phloem'}
    assert set(out[labels == 1]) == {1} and set(out[labels == 2]) == {2}
    assert set(out[labels == 3]) == {0}  # no prediction
    assert out.max() == 2


def test_names_come_from_the_table_without_a_model_map():
    labels = np.array([[1, 2]], np.int32)
    df = pd.DataFrame({'label_id': [1, 2], 'yolo_pred_tissue': ['b', 'a']})
    out, names = yolo_label_map(labels, df, None)
    assert names == {1: 'a', 2: 'b'} and list(out[0]) == [2, 1]


def test_no_prediction_column_gives_nothing():
    out, names = yolo_label_map(
        np.ones((2, 2), np.int32), pd.DataFrame({'label_id': [1]}), {}
    )
    assert out is None and names == {}
