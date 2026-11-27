import numpy as np
import pytest

from w_seg_napari._class_copy import copy_classes, parse_class_list


def test_parse_class_list():
    assert parse_class_list('') is None
    assert parse_class_list('  ') is None
    assert parse_class_list(None) is None
    assert parse_class_list('1,2,4') == {1, 2, 4}
    assert parse_class_list('[1, 2, 4]') == {1, 2, 4}
    assert parse_class_list('1-3; 7') == {1, 2, 3, 7}
    assert parse_class_list('5-3') == {3, 4, 5}
    with pytest.raises(ValueError):
        parse_class_list('1, x')


def test_copy_classes_keeps_cell_separation_and_filters():
    template = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])  # 0 = boundary
    classes = np.array([[4, 4, 4, 5, 5], [4, 4, 4, 5, 5]])
    out = copy_classes(template, classes)
    assert out.tolist() == [[4, 4, 0, 5, 5], [4, 4, 0, 5, 5]]
    kept = copy_classes(template, classes, keep={5})
    assert kept.tolist() == [[0, 0, 0, 5, 5], [0, 0, 0, 5, 5]]
    assert kept.dtype == np.uint32
    with pytest.raises(ValueError):
        copy_classes(template, classes[:, :4])
