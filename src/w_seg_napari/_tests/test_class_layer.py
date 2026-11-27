import numpy as np
import pandas as pd

from w_seg_napari import _class_layer as cl


def _scene():
    labels = np.zeros((20, 40), int)
    labels[:, :10] = 1
    labels[:, 10:20] = 2  # touches cell 1, same class
    labels[:, 20:30] = 3
    original = np.zeros_like(labels)
    original[labels == 1] = 4
    original[labels == 2] = 4
    original[labels == 3] = 7
    df = pd.DataFrame(
        {
            'label_id': [1, 2, 3],
            'yolo_pred_tissue': ['04_primary_phloem'] * 2 + ['07_periderm'],
            'prob_10_primary_phloem_pole': [0.0] * 3,
            'prob_04_primary_phloem': [0.0] * 3,
            'prob_07_periderm': [0.0] * 3,
        }
    )
    return labels, original, df


def test_class_names_from_table():
    _, _, df = _scene()
    names = cl.class_names_from_table(df)
    assert names[10] == '10_primary_phloem_pole'
    assert names[4] == '04_primary_phloem'
    assert names[7] == '07_periderm'


def test_stroke_edits_only_the_cell_it_covers():
    labels, original, df = _scene()
    layer = original.copy()
    layer[5:15, 0:8] = 10  # most of cell 1
    layer[0:2, 18:22] = 10  # spill of a stroke over cell 2 and 3 (small)
    classes = cl.corrected_classes(labels, layer, original, min_changed=0.2)
    got = dict(zip(classes.label_id, classes.class_id))
    assert got == {1: 10, 2: 0, 3: 0}
    out, unknown = cl.add_corrected_column(df, classes)
    assert out['yolo_corrected'].tolist() == [
        '10_primary_phloem_pole',
        '04_primary_phloem',
        '07_periderm',
    ]
    assert out['yolo_edited'].tolist() == [True, False, False]
    assert unknown == []


def test_without_original_painted_pixels_count_and_unknown_id():
    labels, _, df = _scene()
    layer = np.zeros_like(labels)
    layer[labels == 3] = 55  # a class number the table does not have
    layer[labels == 1] = 4  # same as prediction: not edited
    classes = cl.corrected_classes(labels, layer)
    out, unknown = cl.add_corrected_column(df, classes)
    assert unknown == [55]
    assert out['yolo_edited'].tolist() == [False, False, False]
    assert out['yolo_corrected'].iloc[2] == '07_periderm'


def test_header_lines_survive(tmp_path):
    p = tmp_path / 'a_morphology.csv'
    p.write_text('# pixel size 0.5\n# x\nlabel_id,v\n1,2\n')
    head, df = cl.read_table_keep_header(p)
    df['new'] = 1
    cl.write_table_keep_header(p, head, df)
    text = p.read_text().splitlines()
    assert text[:2] == ['# pixel size 0.5', '# x']
    assert text[2] == 'label_id,v,new'
