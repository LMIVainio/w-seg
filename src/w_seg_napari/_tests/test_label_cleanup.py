import numpy as np
import pandas as pd
import pytest

from w_seg_napari import _label_cleanup as lc


def _labels():
    lab = np.zeros((20, 30), np.uint32)
    lab[0:8, 0:8] = 1  # touches the border
    lab[5:15, 10:20] = 2  # a real cell
    lab[5:15, 20:23] = 3  # a fragment next to 2
    lab[16:19, 24:27] = 4  # a speck that is not a cell
    return lab


def _fake_flag(monkeypatch, types):
    def flag_labels(labels, wall, bundle, pixel_size=1.0):
        ids = [i for i in np.unique(labels) if i > 0]
        scores = pd.DataFrame(
            {
                'label_id': ids,
                'p_error': [types.get(i, ('ok', 0.1))[1] for i in ids],
                'error_type': [types.get(i, ('ok', 0.1))[0] for i in ids],
                'flagged': [i in types for i in ids],
            }
        )
        return scores, None

    import w_seg_napari._segmentation_errors as se

    monkeypatch.setattr(se, 'flag_labels', flag_labels)


def test_border_cells_are_removed_and_logged():
    out, log = lc.cleanup_labels(_labels(), borders=True)
    assert not (out == 1).any() and (out == 2).any()
    assert list(log['action']) == ['border'] and log['label_id'][0] == 1
    assert log['area_px'][0] == 64


def test_nothing_requested_changes_nothing():
    lab = _labels()
    out, log = lc.cleanup_labels(lab)
    assert (out == lab).all() and log.empty
    assert lc.summarize(log) == 'nothing to change'


def test_fragments_merge_and_spurious_cells_are_deleted(monkeypatch):
    _fake_flag(
        monkeypatch,
        {3: ('fragment', 0.95), 4: ('spurious', 0.9), 2: ('merged', 0.99)},
    )
    out, log = lc.cleanup_labels(
        _labels(), wall=np.zeros((20, 30)), bundle=object(), mode='both'
    )
    assert not (out == 4).any()  # deleted
    assert (
        not (out == 3).any() or not (out == 2).any()
    )  # merged into 2 / 2 into 3
    assert (out == 2).any() or (out == 3).any()
    assert set(log['action']) == {'merge', 'delete'}
    assert 'merged' not in set(log['error_type'])  # too large: left alone


def test_mode_and_threshold_limit_what_happens(monkeypatch):
    _fake_flag(monkeypatch, {3: ('fragment', 0.7), 4: ('spurious', 0.95)})
    kw = {'wall': np.zeros((20, 30)), 'bundle': object()}
    out, log = lc.cleanup_labels(_labels(), mode='delete', min_p=0.8, **kw)
    assert list(log['action']) == ['delete'] and (out == 3).any()
    out, log = lc.cleanup_labels(_labels(), mode='merge', min_p=0.8, **kw)
    assert log.empty  # the fragment is below the limit
    out, log = lc.cleanup_labels(_labels(), mode='merge', min_p=0.6, **kw)
    assert list(log['action']) == ['merge']


def test_flagged_fix_needs_a_model():
    with pytest.raises(ValueError):
        lc.cleanup_labels(_labels(), mode='both')
