"""Session files: saving corrections next to the batch results."""

import types

import numpy as np
import pytest

import w_seg_napari._session_io as sio

h5py = pytest.importorskip('h5py')


def _batch_arrays(h=12, w=16):
    return {
        'raw': np.zeros((h, w, 3), np.uint8),
        'pred_sem': np.full((h, w), 0.5, np.float32),
        'pred_aff': np.ones((2, h, w), np.float32),
        'sp_labels': np.ones((h, w), np.uint64),
        'gasp_labels': np.full((h, w), 2, np.uint64),
    }


def _make(tmp_path):
    path = tmp_path / 'a_session.h5'
    sio.save_session(path, _batch_arrays(), {'source_file': 'a.tif'})
    return path


class FakeLayer:
    def __init__(self, kind, name, data, **kw):
        self._type_string = kind
        self.name = name
        self.data = data
        self.opacity = 0.5
        self.scale = (1.0, 1.0)
        self.translate = (0.0, 0.0)
        self.visible = True
        self.blending = types.SimpleNamespace(value='additive')
        self.colormap = types.SimpleNamespace(name='gray', color_dict=kw.get('colors'))
        self.rgb = False
        self.shape_type = ['polygon'] * len(data) if kind == 'shapes' else []


def test_layer_to_payload_skips_other_layer_kinds():
    assert sio.layer_to_payload(FakeLayer('points', 'p', [])) is None


def test_session_round_trip(tmp_path):
    path = _make(tmp_path)
    store = sio.load_session(path)
    assert set(store) == set(_batch_arrays())
    assert sio.read_attrs(path)['source_file'] == 'a.tif'
    assert sio.session_kind(path) == 'session'


def test_corrections_are_added_and_originals_kept(tmp_path):
    path = _make(tmp_path)
    fixed = np.full((12, 16), 9, np.int32)
    sio.save_edits(path, corrected=fixed, corrected_layer='corrected_a')
    store = sio.load_session(path)
    assert (store[sio.CORRECTED] == 9).all()
    assert (store['gasp_labels'] == 2).all()
    assert sio.read_attrs(path)['corrected_layer'] == 'corrected_a'
    labels, origin = sio.editable_labels(store)
    assert origin == 'corrected' and (labels == 9).all()


def test_editable_labels_prefers_gasp_over_superpixels(tmp_path):
    labels, origin = sio.editable_labels(sio.load_session(_make(tmp_path)))
    assert origin == 'gasp' and (labels == 2).all()


def test_wrong_shape_is_refused_and_file_untouched(tmp_path):
    path = _make(tmp_path)
    with pytest.raises(ValueError):
        sio.save_edits(path, corrected=np.zeros((5, 5), np.int32))
    assert sio.CORRECTED not in sio.load_session(path)


def test_extra_layers_round_trip(tmp_path):
    path = _make(tmp_path)
    layer = FakeLayer(
        'labels', 'classes', np.arange(12 * 16, dtype=np.int32).reshape(12, 16),
        colors={1: [1.0, 0.0, 0.0, 1.0], None: [0, 0, 0, 0]},
    )
    shapes = FakeLayer('shapes', 'rois', [np.array([[0, 0], [0, 4], [4, 4]])])
    payloads = [
        sio.layer_to_payload(layer, order=0),
        sio.layer_to_payload(shapes, order=1),
    ]
    sio.save_edits(path, extras=payloads)
    back = sio.read_extras(path)
    assert [p['name'] for p in back] == ['classes', 'rois']
    assert back[0]['custom_label_colors'][1] == [1.0, 0.0, 0.0, 1.0]
    assert (back[0]['data'] == layer.data).all()
    assert back[0]['blending'] == 'additive' and back[0]['opacity'] == 0.5
    assert back[1]['shape_type'] == ['polygon']
    assert (back[1]['shapes'][0] == shapes.data[0]).all()


def test_batch_rerun_replaces_results_but_keeps_extras(tmp_path):
    path = _make(tmp_path)
    layer = FakeLayer('labels', 'classes', np.zeros((12, 16), np.int32))
    sio.save_edits(
        path,
        corrected=np.ones((12, 16), np.int32),
        extras=[sio.layer_to_payload(layer)],
    )
    arrays = _batch_arrays()
    arrays['gasp_labels'] = np.full((12, 16), 5, np.uint64)
    sio.save_session(path, arrays, {'source_file': 'a.tif'})
    store = sio.load_session(path)
    assert (store['gasp_labels'] == 5).all()
    assert sio.CORRECTED not in store  # not in `arrays`, so stale
    assert [p['name'] for p in sio.read_extras(path)] == ['classes']


def test_old_layer_export_is_recognised(tmp_path):
    path = tmp_path / 'old.h5'
    with h5py.File(path, 'w') as f:
        d = f.create_dataset('my layer', data=np.zeros((4, 4), np.uint16))
        d.attrs['layer_type'] = 'image'
        d.attrs['order'] = 0
        d.attrs['colormap'] = 'gray'
        d.attrs['blending_mode'] = 'additive'
    assert sio.session_kind(path) == 'layers'
    (p,) = sio.read_layer_file(path)
    assert p['name'] == 'my layer' and p['blending'] == 'additive'


def test_hand_made_session_holds_image_and_labels(tmp_path):
    path = tmp_path / 'mine_session.h5'
    img = np.zeros((12, 16, 3), np.uint8)
    lab = np.ones((12, 16), np.int32)
    sio.save_edits(path, corrected=lab, raw=img, source_name='mine')
    store = sio.load_session(path)
    assert set(store) == {'raw', sio.CORRECTED}
    assert sio.read_attrs(path)['source_file'] == 'mine'
    assert sio.session_kind(path) == 'session'
    with pytest.raises(ValueError):
        sio.save_edits(
            tmp_path / 'x_session.h5',
            corrected=np.ones((3, 3), np.int32),
            raw=img,
        )
