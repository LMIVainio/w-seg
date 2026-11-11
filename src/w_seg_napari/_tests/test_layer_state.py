import json

import w_seg_napari._session_io as sio


class _FakeLayer:
    def __init__(self, name, visible=True, opacity=1.0):
        self.name, self.visible, self.opacity = name, visible, opacity


class _FakeLayers(list):
    def __contains__(self, name):
        return any(la.name == name for la in self)

    def __getitem__(self, key):
        if isinstance(key, str):
            return next(la for la in self if la.name == key)
        return super().__getitem__(key)

    def index(self, name):
        return [la.name for la in self].index(name)

    def move(self, src, dst):
        self.insert(dst, self.pop(src))


class _FakeViewer:
    def __init__(self, layers):
        self.layers = _FakeLayers(layers)


def test_layer_state_round_trip_restores_order_visibility_opacity():
    saved = _FakeViewer(
        [
            _FakeLayer('geo'),
            _FakeLayer('image', opacity=0.5),
            _FakeLayer('flagged_x'),
            _FakeLayer('labels', visible=False),
        ]
    )
    state = sio.parse_layer_state(json.dumps(sio.layer_state_of(saved)))
    assert [s['name'] for s in state] == ['geo', 'image', 'labels']

    fresh = _FakeViewer(
        [
            _FakeLayer('labels'),
            _FakeLayer('other'),
            _FakeLayer('image'),
            _FakeLayer('geo'),
        ]
    )
    assert sio.apply_layer_state(fresh, state) == 3
    assert [la.name for la in fresh.layers][:3] == ['geo', 'image', 'labels']
    assert fresh.layers['image'].opacity == 0.5
    assert fresh.layers['labels'].visible is False


def test_parse_layer_state_tolerates_garbage():
    assert sio.parse_layer_state(None) == []
    assert sio.parse_layer_state('not json') == []
