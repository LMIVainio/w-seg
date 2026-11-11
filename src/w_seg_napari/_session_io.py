"""
Session files: everything the plugin keeps for one image in one h5 file.

    <folder>/wseg_batch/<image>_session.h5

Written by the batch run (predictions, labels, raw image) and extended when
you correct the result in napari ("Save session" in the Batch tab):

    raw, pred_sem, pred_aff, pred_geo      the image and the model's output
    sp_labels, gasp_labels, class_labels   what the batch computed
    labels_corrected                       your corrected cell labels
    extra/<layer>                          other layers you chose to keep
                                           (class layers, shapes, ...)
    attributes: source_file, model_config_json, fe_config_json, ...

The first group is owned by the batch run: a new run replaces it. Everything
else (extra/) is kept when the batch rewrites the file. Files are rewritten
through a temporary file and then moved into place, so an interrupted save
cannot damage an existing session.

Older "Export Napari Layers" files (one dataset per layer, no pipeline
arrays) can still be opened (read_layer_file); they hold layers only.

This module needs h5py and numpy only (no napari, no Qt): layers are handled
through a few attributes, so it is easy to test.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

SESSION_SUFFIX = '_session.h5'
CORRECTED = 'labels_corrected'
EXTRA_GROUP = 'extra'
# datasets that a batch run produces and replaces
MANAGED = (
    'raw',
    'pred_sem',
    'pred_aff',
    'pred_geo',
    'sp_labels',
    'gasp_labels',
    'class_labels',
    'yolo_labels',
    'error_map',
    'error_type_map',
)
OWNED = (*MANAGED, CORRECTED)


# ------------------------------------------------------------ plain arrays
def load_session(path) -> dict:
    """{name: array} of every top-level dataset of the file."""
    import h5py

    with h5py.File(path, 'r') as f:
        return {
            key: f[key][()] for key in f if isinstance(f[key], h5py.Dataset)
        }


def has_dataset(path, key) -> bool:
    import h5py

    with h5py.File(path, 'r') as f:
        return key in f


def read_attrs(path) -> dict:
    """The file's attributes as plain Python values."""
    import h5py

    out = {}
    with h5py.File(path, 'r') as f:
        for key, val in f.attrs.items():
            out[key] = _plain(val)
    return out


def _plain(val):
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _rewrite(
    path,
    arrays=None,
    drop=(),
    attrs=None,
    extras=None,
):
    """
    Write the file anew (through a temporary file): everything already in it
    is kept except the names in `drop`, the names in `arrays` (which are
    replaced) and, when `extras` is not None, the extra/ group (replaced by
    these layer payloads). Attributes are merged.
    """
    import h5py

    arrays = arrays or {}
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(tmp, 'w') as new:
            if path.is_file():
                with h5py.File(path, 'r') as old:
                    for key, val in old.attrs.items():
                        new.attrs[key] = val
                    for key in old:
                        if key in drop or key in arrays:
                            continue
                        if key == EXTRA_GROUP and extras is not None:
                            continue
                        old.copy(key, new)
            for key, val in arrays.items():
                new.create_dataset(key, data=val, compression='gzip')
            if extras is not None and extras:
                write_layers(new.create_group(EXTRA_GROUP), extras)
            for key, val in (attrs or {}).items():
                new.attrs[key] = val
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_session(path, arrays: dict, attrs: dict):
    """
    Store the results of a batch run. The pipeline arrays in the file are
    replaced by `arrays` (a name that is not in `arrays` is removed, so stale
    results cannot linger); extra/ is kept.
    """
    _rewrite(
        path,
        arrays=arrays,
        drop=[k for k in OWNED if k not in arrays],
        attrs=attrs,
    )


def base_name(source) -> str:
    """Image name without its file extension ('a.tif' -> 'a', 'sample 1.2' stays)."""
    from w_seg_napari._batch_io import VALID_EXTS

    p = Path(str(source))
    return p.stem if p.suffix.lower() in VALID_EXTS else p.name


LAYER_STATE_ATTR = 'layer_state_json'
# layers that come and go while reviewing; they are not part of a session
TRANSIENT_LAYERS = (
    'flagged_',
    'review_cell',
    'label_selection',
    'region',
    'region_image',
)


def _is_transient(name):
    return any(str(name).startswith(t) for t in TRANSIENT_LAYERS)


def layer_state_of(viewer):
    """Bottom to top: name, visible and opacity of every layer worth keeping."""
    return [
        {
            'name': la.name,
            'visible': bool(la.visible),
            'opacity': float(la.opacity),
        }
        for la in viewer.layers
        if not _is_transient(la.name)
    ]


def parse_layer_state(text):
    """The list stored by save_edits; [] when absent or unreadable."""
    if not text:
        return []
    try:
        state = json.loads(text)
    except (TypeError, ValueError):
        return []
    return [
        s
        for s in state
        if isinstance(s, dict) and isinstance(s.get('name'), str)
    ]


def apply_layer_state(viewer, state):
    """
    Restore visibility, opacity and order for the saved layers that are in the
    viewer. Returns how many saved layers were found.
    """
    found = [s for s in state if s['name'] in viewer.layers]
    for s in found:
        layer = viewer.layers[s['name']]
        if 'visible' in s:
            layer.visible = bool(s['visible'])
        if 'opacity' in s:
            layer.opacity = float(s['opacity'])
    for i, s in enumerate(
        found
    ):  # saved layers keep their order, at the bottom
        cur = viewer.layers.index(s['name'])
        if cur != i:
            viewer.layers.move(cur, i)
    return len(found)


def save_edits(
    path,
    corrected=None,
    corrected_layer=None,
    extras=None,
    raw=None,
    source_name=None,
    layer_state=None,
):
    """
    Add what you did in napari to a session file.

    corrected   int array: the corrected cell labels (stored as
                labels_corrected; the original labels stay untouched)
    extras      list of layer payloads (see layer_to_payload) that replace the
                extra/ group; None leaves it as it is, [] clears it
    raw         the image, for a session that is made by hand (no batch run
                behind it); stored as `raw`, with source_name as source_file
                so the batch can open it later
    layer_state list from layer_state_of(viewer): visibility, opacity and
                order of the viewer's layers, restored by apply_layer_state
    Raises ValueError when the labels do not fit the image in the file.
    """
    arrays = {}
    attrs = {'edited': datetime.now().isoformat(timespec='seconds')}
    if layer_state is not None:
        attrs[LAYER_STATE_ATTR] = json.dumps(layer_state)
    if raw is not None:
        arrays['raw'] = np.ascontiguousarray(raw)
        if source_name:
            attrs['source_file'] = str(source_name)
    if corrected is not None:
        corrected = np.asarray(corrected)
        if corrected.ndim != 2:
            raise ValueError('The corrected labels must be a 2D image.')
        path = Path(path)
        if raw is not None and tuple(corrected.shape) != tuple(
            np.shape(raw)[:2]
        ):
            raise ValueError(
                f'The labels layer has shape {tuple(corrected.shape)} but '
                f'the image layer has {tuple(np.shape(raw)[:2])}.'
            )
        if path.is_file() and raw is None:
            shape = _stored_shape(path)
            if shape is not None and tuple(corrected.shape) != shape:
                raise ValueError(
                    f'The labels layer has shape {tuple(corrected.shape)} but '
                    f'the session was made from an image of {shape}.'
                )
        arrays[CORRECTED] = corrected
        if corrected_layer:
            attrs['corrected_layer'] = str(corrected_layer)
    _rewrite(path, arrays=arrays, attrs=attrs, extras=extras)


def _stored_shape(path):
    """(H, W) of the image in a session file, None if there is none."""
    import h5py

    with h5py.File(path, 'r') as f:
        for key in ('pred_sem', 'gasp_labels', 'sp_labels'):
            if key in f:
                return tuple(f[key].shape[-2:])
        if 'raw' in f:
            return tuple(f['raw'].shape[:2])
    return None


# ---------------------------------------------------- what to show in napari
def editable_labels(store):
    """
    (array, origin) of the labels a person should edit: the saved corrected
    labels if there are any, otherwise a copy of the GASP labels, otherwise of
    the superpixel labels. (None, None) when the session has no labels.
    """
    if CORRECTED in store:
        return np.array(store[CORRECTED]), 'corrected'
    if 'gasp_labels' in store:
        return np.array(store['gasp_labels']), 'gasp'
    if 'sp_labels' in store:
        return np.array(store['sp_labels']), 'superpixels'
    return None, None


def session_kind(path):
    """'session' (this format), 'layers' (an old layer export) or 'unknown'."""
    import h5py

    with h5py.File(path, 'r') as f:
        keys = set(f.keys())
        if keys & set(OWNED) or EXTRA_GROUP in keys:
            return 'session'
        for key in keys:
            if 'layer_type' in f[key].attrs:
                return 'layers'
    return 'unknown'


def session_results(path):
    """
    Read a session file for display. Returns (results, store, info):
      results  what the batch hands to render_batch_results (raw, pred_sem,
               affinity_layers, geometry_layers, sp/gasp/class labels)
      store    the arrays of the file
      info     base (image name without extension), source_file, origin
               (which labels are offered for editing), has_corrected, extras
    """
    from w_seg_napari._affinity import layer_payloads
    from w_seg_napari._geometry_fields import geometry_payloads

    store = load_session(path)
    attrs = read_attrs(path)
    source = attrs.get('source_file') or Path(path).name.removesuffix(
        SESSION_SUFFIX
    )
    base = base_name(source)
    results = {}
    if 'raw' in store:
        results['raw'] = store['raw']
    if 'pred_sem' in store:
        results['pred_sem'] = np.squeeze(store['pred_sem'])
    cfg = None
    if attrs.get('model_config_json'):
        cfg = json.loads(attrs['model_config_json'])
    if cfg is not None and 'pred_aff' in store:
        results['affinity_layers'] = layer_payloads(
            store['pred_aff'], cfg, base
        )
    if 'pred_geo' in store:
        results['geometry_layers'] = geometry_payloads(store['pred_geo'], base)
    for key in (
        'sp_labels',
        'gasp_labels',
        'class_labels',
        'yolo_labels',
        'error_map',
        'error_type_map',
    ):
        if key in store:
            results[key] = store[key]
    if 'yolo_labels' in store and attrs.get('yolo_classes_json'):
        results['yolo_classes'] = {
            int(k): v
            for k, v in json.loads(attrs['yolo_classes_json']).items()
        }
    labels, origin = editable_labels(store)
    extras = read_extras(path)
    info = {
        'base': base,
        'source_file': source,
        'origin': origin,
        'has_corrected': CORRECTED in store,
        'extras': extras,
        'editable': labels,
        'layer_state': parse_layer_state(attrs.get(LAYER_STATE_ATTR)),
    }
    return results, store, info


# ------------------------------------------------------------ layer payloads
def _safe(name):
    return str(name).replace('/', '_')


def layer_to_payload(layer, order=0):
    """
    A plain dict for a napari Image, Labels or Shapes layer (None for any
    other kind). Works through attributes only, no napari import.
    """
    kind = getattr(layer, '_type_string', None)
    if kind not in ('image', 'labels', 'shapes'):
        return None
    blending = getattr(layer.blending, 'value', str(layer.blending))
    payload = {
        'name': layer.name,
        'layer_type': kind,
        'order': int(order),
        'opacity': float(layer.opacity),
        'scale': [float(s) for s in layer.scale],
        'translate': [float(t) for t in layer.translate],
        'visible': bool(layer.visible),
        'blending': str(blending),
    }
    if kind == 'shapes':
        payload['shapes'] = [np.asarray(s) for s in layer.data]
        payload['shape_type'] = list(layer.shape_type)
        return payload
    data = layer.data
    if getattr(layer, 'multiscale', False):
        data = data[0]
    payload['data'] = np.ascontiguousarray(data)
    payload['colormap'] = getattr(layer.colormap, 'name', str(layer.colormap))
    if kind == 'image':
        payload['rgb'] = bool(getattr(layer, 'rgb', False))
    else:
        colors = None
        cm = layer.colormap
        if hasattr(cm, 'color_dict') and cm.color_dict:
            colors = cm.color_dict
        if colors:
            payload['custom_label_colors'] = {
                str(k): [float(x) for x in np.asarray(v).ravel()]
                for k, v in colors.items()
                if k is not None
            }
    return payload


def write_layers(group, payloads):
    """Write layer payloads into an open h5 group (one dataset / group each)."""
    used = set()
    for index, p in enumerate(payloads):
        name = _safe(p['name'])
        n = 2
        while name in used:
            name = f'{_safe(p["name"])}_{n}'
            n += 1
        used.add(name)
        if p['layer_type'] == 'shapes':
            node = group.create_group(name)
            sg = node.create_group('shapes_data')
            for i, s in enumerate(p.get('shapes', [])):
                sg.create_dataset(f'shape_{i}', data=np.asarray(s))
            node.attrs['shape_types'] = json.dumps(p.get('shape_type', []))
        else:
            node = group.create_dataset(
                name,
                data=np.ascontiguousarray(p['data']),
                compression='gzip',
                compression_opts=4,
            )
            node.attrs['colormap'] = str(p.get('colormap', 'gray'))
            node.attrs['rgb'] = bool(p.get('rgb', False))
            if p.get('custom_label_colors'):
                node.attrs['custom_label_colors'] = json.dumps(
                    p['custom_label_colors']
                )
        node.attrs['layer_name'] = str(p['name'])
        node.attrs['layer_type'] = p['layer_type']
        node.attrs['order'] = int(p.get('order', index))
        node.attrs['opacity'] = float(p.get('opacity', 1.0))
        node.attrs['scale'] = np.asarray(p.get('scale', [1.0, 1.0]), float)
        node.attrs['translate'] = np.asarray(
            p.get('translate', [0.0, 0.0]), float
        )
        node.attrs['visible'] = bool(p.get('visible', True))
        node.attrs['blending'] = str(p.get('blending', 'translucent'))


def read_layers(group):
    """Layer payloads (sorted by their saved order) from an open h5 group."""
    import h5py

    out = []
    for key in group:
        item = group[key]
        a = item.attrs
        ltype = _plain(a.get('layer_type', 'image'))
        p = {
            'name': _plain(a.get('layer_name', key)),
            'layer_type': ltype,
            'order': int(_plain(a.get('order', 0))),
            'opacity': float(_plain(a.get('opacity', 1.0))),
            'scale': _optional_list(a.get('scale')),
            'translate': _optional_list(a.get('translate')),
            'visible': bool(_plain(a.get('visible', True))),
            # old exports called it blending_mode
            'blending': _plain(
                a.get('blending', a.get('blending_mode', 'translucent'))
            ),
        }
        if isinstance(item, h5py.Group) and ltype == 'shapes':
            p['shape_type'] = json.loads(_plain(a.get('shape_types', '[]')))
            shapes = []
            if 'shapes_data' in item:
                sg = item['shapes_data']
                for k in sorted(
                    sg.keys(),
                    key=lambda x: int(x.split('_')[-1]) if '_' in x else 0,
                ):
                    shapes.append(sg[k][()])
            p['shapes'] = shapes
        elif isinstance(item, h5py.Dataset):
            p['data'] = item[()]
            p['colormap'] = _plain(a.get('colormap', 'gray'))
            p['rgb'] = bool(_plain(a.get('rgb', False))) or (
                ltype == 'image'
                and item.ndim >= 3
                and item.shape[-1] in (3, 4)
                and item.dtype == np.uint8
            )
            if 'custom_label_colors' in a:
                try:
                    raw = json.loads(_plain(a['custom_label_colors']))
                    p['custom_label_colors'] = {
                        int(k): v for k, v in raw.items()
                    }
                except (ValueError, TypeError):
                    pass
        else:
            continue
        out.append(p)
    out.sort(key=lambda p: p['order'])
    return out


def _optional_list(val):
    return None if val is None else [float(x) for x in np.ravel(val)]


def read_extras(path):
    """Layer payloads of the extra/ group ([] if there is none)."""
    import h5py

    with h5py.File(path, 'r') as f:
        if EXTRA_GROUP not in f:
            return []
        return read_layers(f[EXTRA_GROUP])


def read_layer_file(path):
    """Layer payloads of an old 'Export Napari Layers' file (layers only)."""
    import h5py

    with h5py.File(path, 'r') as f:
        return read_layers(f)


def add_layers_to_viewer(viewer, payloads):
    """Add layer payloads to a napari viewer, in their saved order. Returns the names added."""
    names = []
    for p in payloads:
        scale, translate = p.get('scale'), p.get('translate')
        common = {
            'name': p['name'],
            'opacity': p['opacity'],
            'visible': p['visible'],
            'blending': p['blending'],
        }
        if scale:
            common['scale'] = tuple(scale)
        if translate:
            common['translate'] = tuple(translate)
        if p['layer_type'] == 'shapes':
            viewer.add_shapes(
                p.get('shapes', []),
                shape_type=p.get('shape_type') or 'polygon',
                **common,
            )
        elif p['layer_type'] == 'labels':
            kwargs = dict(common)
            colors = p.get('custom_label_colors')
            if colors:
                from napari.utils.colormaps import DirectLabelColormap

                kwargs['colormap'] = DirectLabelColormap(
                    color_dict=colors, default_color=(0.0, 0.0, 0.0, 0.0)
                )
            viewer.add_labels(p['data'], **kwargs)
        else:
            viewer.add_image(
                p['data'],
                colormap=p.get('colormap', 'gray'),
                rgb=bool(p.get('rgb')),
                **common,
            )
        names.append(p['name'])
    return names
