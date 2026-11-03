import json
import sys
import types
from pathlib import Path

import pytest

from w_seg_napari._model_config import load_config, make_config, save_config
from w_seg_napari._training import yolo


class FakeTrainer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.epochs = 3
        self.epoch = 0
        self.stop = False
        self.metrics = {}
        self.best = self.save_dir / 'weights' / 'best.pt'
        self.last = self.save_dir / 'weights' / 'last.pt'


class FakeYOLO:
    names = {0: '01_a', 1: '02_b'}

    def __init__(self, path):
        self.path = path
        self.callbacks = []
        self.trainer = None

    def add_callback(self, event, fn):
        assert event == 'on_fit_epoch_end'
        self.callbacks.append(fn)

    def train(self, data, project, name, **kw):
        self.kw = dict(kw, data=data)
        t = FakeTrainer(Path(project) / name)
        (t.save_dir / 'weights').mkdir(parents=True)
        t.best.write_bytes(b'best')
        t.last.write_bytes(b'last')
        (t.save_dir / 'results.csv').write_text('epoch,acc\n1,0.5\n')
        self.trainer = t
        for epoch, acc in enumerate([0.5, 0.8, 0.7]):
            t.epoch = epoch
            t.metrics = {'metrics/accuracy_top1': acc, 'val/loss': 1 - acc}
            for fn in self.callbacks:
                fn(t)
            if t.stop:
                break


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType('ultralytics')
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, 'ultralytics', mod)


def _dataset(tmp_path):
    for split in ('train', 'val'):
        for c in ('01_a', '02_b'):
            (tmp_path / 'ds' / split / c).mkdir(parents=True)
    return tmp_path / 'ds'


def _lines(capsys):
    out = capsys.readouterr().out.splitlines()
    return [json.loads(l[2:]) for l in out if l.startswith('@@')]


def test_train_writes_result_and_progress(tmp_path, capsys, fake_ultralytics):
    ds = _dataset(tmp_path)
    info = tmp_path / 'info.json'
    info.write_text(json.dumps({
        'crop': {'context_factor': 2.0}, 'split_mode': 'by image',
        'val_images': ['b'], 'folders': {'1': '01_a', '2': '02_b'},
        'counts': {'train': {'01_a': 5}, 'val': {'01_a': 2}}, 'n_cells': 7,
    }))
    yolo.main(['--dataset', str(ds), '--out-root', str(tmp_path / 'out'),
               '--name', 'tis', '--device', 'cpu', '--info-json', str(info)])
    ev = _lines(capsys)
    assert [e['event'] for e in ev] == ['start', 'progress', 'progress', 'progress', 'done']
    assert ev[2]['best_top1'] == 0.8 and ev[3]['best_top1'] == 0.8
    out = tmp_path / 'out' / 'tis'
    assert (out / 'tis.pt').read_bytes() == b'best'
    res = json.loads((out / 'yolo_result.json').read_text())
    assert res['best_val_top1'] == 0.8
    assert res['class_names'] == {'0': '01_a', '1': '02_b'}
    assert res['crop']['context_factor'] == 2.0
    assert res['dataset']['split_mode'] == 'by image'
    assert (out / 'results.csv').is_file()
    # a second run with the same name does not overwrite the first
    yolo.main(['--dataset', str(ds), '--out-root', str(tmp_path / 'out'),
               '--name', 'tis', '--device', 'cpu'])
    assert (tmp_path / 'out' / 'tis_2' / 'tis.pt').is_file()


def test_stop_file_ends_after_current_epoch(tmp_path, capsys, fake_ultralytics):
    ds = _dataset(tmp_path)
    stop = tmp_path / 'stop'
    stop.write_text('x')
    yolo.main(['--dataset', str(ds), '--out-root', str(tmp_path / 'out'),
               '--device', 'cpu', '--stop-file', str(stop)])
    ev = _lines(capsys)
    assert [e['event'] for e in ev].count('progress') == 1


def test_attach_to_bundle(tmp_path, capsys, fake_ultralytics):
    ds = _dataset(tmp_path)
    src = tmp_path / 'bundle'
    src.mkdir()
    (src / 'model.keras').write_bytes(b'm')
    save_config(src / 'config.json', make_config(name='m'))
    yolo.main(['--dataset', str(ds), '--out-root', str(tmp_path / 'out'),
               '--name', 'tis', '--device', 'cpu', '--attach-to', str(src),
               '--out-bundle', str(tmp_path / 'new_bundle')])
    done = _lines(capsys)[-1]
    assert done['bundle'] == str(tmp_path / 'new_bundle')
    cfg = load_config(tmp_path / 'new_bundle' / 'config.json')
    assert cfg['companions']['yolo'] == 'tis.pt'
    assert cfg['yolo']['training']['best_val_top1'] == 0.8
    assert cfg['yolo']['class_names'] == {'0': '01_a', '1': '02_b'}


def test_missing_dataset_and_args(tmp_path, fake_ultralytics):
    with pytest.raises(FileNotFoundError):
        yolo.main(['--dataset', str(tmp_path), '--out-root', str(tmp_path)])
    with pytest.raises(SystemExit):
        yolo.parse_args(['--dataset', 'a', '--out-root', 'b', '--attach-to', 'c'])
