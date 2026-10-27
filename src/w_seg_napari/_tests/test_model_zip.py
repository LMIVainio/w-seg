import json
import zipfile

import pytest

from w_seg_napari._model_zip import install_zip, is_bundle_dir


def _make_zip(path, members):
    with zipfile.ZipFile(path, 'w') as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def test_install_flat_zip(tmp_path):
    z = tmp_path / 'm.zip'
    _make_zip(
        z, {'model.keras': 'x', 'config.json': json.dumps({}), 'yolo.pt': 'y'}
    )
    target = install_zip(str(z), 'm', tmp_path / 'models')
    assert is_bundle_dir(target)
    assert (target / 'yolo.pt').read_text() == 'y'


def test_install_zip_with_folder(tmp_path):
    z = tmp_path / 'm.zip'
    _make_zip(z, {'inner/model.keras': 'x', 'inner/config.json': '{}'})
    target = install_zip(str(z), 'm', tmp_path / 'models')
    assert is_bundle_dir(target)


def test_installed_model_is_kept_unless_overwrite(tmp_path):
    z = tmp_path / 'm.zip'
    _make_zip(z, {'model.keras': 'one', 'config.json': '{}'})
    target = install_zip(str(z), 'm', tmp_path / 'models')
    _make_zip(z, {'model.keras': 'two', 'config.json': '{}'})
    install_zip(str(z), 'm', tmp_path / 'models')
    assert (target / 'model.keras').read_text() == 'one'
    install_zip(str(z), 'm', tmp_path / 'models', overwrite=True)
    assert (target / 'model.keras').read_text() == 'two'


def test_zip_without_bundle_is_refused(tmp_path):
    z = tmp_path / 'm.zip'
    _make_zip(z, {'model.keras': 'x'})
    with pytest.raises(RuntimeError):
        install_zip(str(z), 'm', tmp_path / 'models')
    assert not (tmp_path / 'models' / 'm').exists()


def test_zip_cannot_write_outside_the_folder(tmp_path):
    z = tmp_path / 'm.zip'
    _make_zip(z, {'../evil.txt': 'x', 'model.keras': 'x', 'config.json': '{}'})
    with pytest.raises(RuntimeError):
        install_zip(str(z), 'm', tmp_path / 'models')
    assert not (tmp_path / 'models' / 'evil.txt').exists()
    assert not (tmp_path / 'evil.txt').exists()
