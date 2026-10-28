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


def test_get_companions_tolerates_hand_edited_json(tmp_path):
    from w_seg_napari._model_config import get_companions

    # companions written as a plain string = the YOLO file name
    assert get_companions({'companions': 'a.pt'})['yolo'] == 'a.pt'
    # dict form, missing key, and junk values
    assert get_companions({'companions': {'yolo': 'b.pt'}})['yolo'] == 'b.pt'
    assert get_companions({})['yolo'] is None
    assert get_companions({'companions': 5})['yolo'] is None
    # single .pt in the folder is picked up automatically
    (tmp_path / 'model.keras').write_text('x')
    (tmp_path / 'yolo_x.pt').write_text('x')
    assert get_companions({}, tmp_path)['yolo'] == 'yolo_x.pt'
    # two .pt files are ambiguous: nothing guessed
    (tmp_path / 'other.pt').write_text('x')
    assert get_companions({}, tmp_path)['yolo'] is None
