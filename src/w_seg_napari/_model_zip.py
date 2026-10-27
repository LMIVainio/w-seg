"""
Installing a model zip (standard library only).

A model zip holds one model bundle: model.keras and config.json, plus the
paired YOLO file when the model has one. The files can be at the top level of
the zip or inside a single folder.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError


def is_bundle_dir(folder) -> bool:
    folder = Path(folder)
    return (folder / 'model.keras').is_file() and (
        folder / 'config.json'
    ).is_file()


def _download(url: str, dest: Path):
    req = urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0 (Python/w-seg)'}
    )
    try:
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out:
            shutil.copyfileobj(response, out)
    except (URLError, HTTPError, OSError) as e:
        raise RuntimeError(f'Failed to download {url}: {e}') from e


def extract_safely(zip_path: Path, dest: Path):
    """Extract a zip, refusing any member that would land outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if target != dest and dest not in target.parents:
                raise RuntimeError(f'Unsafe path in model zip: {member}')
        zf.extractall(dest)


def install_zip(
    source: str, name: str, root: Path, overwrite: bool = False
) -> Path:
    """
    Install the zip at ``source`` (local file or URL) as ``root/name`` and
    return that folder. An already installed model is kept unless overwrite.
    """
    target = Path(root) / name
    if is_bundle_dir(target) and not overwrite:
        return target

    Path(root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        zip_path = Path(source)
        if not zip_path.is_file():
            zip_path = tmp / 'download.zip'
            print(f'[w-seg] Downloading model {name} from {source}...')
            _download(source, zip_path)

        extracted = tmp / 'extracted'
        extracted.mkdir()
        try:
            extract_safely(zip_path, extracted)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f'{source} is not a valid zip file') from e

        candidates = [
            c.parent
            for c in extracted.rglob('config.json')
            if is_bundle_dir(c.parent)
        ]
        if not candidates:
            raise RuntimeError(
                'The zip does not contain model.keras and config.json together'
            )
        # shallowest folder wins
        bundle = min(
            candidates, key=lambda d: len(d.relative_to(extracted).parts)
        )
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(bundle), str(target))
    return target
