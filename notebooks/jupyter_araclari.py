"""Path, execution and display helpers shared by the Jupyter notebooks.

Business calculations remain in python_kodlari; importing this module never
starts training, installs packages or writes a result.
"""
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import uuid
import zipfile


def project_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / 'python_kodlari/ayarlar').is_dir() and (candidate / 'model_dashboard/dist').is_dir():
            return candidate
    raise FileNotFoundError('Projenin tamamını indirin ve notebook’u projenin notebooks klasöründen açın.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def latest_run(stage):
    root = project_root()
    for parent in [root / 'outputs/jupyter' / stage, root / 'outputs' / stage]:
        pointer = parent / 'latest.json'
        if pointer.is_file():
            data = read_json(pointer)
            candidates = [parent / str(data.get('run_id', '')), parent / Path(data.get('path', '')).name]
            for path in candidates:
                path = path.resolve()
                if path != parent.resolve() and path.is_relative_to(parent.resolve()) and path.is_dir():
                    return path
        if (parent / 'manifest.json').is_file() or (parent / 'hazirlik_manifest.json').is_file():
            return parent
    return None


def fresh_run(stage):
    parent = project_root() / 'outputs/jupyter' / stage
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    return parent / (stamp + '_' + uuid.uuid4().hex[:8])


def record_run(stage, path):
    parent = project_root() / 'outputs/jupyter' / stage
    path = Path(path).resolve()
    if not path.is_dir() or path.parent != parent.resolve():
        raise ValueError('Yalnızca bu notebook çalışma alanındaki tamamlanmış klasör kaydedilebilir.')
    (parent / 'latest.json').write_text(json.dumps({'run_id': path.name, 'path': str(path)}, indent=2), encoding='utf-8')


def result_path(stage, relative, download):
    run = latest_run(stage)
    if run is not None:
        path = run / relative
        if not path.is_file():
            raise FileNotFoundError(f'Seçili koşuda beklenen sonuç eksik; tarihsel sonuçla karıştırılmadı: {path}')
        return path
    return project_root() / 'model_dashboard/dist/downloads' / download


def require_files(*paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('Gerekli dosyalar bulunamadı:\n' + '\n'.join(missing))


def read_table(path, **kwargs):
    import pandas as pd
    path = Path(path)
    require_files(path)
    with path.open('rb') as f:
        if f.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Git LFS dosyası henüz indirilmemiş. Proje kökünde git lfs pull çalıştırın.')
    return pd.read_csv(path, encoding='utf-8-sig', **kwargs)


@lru_cache(maxsize=1)
def snapshot():
    path = project_root() / 'model_dashboard/dist/data.js'
    text = path.read_text(encoding='utf-8')
    prefix = 'window.DASHBOARD_DATA='
    if not text.startswith(prefix):
        raise ValueError('Dashboard anlık görüntüsü beklenen JSON biçiminde değil.')
    return json.loads(text[len(prefix):].rstrip().removesuffix(';'))


def prepare_configs(input_path=None, overrides=None):
    """Copy all linked dictionaries together; resolve input paths before moving."""
    root = project_root()
    origin = root / 'python_kodlari/ayarlar'
    target = fresh_run('ayarlar')
    target.mkdir()
    for source in sorted(origin.glob('*.json')):
        config = copy.deepcopy((overrides or {}).get(source.name, read_json(source)))
        if 'input' in config and 'path' in config['input']:
            path = Path(input_path) if input_path is not None else origin / config['input']['path']
            config['input']['path'] = str(path.resolve())
        (target / source.name).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return target


def source_input(selection=None):
    root = project_root()
    selection = selection or latest_run('dinamik_feature_selection')
    if selection is not None and (Path(selection) / 'manifest.json').is_file():
        path = Path(read_json(Path(selection) / 'manifest.json').get('source_path', ''))
        if path.is_file():
            return path.resolve()
    return root / 'outputs/banka_test_verisi_v4/sube_model_girdileri_aylik.csv'


def run_python(script, *arguments):
    """Run a CLI in the active kernel's Python; no shell quoting or !python."""
    root = project_root()
    path = root / 'python_kodlari' / script
    require_files(path)
    command = [sys.executable, str(path), *map(str, arguments)]
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    if status:
        raise subprocess.CalledProcessError(status, command)


def extract_bundle(filename):
    """Extract a repository LFS package into a new, isolated local folder."""
    archive = project_root() / 'model_dashboard/dist/downloads' / filename
    require_files(archive)
    with archive.open('rb') as f:
        if f.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Önce proje kökünde git lfs pull çalıştırın.')
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            name = Path(item.filename)
            mode = item.external_attr >> 16
            if name.is_absolute() or '..' in name.parts or (mode & 0o170000) == 0o120000 or '\\' in item.filename:
                raise ValueError('Arşivde geçersiz dosya yolu var.')
        target = fresh_run('arsivler')
        target.mkdir()
        bundle.extractall(target)
    children = list(target.iterdir())
    return children[0] if len(children) == 1 and children[0].is_dir() else target


def current_decision():
    return read_json(result_path('otomatik_kumeleme', 'decision.json', 'auto_cluster_decision.json'))


def qualified_selection():
    run = latest_run('otomatik_kumeleme')
    if run is None or not read_json(run / 'decision.json').get('selected'):
        return None
    return run
