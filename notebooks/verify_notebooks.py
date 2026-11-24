"""Validate notebooks; optionally execute a clean clone preview in real kernels.

python notebooks/verify_notebooks.py --execute --kernel python3
Executed outputs and the disposable clone are kept under .work, not committed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def clone_preview(destination):
    if (ROOT / '.git').exists():
        result = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                                cwd=ROOT, capture_output=True, check=True)
        names = set(result.stdout.decode().split('\0')) - {''}
    else:
        names = {str(p.relative_to(ROOT)) for folder in ['python_kodlari', 'notebooks', 'model_dashboard']
                 for p in (ROOT / folder).rglob('*') if p.is_file()
                 and not {'__pycache__', 'node_modules', '.ipynb_checkpoints'}.intersection(p.parts)}
        names.update(p.name for p in ROOT.iterdir() if p.is_file())
        names.update(line[2:] for line in (ROOT / '.gitignore').read_text().splitlines()
                     if line.startswith('!/outputs/') and not line.endswith('/') and '*' not in line)
    for name in names:
        source = ROOT / name
        if source.suffix == '.zip' or not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--kernel', default='python3')
    args = parser.parse_args()
    notebooks = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    assert len(notebooks) == 7
    count = 0
    for path in notebooks:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source, filename=path.name)
                assert cell.execution_count is None and cell.outputs == [], 'Commit notebooks without execution outputs'
                count += 1
    result = {'notebooks': len(notebooks), 'code_cells': count, 'schema_and_syntax': 'passed'}
    if args.execute:
        from nbclient import NotebookClient
        work = ROOT / '.work'
        work.mkdir(exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix='jupyter_verify_', dir=work))
        clean = session / 'project'
        clone_preview(clean)
        before = hashes(clean / 'outputs')
        executed = session / 'executed'
        executed.mkdir()

        def execute(name, replacements=None, label=None):
            book = nbformat.read(clean / 'notebooks' / name, as_version=4)
            for old, new in (replacements or {}).items():
                matches = [cell for cell in book.cells if cell.cell_type == 'code' and old in cell.source]
                assert len(matches) == 1, (name, old)
                matches[0].source = matches[0].source.replace(old, new)
            NotebookClient(book, timeout=600, kernel_name=args.kernel,
                           resources={'metadata': {'path': str(clean / 'notebooks')}}).execute()
            nbformat.write(book, executed / ((label or Path(name).stem) + '.ipynb'))
            print('PASS:', label or name, flush=True)
            return book

        for path in notebooks:
            execute(path.name)
        assert hashes(clean / 'outputs') == before, 'Preview changed source data or wrote model runs'
        blocked = execute('05_dea_ve_girdi_kontrolleri.ipynb',
                          {'DEA_CALISTIR = False': 'DEA_CALISTIR = True',
                           'GIRDI_KONTROLLERINI_CALISTIR = False': 'GIRDI_KONTROLLERINI_CALISTIR = True'},
                          'guard_no_qualified_dea')
        monthly = execute('06_yeni_ay_ve_dashboard.ipynb',
                          {'YENI_AYI_UYGULA = False': 'YENI_AYI_UYGULA = True'},
                          'guard_no_qualified_assignments')
        def streams(book):
            return '\n'.join(o.get('text', '') for c in book.cells if c.cell_type == 'code'
                             for o in c.outputs if o.output_type == 'stream')
        assert 'DEA başlatılmadı' in streams(blocked)
        assert 'Atama yapılmadı' in streams(monthly)
        assert hashes(clean / 'outputs') == before, 'Blocked stages wrote results'
        execute('01_veri_ve_egitim_setleri.ipynb',
                {'EGITIM_SETINI_HAZIRLA = False': 'EGITIM_SETINI_HAZIRLA = True'},
                'training_preparation')
        pointer = json.loads((clean / 'outputs/jupyter/model_karsilastirma_v4_veri/latest.json').read_text())
        preparation = Path(pointer['path'])
        import pandas as pd
        assert {split: len(pd.read_csv(preparation / f'veri/{split}.csv')) for split in ['train', 'validation', 'test']} == {'train': 3400, 'validation': 600, 'test': 600}
        assert len(pd.read_csv(preparation / 'veri/ocak2027_tahmin_girdisi.csv')) == 200
        result.update(real_kernel='passed', clean_clone_without_history='passed',
                      preview_did_not_write_results=True, no_qualified_model_guards='passed',
                      training_preparation='3400/600/600 + 200 forecast', executed_cases=10,
                      verification_directory=str(session))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
