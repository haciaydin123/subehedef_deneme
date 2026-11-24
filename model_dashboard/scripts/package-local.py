"""Bundle the static dashboard, including its already-compressed downloads."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED,ZIP_STORED

site=Path(__file__).resolve().parents[1]
archive=site.parent/'outputs/hedef_dashboard.zip'
paths=[p for p in sorted((site/'dist').rglob('*')) if p.is_file()]
paths += [site/'README.md',site/'scripts/verify.mjs',Path(__file__).resolve()]
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p,Path('hedef_dashboard')/p.relative_to(site),compress_type=ZIP_STORED if p.suffix=='.zip' else ZIP_DEFLATED)
with ZipFile(archive) as z:
    assert z.testzip() is None
    assert 'hedef_dashboard/dist/index.html' in z.namelist()
print(archive,archive.stat().st_size,'bytes')
