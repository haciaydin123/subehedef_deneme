"""Prepare the small Excel input snapshot, or package the verified V4 dataset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/banka_test_verisi_v4'

def read(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def prepare_summary():
    report=json.loads((OUT/'dogrulama_raporu.json').read_text())
    assert report['status']=='passed'
    examples=report['examples'];events=read('olay_ornekleri.csv')
    wanted={(e['donem'],e['musteri_id']) for e in examples}
    parts=[];customers=[]
    for name,result in [('musteri_sube_aylik.csv',parts),('musteri_aylik.csv',customers)]:
        with (OUT/name).open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if (r['donem'],r['musteri_id']) in wanted:result.append(r)
                if r['donem']>max(x[0] for x in wanted):break
    for e in examples:
        e['olay_tarihi']=next(r['olay_tarihi'] for r in events if r['donem']==e['donem'] and r['musteri_id']==e['musteri_id'] and r['olay_tipi']==e['olay'])
    payload={'bank':read('banka_aylik.csv'),'examples':examples,'customers':customers,'parts':parts,
             'manifest':json.loads((OUT/'manifest.json').read_text())}
    (ROOT/'.work/v4_summary_data.json').write_text(json.dumps(payload,ensure_ascii=False))
    print('Excel summary input prepared.')

def package():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['checks']['independent_event_replay_all_customer_months']
    # Re-check hashes immediately before packaging, without rerunning simulation/models.
    for name,spec in manifest['csv_files'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==spec['sha256']
    source=json.loads((ROOT/'outputs/banka_test_verisi_v3/manifest.json').read_text())
    manifest.update(demographic_source_url=source['source_url'],demographic_source_license=source['license_as_listed_by_source'])
    workbook=OUT/'banka_test_verisi_v4_ozet.xlsx'
    assert workbook.exists()
    manifest['summary_workbook']={'file':workbook.name,'bytes':workbook.stat().st_size,
       'sha256':hashlib.sha256(workbook.read_bytes()).hexdigest(),'formula_and_render_checks':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    code=OUT/'kod';code.mkdir(exist_ok=True)
    for name in ['test_verisi_v4_uret.py','test_verisi_v4_dogrula.py','test_verisi_v4_paketle.py']:
        shutil.copy2(ROOT/'python_kodlari'/name,code/name)
    archive=ROOT/'outputs/banka_test_verisi_v4_paketi.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.inspect.ndjson'):
                z.write(path,Path(OUT.name)/path.relative_to(OUT))
    with ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'registered_customers':manifest['registered_customers'],
       'latest_active_customers':manifest['latest_active_customers'],'event_counts':manifest['event_counts']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-summary',action='store_true');args=p.parse_args()
    prepare_summary() if args.prepare_summary else package()
