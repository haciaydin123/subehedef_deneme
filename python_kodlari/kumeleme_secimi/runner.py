"""Audited selector and frozen-model inference CLI; no clustering is refitted here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import joblib
import pandas as pd

from kumeleme.models import METHODS, predict
from .selection import validate_config,score_candidate,choose,monthly_metrics
from .coverage import build_contract,assignment_coverage

ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(path,rows):pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')
def check_hashes(directory,manifest):
    for name,digest in manifest['artifact_hashes'].items():
        target=(directory/name).resolve()
        if not target.is_relative_to(directory.resolve()) or sha(target)!=digest:raise ValueError(f'Artifact mismatch: {name}')


def resolve_coverage(config,config_path):
    spec=config.get('dea_coverage')
    if not spec or spec.get('require_full_coverage') is not True:raise ValueError('100% DEA branch coverage is required for clustering selection')
    parent=Path(config_path).resolve().parent;base_path=(parent/spec['base_config']).resolve();audit_path=(parent/spec['input_audit_config']).resolve()
    base=read(base_path);audit=read(audit_path)
    if (audit_path.parent/audit['base_config']).resolve()!=base_path:raise ValueError('DEA audit and clustering use different base definitions')
    if base.get('insufficient_cluster_policy')!='error':raise ValueError('DEA must stop on insufficient clusters instead of withholding branches')
    contract=build_contract(base,audit,config['thresholds']['min_cluster_size'])
    contract['source_definitions']={'base':base,'input_audit':audit}
    contract['source_hashes']={str(base_path):sha(base_path),str(audit_path):sha(audit_path)}
    config['thresholds']['min_cluster_size']=contract['required_min_cluster_size']
    return contract


def infer(run,panel,period=None):
    run=Path(run);manifest=read(run/'manifest.json');check_hashes(run,manifest)
    decision=read(run/'decision.json');winner=decision['selected']
    if winner is None:raise ValueError('No qualified main model; inference is disabled')
    contract=read(run/'dea_coverage_contract.json')
    schema=read(run/'selected_schema.json');types={f['name']:str for f in schema['fields'] if f['dtype']=='categorical'}
    frame=pd.read_csv(panel,dtype={'__branch':str,'__period':str,'__group':str,**types})
    keys=['__branch','__period','__group']
    if not set(keys+schema['features'])<=set(frame):raise ValueError('Missing prepared input columns')
    period=period or frame['__period'].max();frame=frame[frame['__period'].eq(period)].copy()
    if frame.empty or frame.duplicated('__branch').any() or frame[keys+schema['features']].isna().any().any():raise ValueError('Empty, duplicate or incomplete branch-month input')
    index=read(run/'selected_model_index.json')
    if set(frame['__group'])-{x['group'] for x in index}:raise ValueError('New group needs a reference model')
    for f in schema['fields']:
        if f['dtype']=='numeric':
            frame[f['name']]=pd.to_numeric(frame[f['name']],errors='raise')
            import numpy as np
            if not np.isfinite(frame[f['name']]).all():raise ValueError('Nonfinite numeric input')
    rows=[]
    for item in index:
        block=frame[frame['__group'].eq(item['group'])].sort_values('__branch')
        if block.empty:continue
        bundle=joblib.load(run/item['path']);labels=predict(bundle,block)
        rows.extend({**r,'method':winner['method'],'mode':'frozen','cluster':int(label),'cluster_id':f'{item["group"]}__C{int(label)+1}',
                     'model_id':item['model_id'],'selection_run_id':manifest['run_id'],'case_id':winner['case_id']} for r,label in zip(block[keys].to_dict('records'),labels))
    assignment_coverage(rows,frame[keys].to_dict('records'),contract['required_min_cluster_size'],winner['k_by_group'])
    return rows


def run(source,config_path,output=None):
    source=Path(source).resolve();manifest=read(source/'manifest.json');config=read(config_path)
    contract=resolve_coverage(config,config_path)
    if not manifest['checks'] or not all(manifest['checks'].values()) or not manifest.get('is_parameter_grid'):raise ValueError('A verified parameter grid is required')
    check_hashes(source,manifest)
    plan=read(source/'grid_plan.json');validate_config(config,manifest['reference_period'],manifest['monitoring_periods'])
    code_paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/'otomatik_kumeleme.py',Path(__file__).parents[1]/'kumeleme/models.py']
    signature=hashlib.sha256(json.dumps({'config':config,'coverage':contract},sort_keys=True).encode()+sha(source/'manifest.json').encode()+''.join(sha(p) for p in sorted(code_paths)).encode()).hexdigest()[:12]
    out=Path(output) if output else ROOT/'outputs/otomatik_kumeleme'/f'{manifest["as_of"]}_{signature}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);save(out/'config.json',config);save(out/'grid_plan.json',plan);save(out/'dea_coverage_contract.json',contract)
    candidates=[];evidence={}
    for case in plan['cases']:
        folder=source/case['relative_path'];evidence[case['case_id']]={name:read(folder/(name+'.json')) for name in ['selected','monthly','sensitivity']}
        settings=read(folder/'config.json');repeat_counts={'seed':settings['seed_repeats'],'subsample':settings['subsample_repeats']}
        evidence[case['case_id']]['repeat_counts']=repeat_counts
        e=evidence[case['case_id']]
        for method in manifest['methods']:
            candidates.append(score_candidate(case,e['selected'],e['monthly'],e['sensitivity'],method,config,repeat_counts))
    decision=choose(candidates,config)
    decision.update(validation_periods=config['validation_periods'],reference_period=manifest['reference_period'],scope='one_method_and_scenario_for_all_indicators_and_groups',
                    weights=config['weights'],thresholds=config['thresholds'],score_is_accuracy=False,
                    coverage_required=True,required_min_cluster_size=contract['required_min_cluster_size'])
    # Freeze the decision before any evaluation-period calculation.
    save(out/'decision.json',decision);save(out/'candidate_scores.json',candidates)
    save(out/'source_evidence.json',evidence)
    table(out/'candidate_scores.csv',[{k:c[k] for k in ['candidate_id','method','feature_ratio','feature_count','bank_clusters','applicable','qualified','score']}|{k:c['metrics'].get(k) for k in config['weights']}|{'worst_group_silhouette':c['metrics'].get('worst_group_silhouette'),'failed_gates':'; '.join(c['reasons'])} for c in candidates])
    geographic=[{'candidate_id':c['candidate_id'],'case_id':c['case_id'],'method':c['method'],**g} for c in candidates for g in c['group_quality']]
    save(out/'geographic_quality.json',geographic);table(out/'geographic_quality.csv',geographic)
    gates=[{'candidate_id':c['candidate_id'],**g} for c in candidates for g in c['gates']];save(out/'gates.json',gates);table(out/'gates.csv',gates)
    winner=decision['selected'];evaluation={};index=[];assignments=[];coverage=[]
    if winner:
        case=next(c for c in plan['cases'] if c['case_id']==winner['case_id']);folder=source/case['relative_path']
        e=evidence[winner['case_id']]
        evaluation=monthly_metrics(e['monthly'],winner['method'],config['evaluation_periods'],case['group_counts']) if config['evaluation_periods'] else {}
        (out/'models').mkdir();shutil.copy2(folder/'input/selected_schema.json',out/'selected_schema.json')
        panel=pd.read_csv(folder/'input/panel.csv',dtype={'__branch':str,'__period':str,'__group':str})
        schema=read(out/'selected_schema.json');panel=panel[panel['__period'].isin([manifest['reference_period']]+manifest['monitoring_periods'])]
        panel[['__branch','__period','__group']+schema['features']].to_csv(out/'prepared_panel.csv',index=False,encoding='utf-8-sig')
        for item in e['selected']:
            if item['method']!=winner['method']:continue
            target='models/'+item['model_file'];shutil.copy2(folder/target,out/target)
            index.append({'group':item['group'],'method':item['method'],'model_id':item['model_id'],'k':item['k'],'path':target})
        assignments=[r for r in read(folder/'assignments.json') if r['method']==winner['method'] and r['mode']=='frozen']
        # This is an operational coverage guard after selection, not a re-ranking on evaluation metrics.
        # A failing latest month stops activation; it never silently discards its small clusters.
        coverage=assignment_coverage(assignments,panel[['__branch','__period','__group']].to_dict('records'),contract['required_min_cluster_size'],winner['k_by_group'])
    save(out/'full_coverage.json',coverage)
    save(out/'evaluation.json',{'periods':config['evaluation_periods'],'role':'reused_historical_evaluation_not_untouched_test','metrics':evaluation,'changes_decision':False})
    save(out/'selected_model_index.json',index);save(out/'primary_assignments.json',assignments)
    assignment_columns=['__branch','__period','__group','method','mode','cluster','cluster_id','model_id']
    pd.DataFrame(assignments,columns=assignment_columns).to_csv(out/'primary_assignments.csv',index=False,encoding='utf-8-sig')
    latest=[r for r in assignments if r['__period']==manifest['as_of']]
    pd.DataFrame(latest,columns=assignment_columns).to_csv(out/'primary_latest.csv',index=False,encoding='utf-8-sig')
    code=out/'code';(code/'kumeleme_secimi').mkdir(parents=True);(code/'kumeleme').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/'kumeleme_secimi'/p.name)
    for name in ['__init__.py','models.py']:shutil.copy2(Path(__file__).parents[1]/'kumeleme'/name,code/'kumeleme'/name)
    shutil.copy2(Path(__file__).parents[1]/'otomatik_kumeleme.py',code/'otomatik_kumeleme.py')
    shutil.copy2(Path(__file__).parents[1]/'requirements_kumeleme.txt',out/'requirements.txt')
    description=f"{METHODS[winner['method']]} · {winner['case_id']} · puan {winner['score']:.4f}" if winner else 'Uygun model yok'
    (out/'SONUCLAR.md').write_text(f"# Otomatik ana kümeleme modeli seçimi\n\n{description}\n\nSeçim: {', '.join(config['validation_periods'])}. Referans: {manifest['reference_period']}. Ayrı geçmiş dönem kontrolü: {', '.join(config['evaluation_periods'])}. Bu aylar daha önce incelendi; dokunulmamış test değildir.\n\n{sum(c['qualified'] for c in candidates)} uygun aday / {sum(c['applicable'] for c in candidates)} uygulanabilir aday. Tek yöntem ve senaryo bütün göstergeler ve coğrafi gruplar için ortaktır; bölgesel modeller ayrı fit edilmiştir.\n\nAralık sonuçları: {json.dumps(evaluation,ensure_ascii=False)}\n\nBu puan doğruluk yüzdesi değildir. Sentetik pilot sonucu gerçek banka başarısını göstermez. DEA ve hedef dağıtımı yapılmadı.\n",encoding='utf-8')
    (out/'METODOLOJI.md').write_text('''# Ana kümeleme modeli seçimi

Aday, yöntem + orijinal alan seçim oranı + banka toplam küme sayısı birleşimidir. URF korunması zorunlu değildir. Tek ortak aday seçilir; coğrafi grupların kendi referans modelleri korunur.

Referans ay modelleri yeniden eğitilmez. Seçimde yalnızca config.json içindeki validation_periods ve referans ay tohum/alt örneklem tekrarları kullanılır. Evaluation ayları karar dosyası yazıldıktan sonra raporlanır; seçime, elemelere veya puana girmez. Bu çalışmanın ayları daha önce incelendiğinden evaluation dokunulmamış test değildir. Değişken seçimi ve kaynak veri geçmişi önceki çalışmaya aittir; yeni uçtan uca test iddiası yoktur.

Önce uygulanabilirlik, tam grup/ay/tekrar kaydı, yakınsama, referans ve sabit doğrulama atamalarında küme boyutu koşulları aranır. Sonra banka ortalama silhouette, aylık yeniden kümeleme ARI, referans alt örneklem ARI, referans tohum ARI ve en zayıf grubun ortalama yeniden kümeleme ARI eşikleri uygulanır. Eşikler pilot politika tercihleridir; evrensel DEA yeterlilik ölçütü değildir.

Her ay eşit ağırlıklıdır. Coğrafi gruplar referans şube sayılarına göre ağırlıklanır. Ortak silhouette bütün adaylarda aynı tam 22 alanlı Gower geometrisinde ölçülmüştür; Gower yöntemini avantajlı kılabilir. İş profilleri ve sonraki gerçek veri dönemleriyle inceleme gereklidir.

Puan = 0.30 × silhouette + 0.30 × yeniden kümeleme ARI + 0.25 × alt örneklem ARI + 0.15 × tohum ARI. Her bileşen [0,1] aralığına kırpılır. Yapılandırılabilir ağırlıklar ham ölçeklerde bir fayda puanıdır; eşit yüzdeli katkı veya doğruluk yüzdesi değildir. Sabit atama ARI puana katılmaz. Eğitim süresi seçim ölçütü değildir; mevcut adaylar zaten küçük ve önceden eğitilmiştir.

Koşulları geçen en yüksek puanlı aday seçilir. Eşitlikte daha az küme, daha az alan, sonra aday kodu kullanılır. Mevcut aday hâlâ uygunsa minimum_gain kadar artış olmadan değiştirilmez. Hiçbir aday uygun değilse selected=null olur ve ana modelle atama engellenir. Evaluation'daki bozulma seçimi geriye dönük değiştirmez; ayrı inceleme sinyali olarak gösterilir.

apply komutu hazırlanmış şube-ay panelini alır. Seçilmiş tek yöntemin bölgesel native modellerini kullanır; yeniden fit etmez. Ham müşteri verisi önce dinamik veri sözlüğündeki aynı hazırlama adımlarından geçirilmelidir. Yeni aylar için önce karşılaştırma paneli/grid güncellenir, sonra seçim yapılandırmasındaki dönemler ilerletilir; değerlendirme ayı seçime alınırsa yeni bir ileri dönem ayrıca ayrılır.
''',encoding='utf-8')
    with (out/'METODOLOJI.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## Bütün şubelerin kapsanması\n\nDEA alanları ve girdi kontrol senaryoları seçimden önce okunur. Gerekli küme alt sınırı bu senaryoların max(m×s,3(m+s)) değerlerinin en büyüğü ve tanımlı pilot alt sınırın maksimumudur: bu koşuda {contract["required_min_cluster_size"]}. Alanlar değişirse seçim yeniden çalıştırılır. Referans/doğrulama boyut koşullarını geçmeyen aday elenir. Seçim dondurulduktan sonra bütün kayıtlı ayların gerçek atamaları kaynak şube evreniyle birebir karşılaştırılır. Sonraki ayda kapsama bozulursa başka aday o aya bakılarak seçilmez; aktivasyon durur. Yeni aya apply işlemi de aynı kontrolü yapar. Bu ön kontrol istatistiksel yeterlilik garantisi değildir.\n')
        f.write(f'\n## Coğrafi grupların ayrı kalite koşulu\n\nHer coğrafi grubun doğrulama dönemindeki sabit atama silhouette ortalaması ayrı hesaplanır. Bütün gruplar aynı {config["thresholds"]["worst_group_silhouette"]:.2f} alt sınırını geçmelidir. En düşük grup ortalaması bu sınırın altındaysa banka ortalaması veya birleşik puan yüksek olsa da aday elenir. Eksik grup/ay ölçüsü de geçerli sayılmaz. Eşik yapılandırılabilir bir pilot politikasıdır; evrensel veya bu veriden optimize edilmiş eşik değildir. İki grup için ayrı algoritma seçilmez. Kontrol yalnız config.validation_periods aylarını kullanır; evaluation ayları bu koşula ve aday sırasına girmez. Uygun aday yoksa seçilen model null kalır; önceki model yeni politika kapsamında otomatik geri alınmaz. Önceki DEA sonuçları yalnız tarihsel olarak korunur.\n')
    result={'run_id':out.name,'source_grid_id':manifest['run_id'],'source_grid_path':str(source),'source_grid_manifest_sha256':sha(source/'manifest.json'),
            'as_of':manifest['as_of'],'reference_period':manifest['reference_period'],'method_switch_automatic':True,'common_method':True,'dea_executed':False,'target_allocation_executed':False,
            'selected_method':winner['method'] if winner else None,'selected_case_id':winner['case_id'] if winner else None,'latest_branch_count':len(latest),'saved_models':len(index),
            'common_evaluation_feature_count':plan['parent_feature_count'],'full_coverage_required':True,
            'source_branch_count':sum(plan['cases'][0]['group_counts'].values()),'geographic_quality_required':True,
            'required_min_cluster_size':contract['required_min_cluster_size'],'full_coverage_verified':bool(winner and coverage),'checks':{}}
    result['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'manifest.json',result)
    from .verify import verify
    result['checks']=verify(out);save(out/'manifest.json',result)
    report={'manifest':result,'config':config,'decision':decision,'candidates':candidates,'evaluation':read(out/'evaluation.json'),'method_names':METHODS,
            'dea_coverage_contract':contract,'full_coverage':coverage,'geographic_quality':geographic}
    save(out/'report.json',report)
    archive=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Corrupt package')
    save(out.parent/'latest.json',{'path':str(out.resolve()),'run_id':out.name})
    print(json.dumps({'path':str(out),'decision':description,'qualified':sum(c['qualified'] for c in candidates),'checks':result['checks']},ensure_ascii=False))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('select');p.add_argument('--source');p.add_argument('--config',default=str(ROOT/'python_kodlari/ayarlar/otomatik_kumeleme.json'));p.add_argument('--output')
    p=subs.add_parser('verify');p.add_argument('--run',required=True)
    p=subs.add_parser('apply');p.add_argument('--run',required=True);p.add_argument('--panel',required=True);p.add_argument('--output',required=True);p.add_argument('--period')
    args=parser.parse_args()
    if args.command=='select':run(args.source or read(ROOT/'outputs/kumeleme_karsilastirma/latest.json')['path'],args.config,args.output)
    elif args.command=='verify':
        from .verify import verify
        print(json.dumps(verify(args.run),ensure_ascii=False))
    else:
        rows=infer(args.run,args.panel,args.period);out=Path(args.output)
        if out.exists():raise FileExistsError(out)
        out.parent.mkdir(parents=True,exist_ok=True);table(out,rows);print(f'{len(rows)} primary assignments: {out}')
