"""Build the dashboard's static, verified snapshot from the existing ML outputs."""
from pathlib import Path
import collections
import csv
import json
import math
import shutil
import zipfile
import hashlib
import re

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'outputs/lightgbm_ilk_model'
SITE=ROOT/'model_dashboard'
DIST=SITE/'dist'
DIST.mkdir(parents=True,exist_ok=True)
DOWNLOADS=DIST/'downloads';DOWNLOADS.mkdir(exist_ok=True)

report=json.loads((MODEL/'sonuclar/rapor_verisi.json').read_text())
report['manifest']=json.loads((MODEL/'manifest.json').read_text())
report['data_manifest']=json.loads((ROOT/'outputs/banka_test_verisi_v3/manifest.json').read_text())
report['data_manifest_v4']=json.loads((ROOT/'outputs/banka_test_verisi_v4/manifest.json').read_text())
report['model_metadata']=[json.loads(p.read_text()) for p in sorted((MODEL/'modeller').glob('*__metadata.json'))]
COMPARISON=ROOT/'outputs/model_karsilastirma'
if (COMPARISON/'sonuclar/rapor_verisi.json').exists():
    report['comparison']=json.loads((COMPARISON/'sonuclar/rapor_verisi.json').read_text())
    report['comparison']['manifest']=json.loads((COMPARISON/'manifest.json').read_text())
COMPARISON_V4=ROOT/'outputs/model_karsilastirma_v4'
if (COMPARISON_V4/'sonuclar/rapor_verisi.json').exists():
    report['comparison_v4']=json.loads((COMPARISON_V4/'sonuclar/rapor_verisi.json').read_text())
    report['comparison_v4']['manifest']=json.loads((COMPARISON_V4/'manifest.json').read_text())
    with (ROOT/'outputs/banka_test_verisi_v4/ocak2027_banka_hedefleri.csv').open(encoding='utf-8-sig') as f:
        report['budgets_v4']=list(csv.DictReader(f))
    for b in report['budgets_v4']:
        for key in ['referans_aylik_ortalama_tl','artis_orani','hedef_aylik_ortalama_tl','artis_tutari_tl']:b[key]=float(b[key])
        b['gosterge_adi']={'TL_VADESIZ_ORT':'TL vadesiz mevduat ortalaması','TL_VADELI_ORT':'TL vadeli mevduat ortalaması','BIREYSEL_KREDI_ORT':'Bireysel kredi ortalaması'}[b['gosterge_kodu']]
FLAML=ROOT/'outputs/flaml_ortak_v4' if (ROOT/'outputs/flaml_ortak_v4/manifest.json').exists() else ROOT/'outputs/flaml_v4'
if (FLAML/'sonuclar/rapor_verisi.json').exists():
    report['automl']=json.loads((FLAML/'sonuclar/rapor_verisi.json').read_text())
    report['automl']['manifest']=json.loads((FLAML/'manifest.json').read_text())
    assert report['automl']['manifest']['checks'].get('independent_30_bundles_reloaded')
    assert 'search_updates' in report['automl'],'Run flaml_raporla.py before dashboard snapshot.'
SELECTION_ROOT=ROOT/'outputs/dinamik_feature_selection'
SELECTION=None
if (SELECTION_ROOT/'latest.json').exists():
    SELECTION=Path(json.loads((SELECTION_ROOT/'latest.json').read_text())['path']).resolve()
    assert SELECTION.is_relative_to(SELECTION_ROOT.resolve())
    report['feature_selection']=json.loads((SELECTION/'report.json').read_text())
    sm=report['feature_selection']['manifest']
    assert sm['checks'] and all(sm['checks'].values())
    for name,digest in sm['artifact_hashes'].items():
        assert hashlib.sha256((SELECTION/name).read_bytes()).hexdigest()==digest,name
    assert sm['selected_count']==len(report['feature_selection']['selected_schema']['features'])
CLUSTER_ROOT=ROOT/'outputs/kumeleme_karsilastirma'
CLUSTER=None
if (CLUSTER_ROOT/'latest.json').exists():
    CLUSTER=Path(json.loads((CLUSTER_ROOT/'latest.json').read_text())['path']).resolve()
    assert CLUSTER.is_relative_to(CLUSTER_ROOT.resolve())
    report['clustering']=json.loads((CLUSTER/'report.json').read_text())
    cm=report['clustering']['manifest']
    assert cm['checks'] and all(cm['checks'].values())
    for name,digest in cm['artifact_hashes'].items():
        assert hashlib.sha256((CLUSTER/name).read_bytes()).hexdigest()==digest,name
    report['clustering']['matches_current_selection']=bool(SELECTION and cm['feature_selection_run_id']==report['feature_selection']['manifest']['run_id'])
    if SELECTION:report['feature_selection']['clustering_available']=report['clustering']['matches_current_selection']
AUTO_ROOT=ROOT/'outputs/otomatik_kumeleme'
AUTO=None
if (AUTO_ROOT/'latest.json').exists():
    AUTO=Path(json.loads((AUTO_ROOT/'latest.json').read_text())['path']).resolve()
    assert AUTO.is_relative_to(AUTO_ROOT.resolve())
    report['auto_cluster_selection']=json.loads((AUTO/'report.json').read_text())
    am=json.loads((AUTO/'manifest.json').read_text())
    assert am==report['auto_cluster_selection']['manifest']
    assert am['checks'] and all(am['checks'].values())
    for name,digest in am['artifact_hashes'].items():
        assert hashlib.sha256((AUTO/name).read_bytes()).hexdigest()==digest,name
    assert json.loads((AUTO/'decision.json').read_text())==report['auto_cluster_selection']['decision']
    assert json.loads((AUTO/'candidate_scores.json').read_text())==report['auto_cluster_selection']['candidates']
    assert json.loads((AUTO/'evaluation.json').read_text())==report['auto_cluster_selection']['evaluation']
    if am.get('geographic_quality_required'):
        assert json.loads((AUTO/'geographic_quality.json').read_text())==report['auto_cluster_selection']['geographic_quality']
    if am.get('full_coverage_required'):
        assert json.loads((AUTO/'dea_coverage_contract.json').read_text())==report['auto_cluster_selection']['dea_coverage_contract']
        assert json.loads((AUTO/'full_coverage.json').read_text())==report['auto_cluster_selection']['full_coverage']
        assert am['full_coverage_verified'] or report['auto_cluster_selection']['decision']['selected'] is None
    matches=bool(CLUSTER and am['source_grid_id']==report['clustering']['manifest']['run_id'] and am['source_grid_manifest_sha256']==hashlib.sha256((CLUSTER/'manifest.json').read_bytes()).hexdigest() and report['clustering']['matches_current_selection'])
    report['auto_cluster_selection']['matches_current_grid']=matches
    if matches:report['clustering']['automatic_selection']=report['auto_cluster_selection']
    report['clustering']['current_policy_approved']=bool(matches and report['auto_cluster_selection']['decision']['selected'])
DEA_ROOT=ROOT/'outputs/dea_verimlilik'
DEA=None
if (DEA_ROOT/'latest.json').exists():
    DEA=Path(json.loads((DEA_ROOT/'latest.json').read_text())['path']).resolve()
    assert DEA.is_relative_to(DEA_ROOT.resolve())
    report['dea']=json.loads((DEA/'report.json').read_text())
    dm=json.loads((DEA/'manifest.json').read_text())
    assert dm==report['dea']['manifest'] and dm['checks'] and all(dm['checks'].values())
    for name,digest in dm['artifact_hashes'].items():
        assert hashlib.sha256((DEA/name).read_bytes()).hexdigest()==digest,name
    for name in ['scores','potentials','clusters','peers','summary','config']:
        assert json.loads((DEA/(name+'.json')).read_text())==report['dea'][name]
    report['dea']['matches_current_selection']=bool(AUTO and report['auto_cluster_selection']['matches_current_grid'] and dm['source_selection_run_id']==report['auto_cluster_selection']['manifest']['run_id'] and dm['source_selection_manifest_sha256']==hashlib.sha256((AUTO/'manifest.json').read_bytes()).hexdigest())
    report['dea']['matches_current_data']=hashlib.sha256(Path(dm['source_data_path']).read_bytes()).hexdigest()==dm['source_data_sha256']
    report['dea']['matches_latest_period']=bool(AUTO and dm['as_of']==report['auto_cluster_selection']['manifest']['as_of'])
    report['dea']['active']=report['dea']['matches_current_selection'] and report['dea']['matches_current_data'] and report['dea']['matches_latest_period']
    with (DEA/'dea_input.csv').open(encoding='utf-8-sig') as f:
        report['dea']['input_rows']=[{**{k:r[k] for k in ['__branch','__group','cluster_id']},**{x['name']:float(r[x['name']]) for x in report['dea']['config']['inputs']}} for r in csv.DictReader(f)]
    if CLUSTER:report['clustering']['dea_available']=report['dea']['active']
    if AUTO:report['auto_cluster_selection']['dea_available']=report['dea']['active']
DEA_AUDIT_ROOT=ROOT/'outputs/dea_girdi_kontrol'
DEA_AUDIT=None
if (DEA_AUDIT_ROOT/'latest.json').exists():
    DEA_AUDIT=Path(json.loads((DEA_AUDIT_ROOT/'latest.json').read_text())['path']).resolve()
    assert DEA_AUDIT.is_relative_to(DEA_AUDIT_ROOT.resolve())
    audit=json.loads((DEA_AUDIT/'report.json').read_text());aq=json.loads((DEA_AUDIT/'manifest.json').read_text())
    assert audit['manifest']==aq and aq['checks'] and all(aq['checks'].values())
    for name,digest in aq['artifact_hashes'].items():
        assert hashlib.sha256((DEA_AUDIT/name).read_bytes()).hexdigest()==digest,name
    for name in ['policy','recommendation','quality','coverage','correlations','comparisons','temporal_summary','temporal_within_cluster_ranks','stress_summary','staff_profile']:
        assert json.loads((DEA_AUDIT/(name+'.json')).read_text())==audit[name]
    audit['active']=bool(DEA and report['dea']['active'] and aq['source_dea_run_id']==report['dea']['manifest']['run_id'] and aq['source_dea_manifest_sha256']==hashlib.sha256((DEA/'manifest.json').read_bytes()).hexdigest() and aq['source_data_sha256']==report['dea']['manifest']['source_data_sha256'])
    # Derive review rows from verified raw experiment records; keep them out of the main DEA scores.
    influence=collections.defaultdict(list)
    for row in json.loads((DEA_AUDIT/'peer_removal.json').read_text()):
        influence[(row['scenario'],row['cluster_id'],row['removed_peer'])].append(row)
    audit['influential_peers']=[]
    for (scenario,cluster,peer),rows in influence.items():
        worst=max(rows,key=lambda r:r['score_change'])
        if worst['score_change']>audit['policy']['peer_removal_max_review']:
            audit['influential_peers'].append({'scenario':scenario,'cluster_id':cluster,'removed_peer':peer,'affected_branches':sum(r['score_change']>audit['policy']['material_score_change'] for r in rows),
                'worst_affected_branch':worst['branch'],'max_score_change':worst['score_change'],'before':worst['baseline'],'after':worst['stressed']})
    audit['influential_peers'].sort(key=lambda r:(r['scenario'],-r['max_score_change']))
    if DEA:report['dea']['input_audit']=audit
with (ROOT/'outputs/banka_test_verisi_v3/ocak2027_banka_hedefleri.csv').open(encoding='utf-8-sig') as f:
    budgets=list(csv.DictReader(f))
for b in budgets:
    for k in ['referans_aylik_ortalama_tl','artis_orani','hedef_aylik_ortalama_tl','artis_tutari_tl']:b[k]=float(b[k])
report['budgets']=budgets

monthly=collections.defaultdict(list)
with (MODEL/'sonuclar/test_ve_validasyon_tahminleri.csv').open(encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        monthly[(r['gosterge'],r['model'],r['split'],r['hedef_donemi'],r['sube_grubu'])].append((float(r['gercek_tl']),float(r['tahmin_tl'])))
combined=collections.defaultdict(list)
for (target,model,split,period,group),values in list(monthly.items()):combined[(target,model,split,period,'ALL')]+=values
monthly.update(combined)
report['monthly_metrics']=[]
for (target,model,split,period,group),pairs in sorted(monthly.items()):
    actual=sum(y for y,p in pairs);predicted=sum(p for y,p in pairs)
    report['monthly_metrics'].append({'gosterge':target,'model':model,'split':split,'hedef_donemi':period,
      'sube_grubu':group,'n':len(pairs),'gercek_toplam_tl':actual,'tahmin_toplam_tl':predicted,
      'wape':sum(abs(y-p) for y,p in pairs)/actual,'mae_tl':sum(abs(y-p) for y,p in pairs)/len(pairs)})
assert len(report['monthly_metrics'])==216
report['downloads']=[]
def add(source,filename,title,detail,kind):
    target=DOWNLOADS/filename;shutil.copy2(source,target)
    report['downloads'].append({'path':'downloads/'+filename,'name':filename,'title':title,
      'description':detail,'kind':kind,'bytes':target.stat().st_size})

add(MODEL/'METODOLOJI.md','metodoloji.md','İlk V3 LightGBM metodolojisi','Hedef tanımı, zaman ayrımı, LightGBM ve önem hesapları.','Belge')
add(MODEL/'LightGBM_sonuc_raporu.xlsx','LightGBM_sonuc_raporu.xlsx','Excel sonuç raporu','Metrikler, değişken önemleri ve kararlılık tabloları.','Excel')
add(ROOT/'outputs/lightgbm_ilk_model_paketi.zip','lightgbm_model_paketi.zip','Tam model paketi','Eğitilmiş modeller, eğitim setleri, kodlar ve sonuçlar.','ZIP')
if 'comparison' in report:
    add(COMPARISON/'METODOLOJI.md','karsilastirma_metodoloji.md','Üç algoritmanın metodolojisi',
        'Ortak veri, özellikler, ayar araması, zaman ölçümü ve karşılaştırma sınırları.','Belge')
    add(ROOT/'outputs/model_karsilastirma_paketi.zip','model_karsilastirma_paketi.zip','Üç algoritma model paketi',
        'LightGBM, Random Forest, CatBoost: 36 model, 72 dosya, tahminler ve kodlar.','ZIP')
    for name,title,detail in [('toplam_metrikler','Üç algoritma · toplam metrikler','Yapısal ve geçmiş bakiyeli modellerin doğrulama ve test hataları.'),
                              ('model_metrikleri','Üç algoritma · grup metrikleri','Şube grubu bazında hata, fit ve arama süreleri.'),
                              ('degisken_onemi','Üç algoritma · değişken önemi','Ortak sekiz tekrarlı permütasyon ve ayrı native önem ölçüleri.')]:
        add(COMPARISON/'sonuclar'/f'{name}.csv',f'karsilastirma_{name}.csv',title,detail,'CSV')
if 'comparison_v4' in report:
    add(COMPARISON_V4/'METODOLOJI.md','v4_karsilastirma_metodoloji.md','V4 karşılaştırma metodolojisi',
        'Dinamik aidiyet, 25/34/49 özellik, ay sonu stok referansı ve 54 model.','Belge')
    add(COMPARISON_V4/'SONUCLAR.md','v4_sonuclar.md','V4 ölçülen sonuçlar','Üç görünüm, doğrulama seçimi ve ayrı test sonuçları.','Belge')
    add(ROOT/'outputs/model_karsilastirma_v4_paketi.zip','model_karsilastirma_v4_paketi.zip','V4 tam model paketi',
        '54 modelin 108 kaydı, veri ayrımları, kodlar, tahminler ve metrikler.','ZIP')
    for name,title in [('toplam_metrikler','V4 toplam metrikler'),('model_metrikleri','V4 grup ve süre metrikleri'),
                        ('degisken_onemi','V4 değişken önemleri'),('v3_v4_test_karsilastirma','V3 ve V4 test tabloları')]:
        add(COMPARISON_V4/'sonuclar'/f'{name}.csv',f'v4_{name}.csv',title,'V4 koşusundan doğrulanmış kayıtlar.','CSV')
    for name,label,n in [('train','V4 eğitim seti',3400),('validation','V4 doğrulama seti',600),('test','V4 test seti',600)]:
        add(COMPARISON_V4/'veri'/f'{name}.csv',f'v4_{name}.csv',label,f'{n} şube-ay. Güncel müşteri kitlesi ve gün ağırlıklı bakiye.','CSV')
    add(COMPARISON_V4/'veri/ozellik_sozlugu.csv','v4_ozellik_sozlugu.csv','V4 özellik sözlüğü','Üç görünümde alan tanımları ve kullanım işaretleri.','CSV')
if 'automl' in report:
    add(FLAML/'METODOLOJI.md','flaml_metodoloji.md','FLAML protokolü','Süre sınırı, doğrulama seçimi, native model ve referans kuralları.','Belge')
    add(FLAML/'SONUCLAR.md','flaml_sonuclar.md','FLAML sonuç özeti','AutoML adayı, seçilen sistem ve basit referansların karşılaştırması.','Belge')
    flaml_package='flaml_ortak_v4_paketi.zip' if FLAML.name=='flaml_ortak_v4' else 'flaml_v4_paketi.zip'
    add(ROOT/'outputs'/flaml_package,'flaml_v4_paketi.zip','FLAML model ve kod paketi','30 native paket, arama kayıtları, veri ayrımları ve Python kodları.','ZIP')
    for filename,title in [('toplam_metrikler','FLAML banka metrikleri'),('model_metrikleri','FLAML grup metrikleri'),('arama_gecmisi','FLAML bütün denemeler')]:
        add(FLAML/'sonuclar'/f'{filename}.csv',f'flaml_{filename}.csv',title,'Doğrulanmış FLAML V4 koşusunun gerçek çıktıları.','CSV')
if SELECTION:
    for source,filename,title,kind in [
        ('config.json','selection_dictionary.json','Dinamik alan sözlüğü','JSON'),
        ('selected_schema.json','selection_schema.json','Seçilen ve dondurulmuş şema','JSON'),
        ('decisions.csv','selection_decisions.csv','Değişken seçimi kararları','CSV'),
        ('cluster_input_latest.csv','selection_cluster_input_latest.csv','Son ay kümeleme girdisi','CSV'),
        ('importance_repeats.csv','selection_importance_repeats.csv','Dönemsel önem tekrarları','CSV'),
        ('drift.json','selection_drift.json','Şema ve dağılım kontrolleri','JSON'),
        ('SONUCLAR.md','selection_results.md','Dinamik seçim sonuçları','Belge'),
        ('METODOLOJI.md','selection_methodology.md','Dinamik seçim metodolojisi','Belge')]:
        add(SELECTION/source,filename,title,'Sözlükten çalışan CatBoost destekli ortak kümeleme değişken seçimi.',kind)
    add(SELECTION.parent/(SELECTION.name+'.zip'),'selection_package.zip','Dinamik değişken seçimi paketi',
        'Alan sözlüğü, modeller, kararlı seçim kararları, Python kodları ve kümeleme girdileri.','ZIP')
if CLUSTER:
    for source,filename,title,kind in [
        ('SONUCLAR.md','clustering_results.md','Kümeleme karşılaştırma sonuçları','Belge'),
        ('METODOLOJI.md','clustering_methodology.md','URF ve alternatifler metodolojisi','Belge'),
        ('assignments.csv','clustering_assignments.csv','Bütün şube küme atamaları','CSV'),
        ('profiles.csv','clustering_profiles.csv','Küme müşteri profilleri','CSV'),
        ('candidates.csv','clustering_candidates.csv','Küme sayısı adayları','CSV'),
        ('monthly.csv','clustering_monthly.csv','Aylık küme kararlılığı','CSV'),
        ('sensitivity.csv','clustering_sensitivity.csv','Tohum ve alt örneklem karşılaştırması','CSV'),
        ('drift.csv','clustering_drift.csv','Referans aralığı dışındaki alanlar','CSV'),
        ('selected.json','clustering_selected.json','Referans ayda seçilen kümeler','JSON')]:
        add(CLUSTER/source,filename,title,'Kaynak gridin önceki URF referans çıktıları; güncel ana seçim ayrı otomatik seçim dosyalarındadır.','CSV' if kind=='CSV' else kind)
    add(CLUSTER.parent/(CLUSTER.name+'.zip'),'clustering_package.zip','Kümeleme tam çalışma paketi',
        'Native modeller, şube atamaları, profiller, karşılaştırmalar, Python kodları ve yöntem ayarları.','ZIP')
    if (CLUSTER/'grid_scores.csv').exists():
        add(CLUSTER/'grid_scores.csv','clustering_grid_scores.csv','Alan oranı ve toplam küme senaryoları',
            '%25/%50 orijinal alan seçimi, kodlama sonrası %25 mtry, banka geneli 5/7 küme karşılaştırması.','CSV')
if AUTO:
    for source,filename,title,kind in [
        ('decision.json','auto_cluster_decision.json','Ana kümeleme modeli kararı','JSON'),
        ('candidate_scores.csv','auto_cluster_scores.csv','Ana model aday puanları','CSV'),
        ('gates.csv','auto_cluster_gates.csv','Ana model kalite ve boyut koşulları','CSV'),
        ('primary_latest.csv','auto_cluster_latest.csv','Seçilen ana yöntemin son ay atamaları','CSV'),
        ('evaluation.json','auto_cluster_evaluation.json','Seçimden ayrı geçmiş dönem kontrolü','JSON'),
        ('METODOLOJI.md','auto_cluster_methodology.md','Otomatik ana model seçim metodolojisi','Belge'),
        ('SONUCLAR.md','auto_cluster_results.md','Otomatik ana model sonuçları','Belge')]:
        add(AUTO/source,filename,('Onaylı model yok; atama üretilmedi' if source=='primary_latest.csv' and not report['auto_cluster_selection']['decision']['selected'] else title),'Tek ortak yöntem ve senaryo; koşulları geçmeyen aday ana model olamaz.',kind)
    add(AUTO.parent/(AUTO.name+'.zip'),'auto_cluster_package.zip','Otomatik kümeleme seçim ve model paketi',
        f"Seçim politikası, aday puanları, kalite koşulları, {am['saved_models']} onaylı native model ve atama kodları.",'ZIP')
    if (AUTO/'dea_coverage_contract.json').exists():
        add(AUTO/'dea_coverage_contract.json','auto_cluster_coverage_contract.json','DEA tam kapsam sözleşmesi','Ana ve ek girdi senaryolarından hesaplanan ortak küme alt sınırı.','JSON')
        add(AUTO/'full_coverage.json','auto_cluster_full_coverage.json','Bütün ayların şube kapsam kontrolü','Kaynak şube evreni, atanan şubeler ve gerçek küme büyüklükleri.','JSON')
    if (AUTO/'geographic_quality.csv').exists():
        add(AUTO/'geographic_quality.csv','auto_cluster_geographic_quality.csv','Coğrafi grup ayrışma koşulları','Her aday için grup puanı, pilot eşik ve geçme/kalma kararı.','CSV')
if DEA:
    for source,filename,title,kind in [
        ('scores.csv','dea_scores.csv','DEA şube verimlilik skorları','CSV'),
        ('potentials.csv','dea_potentials.csv','DEA ürün bazında sınır projeksiyonları','CSV'),
        ('peers.csv','dea_peers.csv','DEA emsal şubeleri ve katsayıları','CSV'),
        ('clusters.csv','dea_clusters.csv','DEA küme büyüklük ve skor özeti','CSV'),
        ('dea_input.csv','dea_input.csv','DEA girdi ve çıktı kesiti','CSV'),
        ('config.json','dea_config.json','DEA alan ve model sözlüğü','JSON'),
        ('METODOLOJI.md','dea_methodology.md','DEA metodolojisi ve kaynakları','Belge'),
        ('SONUCLAR.md','dea_results.md','DEA sonuç özeti','Belge')]:
        add(DEA/source,filename,title,('Tarihsel koşu; güncel politikayla onaylı değildir. ' if not report['dea']['active'] else '')+'Küme içi çıktı odaklı CRS-DEA; VRS duyarlılığı. Projeksiyonlar banka hedefi değildir.',kind)
    add(DEA.parent/(DEA.name+'.zip'),'dea_package.zip','DEA tam çalışma paketi','Girdi kesiti, skorlar, LP sertifikaları, emsaller, bağımsız doğrulama ve Python kodları.','ZIP')
if DEA_AUDIT:
    for source,filename,title,kind in [
        ('SONUCLAR.md','dea_audit_results.md','DEA girdi kontrol sonucu','Belge'),
        ('METODOLOJI.md','dea_audit_methodology.md','DEA girdi kontrol protokolü','Belge'),
        ('quality.csv','dea_audit_quality.csv','DEA girdi/çıktı veri kalitesi','CSV'),
        ('coverage.csv','dea_audit_coverage.csv','DEA girdi senaryoları kapsamı','CSV'),
        ('comparisons.csv','dea_audit_comparisons.csv','Aynı şubelerde girdi karşılaştırması','CSV'),
        ('correlations.csv','dea_audit_correlations.csv','Küme içi girdi ilişkileri','CSV'),
        ('stress_summary.csv','dea_audit_stress.csv','DEA duyarlılık özeti','CSV'),
        ('peer_removal.csv','dea_audit_peer_removal.csv','Tek emsal çıkarma deneyleri','CSV'),
        ('temporal_summary.csv','dea_audit_temporal.csv','DEA aylık skor değişimleri','CSV'),
        ('recommendation.json','dea_audit_recommendation.json','DEA girdi önerisi ve inceleme gerekçeleri','JSON')]:
        add(DEA_AUDIT/source,filename,title,'Müşteri / müşteri+personel; eşit örneklem, yerel stres ve emsal etkisi. Ana model değiştirilmedi.',kind)
    add(DEA_AUDIT.parent/(DEA_AUDIT.name+'.zip'),'dea_audit_package.zip','DEA girdi kontrolleri tam paketi','Sekiz doğrulanmış koşu, stres deneyleri, bağımsız kontrol kodları ve öneri.','ZIP')
for name,label,n in [('train','Eğitim seti',3400),('validation','Doğrulama seti',600),('test','Test seti',600)]:
    add(MODEL/'veri'/f'{name}.csv',f'{name}.csv','V3 '+label,f'{n:,} şube-ay kaydı; ilk V3 deneyi.','CSV')
for name,title,detail in [('ozellik_sozlugu','Özellik sözlüğü','42 alanın tanımı, grubu ve kullanım amacı.')]:
    add(MODEL/'veri'/f'{name}.csv',f'{name}.csv',title,detail,'CSV')
for name,title,detail in [('degisken_onemi','Değişken önemleri','Gain ve doğrulama permutation importance.'),
                         ('kararli_degisken_adaylari','Kararlılık adayları','Üç zaman penceresinde özellik değerlendirmesi.')]:
    add(MODEL/'sonuclar'/f'{name}.csv',f'{name}.csv',title,detail,'CSV')
with zipfile.ZipFile(DOWNLOADS/'python_kodlari.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['pipeline.py','tahmin.py','grafikler.py','dogrula.py','model_karsilastir.py',
                 'model_karsilastir_dogrula.py','model_karsilastirma_raporla.py','tahmin_karsilastir.py',
                 'model_v4_protokol.py','v4_egitim_seti_hazirla.py','model_v4_raporla.py',
                 'flaml_pipeline.py','flaml_tahmin.py','flaml_dogrula.py','flaml_raporla.py','flaml_uyumluluk_kontrol.py',
                 'flaml_ortak_model.py','flaml_ortak_dogrula.py','flaml_ortak_raporla.py',
                 'dashboard_veri_hazirla.py','requirements.txt','README.md']:
        z.write(ROOT/'python_kodlari'/name,arcname='python_kodlari/'+name)
    if 'comparison_v4' in report:z.write(COMPARISON_V4/'requirements.txt',arcname='python_kodlari/requirements_v4.txt')
    if 'automl' in report:z.write(FLAML/'requirements.txt',arcname='python_kodlari/requirements_flaml.txt')
    z.write(ROOT/'python_kodlari/dinamik_feature_selection.py',arcname='python_kodlari/dinamik_feature_selection.py')
    z.write(ROOT/'python_kodlari/kumeleme_karsilastir.py',arcname='python_kodlari/kumeleme_karsilastir.py')
    z.write(ROOT/'python_kodlari/otomatik_kumeleme.py',arcname='python_kodlari/otomatik_kumeleme.py')
    z.write(ROOT/'python_kodlari/dea_verimlilik.py',arcname='python_kodlari/dea_verimlilik.py')
    z.write(ROOT/'python_kodlari/dea_girdi_kontrol.py',arcname='python_kodlari/dea_girdi_kontrol.py')
    z.write(ROOT/'python_kodlari/requirements_dea.txt',arcname='python_kodlari/requirements_dea.txt')
    z.write(ROOT/'python_kodlari/requirements_kumeleme.txt',arcname='python_kodlari/requirements_kumeleme.txt')
    for directory,pattern in [('dinamik_secim','*.py'),('kumeleme','*.py'),('kumeleme_secimi','*.py'),('verimlilik','*.py'),('verimlilik_kontrol','*.py'),('ayarlar','*.json'),('tests','test_*.py')]:
        for path in sorted((ROOT/'python_kodlari'/directory).glob(pattern)):
            z.write(path,arcname=Path('python_kodlari')/path.relative_to(ROOT/'python_kodlari'))
    z.write(ROOT/'python_kodlari/dinamik_secim/README.md',arcname='python_kodlari/dinamik_secim/README.md')
    z.write(ROOT/'python_kodlari/kumeleme_secimi/README.md',arcname='python_kodlari/kumeleme_secimi/README.md')
    z.write(ROOT/'python_kodlari/verimlilik/README.md',arcname='python_kodlari/verimlilik/README.md')
    z.write(ROOT/'python_kodlari/verimlilik_kontrol/README.md',arcname='python_kodlari/verimlilik_kontrol/README.md')
report['downloads'].append({'path':'downloads/python_kodlari.zip','name':'python_kodlari.zip',
 'title':'Python kodları','description':'Pipeline, tahmin, grafik ve doğrulama dosyaları.','kind':'PY / ZIP',
 'bytes':(DOWNLOADS/'python_kodlari.zip').stat().st_size})

# All displayed model values are read from the completed experiment, not invented.
for target in ['tl_vadesiz','tl_vadeli','kredi']:
    for split in ['VALIDATION','TEST']:
        for model in ['yapisal','gecikmeli','son_ayi_koru','egitim_ortalamasi']:
            score=next(x for x in report['total_metrics'] if x['gosterge']==target and x['split']==split and x['model']==model)
            parts=[x for x in report['monthly_metrics'] if x['gosterge']==target and x['split']==split and x['model']==model and x['sube_grubu']=='ALL']
            actual=sum(x['gercek_toplam_tl'] for x in parts)
            error=sum(x['mae_tl']*x['n'] for x in parts)
            assert abs(score['wape']-error/actual)<1e-12
if 'comparison' in report:
    C=report['comparison']
    for score in C['total_metrics']:
        parts=[r for r in C['monthly_metrics'] if r['gosterge']==score['gosterge'] and r['model']==score['model']
               and r['split']==score['split'] and r['sube_grubu']=='ALL']
        assert abs(sum(x['mae_tl']*x['n'] for x in parts)/sum(x['gercek_toplam_tl'] for x in parts)-score['wape'])<1e-12
    assert C['manifest']['checks']['independent_all_72_model_reload_and_inference']
if 'comparison_v4' in report:
    C=report['comparison_v4']
    assert C['manifest']['checks']['independent_saved_models_reload_and_inference']
    assert C['manifest']['checks']['v4_raw_labels_current_owner_features_and_closing_stocks_verified']
    for score in C['total_metrics']:
        parts=[r for r in C['monthly_metrics'] if r['gosterge']==score['gosterge'] and r['model']==score['model']
               and r['split']==score['split'] and r['sube_grubu']=='ALL']
        assert abs(sum(x['mae_tl']*x['n'] for x in parts)/sum(x['gercek_toplam_tl'] for x in parts)-score['wape'])<1e-12
if 'automl' in report:
    C=report['automl']
    for score in C['total_metrics']:
        parts=[r for r in C['monthly_metrics'] if r['gosterge']==score['gosterge'] and r['model']==score['model']
               and r['split']==score['split'] and r['sube_grubu']=='ALL']
        assert abs(sum(x['mae_tl']*x['n'] for x in parts)/sum(x['gercek_toplam_tl'] for x in parts)-score['wape'])<1e-12

encoded=json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)
(DIST/'data.json').write_text(encoded,encoding='utf-8')
(DIST/'data.js').write_text('window.DASHBOARD_DATA='+encoded.replace('</','<\\/')+';\n',encoding='utf-8')
assets=['styles.css','data.js','flaml.js','selection.js','clustering.js','cluster-selection.js','dea-checks.js','dea.js','app.js']
build=hashlib.sha256(b''.join((DIST/name).read_bytes() for name in assets)).hexdigest()[:16]
html=(DIST/'index.html').read_text(encoding='utf-8')
html,replaced=re.subn(r'((?:src|href)=")(styles\.css|data\.js|flaml\.js|selection\.js|clustering\.js|cluster-selection\.js|dea-checks\.js|dea\.js|app\.js)(?:\?v=[^"]*)?(")',
    lambda match:match[1]+match[2]+'?v='+build+match[3],html)
assert replaced==len(assets),'Dashboard asset references are incomplete.'
(DIST/'index.html').write_text(html,encoding='utf-8')
print('Dashboard snapshot:',len(encoded.encode()),'bytes;',len(report['downloads']),'downloads; build',build,'; all metrics verified.')
