import {readFileSync, existsSync, statSync} from 'node:fs';
import {resolve, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
import assert from 'node:assert/strict';

// Render functions are exercised as plain JavaScript; this is not browser QA.
const root=resolve(dirname(fileURLToPath(import.meta.url)), '../dist');
const nodes=new Map();
const node=key=>{
  if(!nodes.has(key))nodes.set(key,{innerHTML:'',textContent:'',value:'',hidden:false,classList:{add(){},remove(){}},focus(){},click(){}});
  return nodes.get(key);
};
const handlers={};
const document={querySelector:node,querySelectorAll:()=>[],getElementById:id=>node('#'+id),activeElement:null,
  addEventListener:(name,fn)=>handlers[name]=fn,createElement:()=>({click(){}})};
const window={scrollTo(){},addEventListener:(name,fn)=>handlers[name]=fn};
const context=vm.createContext({document,window,location:{hash:'#overview'},console,Intl,Blob,URL,setTimeout,clearTimeout});
vm.runInContext(readFileSync(resolve(root,'data.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'flaml.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'selection.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'clustering.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'cluster-selection.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'dea-checks.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'dea.js'),'utf8'),context);
vm.runInContext(readFileSync(resolve(root,'app.js'),'utf8'),context);
const D=window.DASHBOARD_DATA;
const selectionBlocked=!!(D.auto_cluster_selection?.matches_current_grid&&!D.auto_cluster_selection.decision.selected);
let checks=0;
const change=(id,value,checked=false)=>handlers.change({target:{id,value,checked}});
const route=name=>{context.location.hash='#'+name;handlers.hashchange();};
const content=()=>node('#content').innerHTML;
for(const target of ['tl_vadesiz','tl_vadeli','kredi']){
  change('target-select',target);
  for(const group of ['ALL','UC_BUYUK_IL','DIGER_ILLER']){
    change('group-select',group);
    for(const split of ['TEST','VALIDATION']){
      change('split-select',split);
      for(const name of ['overview','pipeline','methodology','data','benchmark','files',...(D.automl?['automl']:[]),...(D.feature_selection?['selection']:[]),...(D.clustering?['clustering']:[]),...(D.auto_cluster_selection?['model-selection']:[]),...(D.dea?['dea']:[])]){
        route(name);
        assert.ok(content().length>500,`${target}/${group}/${split}/${name}`);
        assert.ok(!/NaN|undefined|Infinity/.test(content()),`Missing display value: ${name}`);
        assert.equal(node('.filters').hidden,['methodology','data','files','selection','clustering','model-selection','dea'].includes(name));
        if(D.feature_selection){
          assert.equal(node('#latest-study').hidden,false);
          assert.ok(node('#latest-study').innerHTML.includes(selectionBlocked?'Kalite koşullarını sağlayan ortak model yok':D.dea?.active?'DEA verimlilik analizi tamamlandı':D.clustering?.matches_current_selection?'Şube kümeleme tamamlandı':'Değişken seçimi tamamlandı'));
          assert.ok(node('#latest-study').innerHTML.includes(selectionBlocked?'Yeni atama ve DEA durduruldu':D.dea?.active?'banka hedefini şubelere dağıtma':D.clustering?.matches_current_selection?'DEA tasarımı':'Kümeler henüz oluşturulmadı'));
        }
        checks++;
      }
      assert.ok(!/V3 LightGBM|href="#results"|href="#importance"/.test(node('#navigation').innerHTML));
      for(const legacy of ['results','importance']){
        route(legacy);
        assert.equal(node('#page-title').textContent,'Algoritma karşılaştırması');
        assert.ok(content().includes('Üç algoritmada değişken önemi'));
        checks++;
      }
      route('overview');
      for(const model of ['flaml_automl','ay_sonu_koru']){
        const r=(group==='ALL'?D.automl.total_metrics:D.automl.metrics).find(x=>x.gosterge===target&&x.model===model&&x.split===split&&(group==='ALL'||x.sube_grubu===group));
        const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.wape*100)+'%';
        assert.ok(content().includes(expected),`Overview WAPE differs: ${expected}`);
        checks++;
      }
      assert.ok(content().includes('Bütün göstergelerde ortak model: CatBoost'));
      route('data');
      assert.ok(content().includes('V4 veri akışı'));
      assert.ok(content().includes(new Intl.NumberFormat('tr-TR').format(D.data_manifest_v4.latest_active_customers)));
      if(D.automl){
        route('automl');
        for(const model of ['flaml_automl','secilen_sistem','lightgbm__hareketli','random_forest__hareketli','catboost__hareketli','ay_sonu_koru','son_ayi_koru','egitim_ortalamasi']){
          const r=(group==='ALL'?D.automl.total_metrics:D.automl.metrics).find(x=>x.gosterge===target&&x.model===model&&x.split===split&&(group==='ALL'||x.sube_grubu===group));
          assert.ok(r);
          const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.wape*100)+'%';
          assert.ok(content().includes(expected),`FLAML WAPE differs: ${expected}`);
          checks++;
        }
        assert.ok(content().includes('doğrulama'));
        assert.ok(content().includes('arama eğrisi'));
        if(D.automl.global_selection){
          const g=D.automl.global_selection;
          assert.ok(content().includes('Bütün göstergelerde ortak algoritma:'));
          assert.ok(!content().includes('Referans veya mevcut model de kazanabilir'));
          assert.equal(new Set(D.automl.model_metadata.map(m=>m.best_learner)).size,1);
          assert.ok(D.automl.model_metadata.every(m=>m.best_learner===g.selected_learner&&m.common_algorithm));
          for(const r of g.algorithm_scores){
            assert.ok(Math.abs(Object.values(r.indicator_wape).reduce((a,b)=>a+b,0)/3-r.mean_indicator_wape)<1e-12);
            const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.mean_indicator_wape*100)+'%';
            assert.ok(content().includes(expected));checks++;
          }
          assert.equal(g.selected_learner,[...g.algorithm_scores].sort((a,b)=>a.mean_indicator_wape-b.mean_indicator_wape)[0].learner);
          checks++;
        }
      }
      route('benchmark');
      for(const [version,C] of [['v3',D.comparison],['v4',D.comparison_v4]].filter(x=>x[1])){
        change('benchmark-dataset',version);
      for(const view of Object.keys(C.manifest.feature_counts)){
        change('benchmark-view',view);
        const scores=(group==='ALL'?C.total_metrics:C.metrics)
          .filter(x=>x.gosterge===target&&x.split===split&&(group==='ALL'||x.sube_grubu===group)&&x.gorunum===view);
        assert.equal(scores.length,3);
        for(const score of scores){
          const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:2,maximumFractionDigits:2}).format(score.wape*100)+'%';
          assert.ok(content().includes(expected),`Benchmark WAPE differs: ${expected}`);
        }
        for(const algorithm of ['lightgbm','random_forest','catboost'])for(const metric of ['native','permutation']){
          change('benchmark-algorithm',algorithm);change('benchmark-importance',metric);
          assert.ok(!/NaN|undefined|Infinity/.test(content()));
          assert.ok(content().includes('tüm özellikler'));
          checks++;
        }
        if(version==='v4'){
          assert.ok(content().includes('Ay sonu stokunu koru'));
          assert.ok(content().includes('54 model'));
        }
      }
      }
    }
  }
}
if(D.feature_selection){
  const F=D.feature_selection;
  route('selection');
  assert.ok(node('#navigation').innerHTML.includes('Değişken seçimi'));
  assert.ok(content().includes('Kümeleme için ortak değişken seti'));
  assert.ok(content().includes('Seçilen alanların dönem kararlılığı'));
  assert.ok(content().includes('Alan sözlüğü: kaynak sütunları ve kullanım rolleri'));
  assert.equal(F.manifest.selected_count,F.selected_schema.features.length);
  for(const decision of F.decisions){
    assert.ok(content().includes(decision.name),`Missing selection decision: ${decision.name}`);
    assert.ok(content().includes(decision.reason.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')));
    checks++;
  }
  for(const r of F.metrics){
    const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.wape*100)+'%';
    assert.ok(content().includes(expected));checks++;
  }
  for(const d of F.decisions.filter(r=>r.status==='selected')){
    for(const value of Object.values(d.window_scores)){
      const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(value*100);
      assert.ok(content().includes(expected),`Stability cell differs: ${d.name}`);checks++;
    }
  }
  for(const field of F.selected_schema.config.fields){
    assert.ok(content().includes(field.source),`Missing dictionary source: ${field.source}`);checks++;
  }
  for(const status of new Set(F.decisions.map(r=>r.status))){
    change('selection-status',status);
    assert.ok(!/NaN|undefined|Infinity/.test(content()));
    for(const r of F.decisions.filter(r=>r.status===status))assert.ok(content().includes(r.name));
    checks++;
  }
  change('selection-status','all');
  for(const family of new Set(F.decisions.map(r=>r.family))){
    change('selection-family',family);assert.ok(!/NaN|undefined|Infinity/.test(content()));checks++;
  }
  change('selection-family','all');
  handlers.input({target:{id:'selection-search',value:'__no_such_feature__'}});
  assert.ok(content().includes('Aramayla eşleşen alan yok'));
  handlers.input({target:{id:'selection-search',value:''}});
  // Different indicator names and an extra feature must render without app code edits.
  const alternate=JSON.parse(JSON.stringify(F));
  alternate.manifest.targets=[{name:'different_target',label:'Farklı gösterge'}];
  alternate.metrics=alternate.metrics.map(r=>({...r,target:'different_target'}));
  alternate.decisions.push({name:'new_unseen_feature',label:'Yeni müşteri özelliği',family:'yeni_aile',dtype:'numeric',status:'review_unmapped',reason:'Yeni alan incelemesi'});
  const html=window.renderSelectionPage(alternate,{selectionStatus:'all',selectionFamily:'all',selectionSearch:''});
  assert.ok(html.includes('new_unseen_feature')&&html.includes('Farklı gösterge'));
  assert.ok(!/NaN|undefined|Infinity/.test(html));checks++;
  context.location.hash='';handlers.hashchange();
  assert.equal(node('#page-title').textContent,D.dea?.active?'Küme içi verimlilik analizi':D.auto_cluster_selection?.matches_current_grid?'Ana kümeleme modeli seçimi':D.clustering?.matches_current_selection?'Şube kümeleri ve karşılaştırma':'Dinamik değişken seçimi');checks++;
}
if(D.clustering){
  const C=D.clustering;
  route('clustering');change('cluster-scenario',C.parameter_grid?.selected_case_id||'');
  assert.ok(content().includes(D.dea?.active?'Seçilen ana modelin DEA analizi tamamlandı':selectionBlocked?'güncel politikaya göre uygun ana model yok':'Kümeler oluşturuldu; bu seçim için DEA henüz yapılmadı'));
  for(const method of C.manifest.methods){
    change('cluster-method',method);
    for(const group of ['ALL',...C.manifest.groups]){
      change('cluster-group',group);
      for(const period of [C.manifest.reference_period,C.manifest.as_of]){
        change('cluster-period',period);
        assert.ok(!/NaN|undefined|Infinity/.test(content()));
        for(const r of C.summary.filter(r=>group==='ALL'||r.group===group)){
          const expected=new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.common_gower_silhouette);
          assert.ok(content().includes(expected));checks++;
        }
        const members=C.assignments.filter(r=>r.method===method&&r.__period===period&&r.mode==='frozen'&&(group==='ALL'||r.__group===group));
        assert.ok(content().includes('Küme profilleri')||content().includes('küme profilleri'));
        for(const member of members)assert.ok(content().includes(member.__branch));
        checks++;
      }
    }
  }
  handlers.input({target:{id:'cluster-search',value:'__NO_BRANCH__'}});
  assert.ok(content().includes('Aramayla eşleşen şube yok'));
  handlers.input({target:{id:'cluster-search',value:''}});
  assert.equal(C.manifest.primary_method,'urf_kmedoids');
  assert.equal(C.manifest.dea_executed,false);assert.equal(C.manifest.target_allocation_executed,false);
  if(C.parameter_grid){
    assert.ok(content().includes('Alan oranı × toplam küme sayısı denemeleri'));
    const grid=C.parameter_grid;
    assert.equal(grid.mtry_ratio,.25);
    for(const c of grid.cases){
      change('cluster-scenario',c.case_id);change('cluster-method','urf_kmedoids');change('cluster-group','ALL');change('cluster-period',C.manifest.as_of);
      const resolved=window.resolveClusteringData(C,{clusterScenario:c.case_id});
      assert.equal(resolved.manifest.feature_count,Math.ceil(grid.parent_feature_count*c.feature_ratio));
      assert.ok(content().includes(c.feature_count+' ortak alan'));
      assert.ok(!/NaN|undefined|Infinity/.test(content()));
      assert.equal(Object.values(c.k_by_group).reduce((a,b)=>a+b,0),c.bank_clusters);
      for(const r of resolved.summary.filter(r=>r.method==='urf_kmedoids')){
        assert.equal(r.effective_mtry,Math.max(1,Math.floor(r.encoded_feature_count*.25)));
        assert.ok(content().includes(new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.common_gower_silhouette)));
        checks++;
      }
    }
    change('cluster-scenario','');
    for(const r of grid.bank_scores){
      if(!r.applicable){assert.ok(content().includes('Uygulanamaz: karma veri yok'));continue;}
      assert.ok(content().includes(new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(r.reference_score)));checks++;
    }
  }
  const alternate=JSON.parse(JSON.stringify(C));
  alternate.method_names.urf_kmedoids='Özel URF adı';
  const html=window.renderClusteringPage(alternate,{clusterMethod:'urf_kmedoids',clusterGroup:'ALL',clusterPeriod:C.manifest.as_of,clusterSearch:''});
  assert.ok(html.includes('Özel URF adı')&&!/NaN|undefined|Infinity/.test(html));checks++;
}
if(D.auto_cluster_selection){
  const A=D.auto_cluster_selection,w=A.decision.selected;
  route('model-selection');
  assert.equal(node('.filters').hidden,true);
  assert.ok(content().includes('Adayların ortak sıralaması'));
  assert.ok(content().includes('dokunulmamış test değildir'));
  assert.ok(!/NaN|undefined|Infinity/.test(content()));
  if(A.dea_coverage_contract){
    assert.ok(content().includes('Tam şube kapsamı zorunlu'));
    const contract=A.dea_coverage_contract;
    assert.equal(contract.required_min_cluster_size,A.config.thresholds.min_cluster_size);
    for(const r of A.full_coverage){
      assert.equal(r.source_branches,r.assigned_branches);
      assert.ok(r.min_cluster_size>=contract.required_min_cluster_size);
      assert.ok(content().includes(r.period));checks++;
    }
  }
  if(A.geographic_quality){
    assert.ok(content().includes('Her coğrafi grupta asgari küme ayrışması'));
    for(const row of A.geographic_quality){
      assert.equal(row.passed,row.silhouette!==null&&row.silhouette>=row.threshold);
      assert.ok(content().includes(row.case_id));checks++;
    }
    if(!w){
      assert.ok(content().includes('Güncel politikayla uygun ana model yok'));
      assert.ok(node('#latest-study').innerHTML.includes('Yeni atama ve DEA durduruldu'));
      assert.equal(D.dea?.active,false);assert.equal(D.dea?.input_audit?.active,false);
      assert.ok(A.candidates.every(r=>!r.qualified));checks+=5;
    }
  }
  for(const c of A.candidates){
    assert.ok(content().includes(c.case_id));
    if(c.score!==null)assert.ok(content().includes(new Intl.NumberFormat('tr-TR',{minimumFractionDigits:3,maximumFractionDigits:3}).format(c.score)));
    assert.equal(c.qualified,c.applicable&&c.gates.every(g=>g.passed));checks++;
  }
  if(w){
    assert.ok(content().includes(A.method_names[w.method]));
    assert.ok(content().includes('Seçim ve sonraki dönem kontrolü'));
    assert.equal(new Set(A.manifest.selected_method?[A.manifest.selected_method]:[]).size,1);
    const base=window.resolveClusteringData(D.clustering,{});
    assert.equal(base.viewed_case_id,w.case_id);
    assert.equal(base.display_primary_method,w.method);
    assert.equal(base.manifest.feature_count,w.feature_count);
    assert.equal(base.selected.filter(r=>r.method===w.method).reduce((n,r)=>n+r.k,0),w.bank_clusters);
    assert.ok(node('#latest-study').innerHTML.includes(D.dea?.active?'DEA verimlilik analizi tamamlandı':A.method_names[w.method]));
    change('cluster-scenario','');change('cluster-method','');change('cluster-group','ALL');change('cluster-period','');route('clustering');
    assert.ok(content().includes('Ana yöntem: '+A.method_names[w.method]));
    assert.ok(content().includes(w.feature_count+' ortak alan'));
    const absent=JSON.parse(JSON.stringify(A));absent.decision.selected=null;
    const html=window.renderClusterSelectionPage(absent);
    assert.ok(html.includes('Uygun model yok')&&!/NaN|undefined|Infinity/.test(html));
    const rejected=JSON.parse(JSON.stringify(D.clustering));rejected.automatic_selection=absent;
    assert.ok(window.renderClusteringPage(rejected,{}).includes('Uygun ana model yok'));
    checks+=12;
  }
}
if(D.dea){
  const E=D.dea,nf=(x,d=3)=>new Intl.NumberFormat('tr-TR',{minimumFractionDigits:d,maximumFractionDigits:d}).format(x);
  route('dea');
  assert.ok(content().includes('Verimlilik = 1 / φ'));
  assert.ok(content().includes('gelecek ay büyüme tahmini veya şube hedefi değildir'));
  assert.ok(content().includes(nf(E.summary.mean_efficiency)));
  assert.equal(E.manifest.target_allocation_executed,false);
  assert.equal(E.summary.calculated_branches,E.summary.branches);
  assert.equal(E.summary.withheld_branches,0);
  assert.equal(E.manifest.full_coverage_required,true);
  assert.equal(E.manifest.full_coverage_verified,true);
  assert.equal(E.summary.required_cluster_n,Math.max(E.config.inputs.length*E.config.outputs.length,3*(E.config.inputs.length+E.config.outputs.length)));
  if(E.active)assert.equal(E.manifest.source_selection_run_id,D.auto_cluster_selection.manifest.run_id);
  else assert.ok(content().includes('Tarihsel DEA sonucu'));
  for(const group of ['ALL',...new Set(E.scores.map(r=>r.group))]){
    change('dea-group',group);
    for(const indicator of E.config.outputs){
      change('dea-indicator',indicator.name);
      assert.ok(!/NaN|undefined|Infinity/.test(content()));
      assert.ok(content().includes(indicator.label));
      const shown=E.scores.filter(r=>group==='ALL'||r.group===group);
      for(const r of shown){
        assert.ok(content().includes(r.branch));
        if(r.efficiency!==null)assert.ok(content().includes(nf(r.efficiency)));
        const p=E.potentials.find(p=>p.branch===r.branch&&p.indicator===indicator.name);
        if(p)assert.ok(content().includes(nf(p.frontier_projection,0)));
        checks++;
      }
    }
  }
  change('dea-group','ALL');
  for(const cluster of E.clusters){
    change('dea-cluster',cluster.cluster_id);
    const branch=E.scores.find(r=>r.cluster_id===cluster.cluster_id);
    change('dea-branch',branch.branch);
    assert.ok(content().includes(branch.branch)&&content().includes(cluster.cluster_id));
    for(const field of E.config.outputs){
      const p=E.potentials.find(p=>p.branch===branch.branch&&p.indicator===field.name);
      if(p)assert.ok(content().includes(nf(p.output_slack,0)));
    }
    checks++;
  }
  change('dea-cluster','ALL');
  handlers.input({target:{id:'dea-search',value:'__NO_BRANCH__'}});
  assert.ok(content().includes('Aramayla eşleşen şube yok'));
  handlers.input({target:{id:'dea-search',value:''}});
  const stale=JSON.parse(JSON.stringify(E));stale.active=false;
  assert.ok(window.renderDEAPage(stale,{}).includes('tarihsel sonuçtur'));
  // Model/schema names and output count are supplied by the dictionary, not fixed in renderer.
  const alternate=JSON.parse(JSON.stringify(E));alternate.config.outputs[0].label='Farklı gösterge etiketi';
  assert.ok(window.renderDEAPage(alternate,{}).includes('Farklı gösterge etiketi'));
  checks+=10;
}
if(D.dea?.input_audit){
  const A=D.dea.input_audit;
  route('dea');
  assert.ok(content().includes('DEA girdi kontrolleri'));
  assert.ok(content().includes('bütün şubeler için ortak kontrol sonucu'));
  assert.equal(A.manifest.source_dea_run_id,D.dea.manifest.run_id);
  assert.equal(A.active,D.dea.active);assert.equal(A.recommendation.main_model_changed,false);
  const nf=(x,d=3)=>new Intl.NumberFormat('tr-TR',{minimumFractionDigits:d,maximumFractionDigits:d}).format(x);
  const c=A.comparisons.find(r=>r.period===A.manifest.as_of);
  assert.ok(A.coverage.every(r=>r.coverage===1&&r.withheld_branches===0&&r.calculated_branches===r.branches));
  assert.ok(content().includes(nf(c.base_mean_same_cohort))&&content().includes(nf(c.augmented_mean_same_cohort)));
  assert.ok(content().includes(String(c.common_scored)));
  assert.ok(content().includes('güven aralığı'));
  for(const r of A.stress_summary){
    assert.ok(content().includes(nf(r.local.p95_abs_change,4)));
    assert.ok(content().includes(nf(r.peer_removal.max_abs_change,4)));checks++;
  }
  for(const r of A.influential_peers){
    assert.ok(content().includes(r.removed_peer));assert.ok(content().includes(r.worst_affected_branch));checks++;
  }
  for(const r of A.correlations.filter(r=>r.period===A.manifest.as_of)){
    if(r.spearman!==null)assert.ok(content().includes(nf(r.spearman)));checks++;
  }
  const stale=JSON.parse(JSON.stringify(A));stale.active=false;
  assert.ok(window.renderDEAInputChecks(stale).includes('tarihsel sonuçtur'));checks++;
  assert.ok(!/NaN|undefined|Infinity/.test(content()));
}
route('pipeline');
for(let id=1;id<=6;id++){
  handlers.click({target:{closest:selector=>selector==='[data-stage]'?{dataset:{stage:String(id)}}:null}});
  assert.ok(content().includes('AŞAMA '+String(id).padStart(2,'0')));checks++;
}
for(const f of D.downloads){
  const path=resolve(root,f.path);
  assert.ok(path.startsWith(root+'/')&&existsSync(path),f.path);
  assert.equal(statSync(path).size,f.bytes,f.path);checks++;
}
const html=readFileSync(resolve(root,'index.html'),'utf8');
const assets=[...html.matchAll(/(?:src|href)="([^"?]+)\?v=([a-f0-9]{16})"/g)];
assert.equal(assets.length,9);
assert.equal(new Set(assets.map(m=>m[2])).size,1);
for(const [,name] of assets){assert.ok(existsSync(resolve(root,name)));checks++;}
console.log(JSON.stringify({render_checks:checks,download_files:D.downloads.length,status:'passed'}));
