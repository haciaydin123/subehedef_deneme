import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NAMES={'tl_vadesiz':'TL vadesiz','tl_vadeli':'TL vadeli','kredi':'Kredi'}
MODELS={'son_ayi_koru':'Son ayı koru','yapisal':'Yapısal LightGBM','gecikmeli':'Gecikmeli LightGBM'}
COLORS={'son_ayi_koru':'#68788A','yapisal':'#245A81','gecikmeli':'#C88932'}

def run(out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    results=out/'sonuclar';charts=out/'grafikler';charts.mkdir(exist_ok=True)
    total=pd.read_csv(results/'toplam_metrikler.csv',encoding='utf-8-sig')
    test=total[total['split'].eq('TEST')]
    fig,ax=plt.subplots(figsize=(10.5,4.8),layout='constrained')
    positions=np.arange(3);width=.24
    for i,model in enumerate(MODELS):
        values=[test[test['gosterge'].eq(t)&test['model'].eq(model)]['wape'].iloc[0] for t in NAMES]
        bars=ax.bar(positions+(i-1)*width,values,width,label=MODELS[model],color=COLORS[model])
        ax.bar_label(bars,labels=[f'{v*100:.2f}%' for v in values],padding=4,fontsize=10)
    ax.set_xticks(positions,list(NAMES.values()));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0,max(test[test['model'].isin(MODELS)]['wape'])*1.28)
    ax.set_ylabel('Test WAPE — daha düşük daha iyi');ax.set_title('Ekim–Aralık 2026: sentetik veri üzerinde bir ay ileri tahmin',loc='left',pad=18)
    ax.legend(frameon=False,ncols=3,loc='upper center');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(charts/'test_hatalari.png',dpi=180);plt.close(fig)

    imp=pd.read_csv(results/'degisken_onemi.csv',encoding='utf-8-sig')
    stable=pd.read_csv(results/'kararli_degisken_adaylari.csv',encoding='utf-8-sig')
    preds=pd.read_csv(results/'test_ve_validasyon_tahminleri.csv',encoding='utf-8-sig')
    fig,axes=plt.subplots(3,2,figsize=(17,13),layout='constrained')
    for i,t in enumerate(NAMES):
        for j,group in enumerate(['UC_BUYUK_IL','DIGER_ILLER']):
            ax=axes[i,j]
            part=imp[imp['gosterge'].eq(t)&imp['sube_grubu'].eq(group)&imp['model'].eq('yapisal')&~imp['ozellik_grubu'].eq('takvim')]
            part=part.nlargest(8,'normalize_delta_mae').sort_values('normalize_delta_mae')
            ss=stable[stable['gosterge'].eq(t)&stable['sube_grubu'].eq(group)].set_index('alan_veya_grup')['kararli_aday']
            base=preds[preds['gosterge'].eq(t)&preds['sube_grubu'].eq(group)&preds['model'].eq('yapisal')&preds['split'].eq('VALIDATION')]['gercek_tl'].mean()
            colors=['#245A81' if bool(ss[x]) else '#A8B3BD' for x in part['alan_veya_grup']]
            ax.barh(part['alan_veya_grup'],part['normalize_delta_mae'],xerr=part['delta_mae_std_tl']/base,
                    color=colors,error_kw={'ecolor':'#59656E','capsize':2,'linewidth':.8})
            ax.xaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='y',labelsize=9)
            ax.set_title(f'{NAMES[t]} — '+('İstanbul/Ankara/İzmir' if j==0 else 'Diğer iller'),loc='left',fontsize=12)
            ax.set_xlabel('ΔMAE / ortalama gerçekleşme');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Yapısal modeller: doğrulama değişken önemleri\nKoyu: üç dönemin en az ikisinde pozitif aday. Hata çubukları tekrar std.; güven aralığı değildir.',fontsize=15)
    fig.savefig(charts/'degisken_onemleri.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    periods=sorted(preds['hedef_donemi'].unique());xx=np.arange(len(periods))
    for ax,t in zip(axes,NAMES):
        part=preds[preds['gosterge'].eq(t)]
        actual=part[part['model'].eq('son_ayi_koru')].groupby('hedef_donemi')['gercek_tl'].sum().reindex(periods)/1e6
        ax.plot(xx,actual,color='#182B3C',linewidth=2.5,marker='o',label='Gerçekleşme')
        for model in MODELS:
            series=part[part['model'].eq(model)].groupby('hedef_donemi')['tahmin_tl'].sum().reindex(periods)/1e6
            ax.plot(xx,series,color=COLORS[model],linestyle='--',marker='.',label=MODELS[model])
        ax.axvspan(2.5,5.5,color='#DDE8F2',alpha=.5)
        ax.set_xticks(xx,periods);ax.set_xlim(-.15,5.15);ax.set_ylabel('Milyon TL');ax.set_title(NAMES[t],loc='left')
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,ncols=2,loc='upper left')
    fig.suptitle('Banka toplamı: bir ay ileri model tahminleri\nTemmuz–Eylül doğrulama; mavi alan Ekim–Aralık test. Finansal veri sentetiktir.',fontsize=14)
    fig.savefig(charts/'aylik_toplam_tahminler.png',dpi=170);plt.close(fig)

    manifest=json.loads((out/'manifest.json').read_text())
    text=['# İlk LightGBM çalışmasının sonuçları','',
      'Üç gösterge iki şube grubunda ayrı modellenmiştir. 6 yapısal model, 6 gecikmeli karşılaştırma modeli ve iki basit referans vardır.',
      '', '## Test sonuçları', '', '| Gösterge | Son ayı koru WAPE | Yapısal LightGBM WAPE | Gecikmeli LightGBM WAPE |',
      '|---|---:|---:|---:|']
    for t,name in NAMES.items():
        vals=[test[test['gosterge'].eq(t)&test['model'].eq(m)]['wape'].iloc[0]*100 for m in MODELS]
        text.append(f'| {name} | %{vals[0]:.2f} | %{vals[1]:.2f} | %{vals[2]:.2f} |')
    text += ['', '**Bu sentetik testte basit son ay referansı üç göstergede de LightGBM modellerinden daha iyi sonuç verdi.**',
      'Yapısal modellerin yüksek R² değerleri bunu değiştirmez. Yalnızca mevcut bakiyeyi bilmek bu veride güçlü bir bilgidir.',
      'Kasım–Aralık aylarının farklı sentetik devam mekanizmasıyla üretilmesi de ayrı bir sınırlamadır. Bu sonuç görüldükten sonra test hatasını düşürmek için ayar değiştirilmedi.',
      '',f'Veri hazırlama: {manifest["preprocessing_seconds"]:.2f} saniye. Modelleme/ayar/önem dahil pipeline: {manifest["pipeline_seconds"]:.2f} saniye.',
      'İlk paket kurulumu bu sürelerin dışındadır. Random Forest kıyaslaması yapılmadığından ona karşı hız üstünlüğü iddia edilmez.',
      '', '## Kararlı değişken adayları','',
      'Aşağıdakiler her modelde üç dönemden en az ikisinde pozitif bulunan ilk beş adaydır. Ortak nihai kümeleme listesi henüz seçilmedi.']
    for group in ['UC_BUYUK_IL','DIGER_ILLER']:
        for t,name in NAMES.items():
            part=stable[stable['sube_grubu'].eq(group)&stable['gosterge'].eq(t)&stable['kararli_aday']].head(5)
            text.append(f'- {group}, {name}: '+', '.join(part['alan_veya_grup'])+'.')
    text += ['', '## Dosyalar', '',
      '- `veri/`: eğitim, doğrulama, test; 200 şube için Ocak tahmin girdileri; özellik sözlüğü.',
      '- `modeller/`: 12 evaluation ve 12 production native model; her model ailesi için metadata.',
      '- `sonuclar/`: grup/toplam/aylık hatalar, süreler, önemler, dönem kararlılığı ve tahminler.',
      '- `METODOLOJI.md`: hedef tanımı, zaman ayrımı, özellik sınıfları, seçim ve sınırlar.',
      '- `LightGBM_sonuc_raporu.xlsx`: kısa inceleme raporu; tam eğitim tabloları CSV dosyalarındadır.',
      '', '## Yeniden çalıştırma','',
      'Python 3.12 ve `requirements.txt` sürümleri kullanılmıştır. Girdi olarak önceki V3 CSV paketi gerekir.',
      'Paket klasörünün içinden:', '', '```bash',
      'python -m venv .venv', 'source .venv/bin/activate','python -m pip install -r requirements.txt',
      'python kod/pipeline.py --source ../banka_test_verisi_v3 --output .',
      'python kod/grafikler.py --output .','```',
      '', 'Modellerden birini Ocak girdileriyle kullanmak için:', '', '```bash',
      'python kod/tahmin.py --metadata modeller/tl_vadesiz__UC_BUYUK_IL__gecikmeli__metadata.json --input veri/ocak2027_tahmin_girdisi.csv --output ornek_tahmin.csv',
      '```', '', 'Ocak model tahminleri banka bütçesi değildir. %15/%10/%5 üst toplam hedefleriyle zorla eşitlenmemiştir.']
    (out/'SONUCLAR.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Charts and SONUCLAR.md saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output.resolve())
