# Dinamik değişken seçimi

Çalışma: 2026-12_88862bd5_20eceb8d_1464854a. Veri sonu: 2026-12; seçim etiketi sonu: 2026-09.
23 tanımlı aday → 22 kalite/korelasyon sonrası aday → **22 seçilen alan**.

Amaç müşteri kitlesi/potansiyeli benzer şubeler için ortak kümeleme girdisi hazırlamaktır. Kümeleme, DEA ve hedef dağıtımı çalıştırılmadı.
CatBoost yalnızca geçmiş üretim ilişkisini sınamak ve değişken seçimini desteklemek için kullanılır. Ay sonu bakiye anchor’ı kullanılmaz. Bu önemler önceki 49 alanlı tahmin modelinin önemleri değildir.

| Alan | Karar | Ortalama ΔWAPE | Pozitif dönem | Gerekçe |
|---|---|---:|---:|---|
| musteri_sayisi | selected | 16.9950 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| ortalama_aylik_gelir_tl | selected | 3.5208 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| sosyoekonomik_skor_sentetik | selected | 2.4974 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| personel_fte_sentetik | selected | 1.6157 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| nufus_sentetik | selected | 0.8230 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| kaynak_gelir_bilinmiyor_orani | selected | 0.6362 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| emekli_maasi_orani | selected | 0.5451 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| rakip_sube_sayisi_sentetik | selected | 0.4419 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| yas_35_alti_orani | selected | 0.4399 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| egitim_bilinmiyor_orani | selected | 0.4351 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| evli_orani | selected | 0.3933 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| yuksek_egitim_orani | selected | 0.3918 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| ortalama_iliski_skoru | selected | 0.3367 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| kaynak_pasif_ay_ort | selected | 0.3088 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| yas_std | selected | 0.3087 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| yas_60_ustu_orani | selected | 0.2738 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| ortalama_yas | selected | 0.2699 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| mobil_aktif_orani | selected | 0.2474 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| maas_musterisi_orani | selected | 0.2390 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| bakim_yukumlulugu_ort | selected | 0.1872 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| ana_banka_orani | selected | 0.1374 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| ekonomik_yapi_sentetik | selected | 0.1166 yüzde puan | 3 | Pozitif katkı ve dönem kararlılığı koşulları sağlandı |
| kaynak_ust_gelir_orani | excluded_correlated | — | — | Benzer bilgiyi ortalama_aylik_gelir_tl temsil ediyor |
| is_birimi | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| ilce_id | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| il | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| sube_segmenti | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| ay_ici_hizmet_verilen_musteri_sayisi | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| musteri_gun | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| havuz_musteri_sayisi | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| portfoy_musteri_sayisi | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| havuz_portfoy_orani | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| tl_vadesiz | output_only | — | — | Açıklanacak üretim; X içine alınmaz |
| doviz_vadesiz_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| altin_hesabi_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| tl_vadeli | output_only | — | — | Açıklanacak üretim; X içine alınmaz |
| doviz_vadeli_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| ihtiyac_kredi_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| konut_kredi_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| tasit_kredi_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| kmh_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| fon_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| hisse_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| tahvil_bono_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| kart_borc_ort_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| vadesiz_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| vadeli_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| kredi | output_only | — | — | Açıklanacak üretim; X içine alınmaz |
| yatirim_ort_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| toplam_varlik_ort_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| yeni_musteri_sayisi | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| ayrilan_musteri_sayisi | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| devir_giris_musteri_sayisi | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| devir_cikis_musteri_sayisi | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| diger_bankaya_kaynak_cikisi_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| diger_bankadan_kaynak_girisi_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| tl_vadesiz_ay_sonu_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| tl_vadeli_ay_sonu_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| kredi_ay_sonu_bakiye_tl | excluded_role | — | — | Mevcut üretim/hareket veya idari bilgi; ilk emsal kümeleme setinin dışında |
| portfoy_musteri_orani | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| ortalama_varlik_tl | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| segment_s1_orani | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| segment_s2_orani | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
| segment_s5_orani | review_role | — | — | Banka içi varlık/segment veya idari tanım; bağımsız potansiyel olarak iş incelemesi gerekli |
