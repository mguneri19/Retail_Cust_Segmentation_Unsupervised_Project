# 🛍️ Retail Customer Segmentation with Unsupervised Learning

Bu proje, perakende sektöründeki müşteri verilerini kullanarak **gözetimsiz öğrenme** yöntemleriyle müşteri segmentasyonu yapmaktadır. K-Means ve Hierarchical Clustering algoritmaları kullanılarak müşteriler davranışsal özelliklerine göre gruplandırılmıştır.

## 📊 Proje Özeti

### İş Problemi
Perakende işletmelerinin müşteri tabanını davranışsal özelliklerine göre segmentlere ayırarak:
- **VIP müşterileri** belirleme
- **Churn riski** olan müşterileri tespit etme  
- **Pazarlama stratejileri** için hedef kitle oluşturma
- **Müşteri yaşam döngüsü** analizi yapma

### Veri Seti
- **20.000 müşteri** verisi (2020-2021)
- **13 değişken** (demografik ve davranışsal)
- **OmniChannel** alışveriş geçmişi (online + offline)
- Perakende sektörü müşteri davranışları

## 🚀 Özellikler

### Veri Ön İşleme
- ✅ **Feature Engineering**: Recency, Tenure, Total Value hesaplama
- ✅ **Log Transformation**: Çarpık dağılımları düzeltme
- ✅ **Scaling**: MinMaxScaler ile normalizasyon
- ✅ **Skewness Analysis**: İstatistiksel veri kalitesi kontrolü

### Machine Learning Algoritmaları
- 🔹 **K-Means Clustering**: Elbow method ile optimum küme sayısı
- 🔹 **Hierarchical Clustering**: Ward linkage ile dendrogram analizi
- 🔹 **Segment Profiling**: Detaylı istatistiksel analiz

### Segmentasyon Sonuçları

#### K-Means (4 Segment)
| Segment | Müşteri Sayısı | Ortalama Değer | Özellik |
|---------|----------------|----------------|---------|
| **VIP** | 4,729 | ₺1,441 | En değerli, sadık müşteriler |
| **Potansiyel** | 4,206 | ₺751 | Orta düzey, gelişim potansiyeli |
| **İnaktif** | 5,548 | ₺514 | Düşük aktivite, churn riski |
| **Yeni** | 5,462 | ₺395 | Yeni müşteriler, düşük hacim |

#### Hierarchical Clustering (3 Segment)
| Segment | Müşteri Sayısı | Ortalama Değer | Özellik |
|---------|----------------|----------------|---------|
| **VIP** | 6,226 | ₺1,072 | En değerli müşteriler |
| **Orta Düzey** | 10,204 | ₺640 | Potansiyeli yüksek müşteriler |
| **İnaktif** | 3,515 | ₺506 | Geri kazanılması gereken |

## 📁 Proje Yapısı

```
Retail_Customer_Segmentation_ML/
├── data/
│   └── customer_data_20k.csv               # Ham veri seti
├── CustomerSegment_ML.py                   # Ana analiz dosyası
├── Retail_Customer_Segmentation.pbix       # Power BI Dashboard
├── customer_segments.csv                   # Segmentasyon sonuçları
├── requirements.txt                        # Python bağımlılıkları
├── README.md                               # Proje dokümantasyonu
├── LICENSE                                 # MIT Lisansı
├── theme1.json                            # Power BI tema dosyası
├── Power BI Dashboard/                     # Dashboard görüntüleri
│   ├── Yönetim Analitik Paneli.png        # Ana dashboard
│   ├── Segmentasyon Karşılaştırması.png   # Segment analizi
│   └── Müşteri Detay Profili.png          # Detay sayfası
├── dendrogram.png                         # Hierarchical clustering görseli
└── elbow_curve.png                        # K-Means elbow curve
```

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Çalıştırma
```bash
python CustomerSegment_ML.py
```

### Power BI Dashboard
Proje ayrıca Power BI dashboard'u içermektedir:
- `Retail_Customer_Segmentation.pbix` dosyasını Power BI Desktop ile açın
- Segmentasyon sonuçlarını interaktif olarak keşfedin
- Özel tema dosyası (`theme1.json`) ile tutarlı görselleştirme

### Gerekli Kütüphaneler
- `pandas` >= 1.5.0
- `numpy` >= 1.21.0
- `matplotlib` >= 3.5.0
- `seaborn` >= 0.11.0
- `scikit-learn` >= 1.1.0
- `scipy` >= 1.9.0
- `yellowbrick` >= 1.4.0

## 📈 Sonuçlar ve İş Değeri

### Segment Bazlı Aksiyon Önerileri

#### 🏆 VIP Segment
- **Özel kampanyalar** ve premium hizmetler
- **Kişiselleştirilmiş** ürün önerileri
- **Sadakat programları** ile ödüllendirme

#### 📈 Potansiyel Segment  
- **Upselling/Cross-selling** fırsatları
- **Kategori genişletme** kampanyaları
- **Omnichannel** deneyim iyileştirme

#### ⚠️ İnaktif Segment
- **Re-engagement** kampanyaları
- **Churn prevention** stratejileri
- **Özel indirimler** ile geri kazanma

#### 🌱 Yeni Segment
- **Onboarding** süreçleri
- **Eğitim içerikleri** ve rehberlik
- **İlk alışveriş** teşvikleri

## 🔍 Teknik Detaylar

### Feature Engineering
```python
# Recency: Son alışverişten geçen gün sayısı
df["recency"] = (today_date - df["last_order_date"]).dt.days

# Tenure: Müşteri yaşı (ilk alışverişten itibaren)
df["tenure"] = (today_date - df["first_order_date"]).dt.days

# Total Value: Toplam müşteri değeri
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
```

### Model Parametreleri
- **K-Means**: n_clusters=4, random_state=42
- **Hierarchical**: n_clusters=3, linkage='ward'
- **Scaling**: MinMaxScaler(0,1)
- **Transformation**: log1p for skewed variables

## 📊 Görselleştirmeler

Proje aşağıdaki görselleştirmeleri içerir:

### Python Görselleştirmeleri
- **Elbow Curve**: Optimum küme sayısı belirleme
- **Dendrogram**: Hierarchical clustering ağacı
- **Distribution Plots**: Değişken dağılımları
- **Segment Comparison**: Küme karşılaştırmaları

### Power BI Dashboard

#### 📊 Yönetim Analitik Paneli
- **Dinamik Yıl Filtresi**: Seçilen yıla göre tüm metrikler güncellenir
- **Ana KPI'lar**: Müşteri sayısı, toplam gelir, ortalama gelir
- **Online Analizi**: Online gelir oranı ve satış kanalı dağılımı
- **Trend Analizleri**: Müşteri sayısı ve gelir-sipariş trendleri
- **Müşteri Profili**: Aktif müşteri oranı ve sadık müşteri oranı

#### 🔍 Segmentasyon Karşılaştırması
- **Algoritma Karşılaştırması**: K-Means vs Ward Clustering
- **Sipariş Dağılımı**: Her segment için sipariş sayısı analizi
- **Detaylı Tablolar**: Segment bazlı istatistiksel karşılaştırmalar
- **Drill-Through**: Dinamik olarak müşteri detay profiline geçiş

#### 👤 Müşteri Detay Profili
- **Bireysel Analiz**: Drill-through ile tek müşteri detayları
- **Segment Bilgileri**: Müşterinin hangi segmentte olduğu ve sınıfları
- **Sipariş Analizi**: Toplam sipariş sayısı ve detayları
- **Gelir Analizi**: Müşteri değeri (monetary) ve gelir bilgileri
- **Zaman Analizi**: Recency (son alışveriş) ve Tenure (müşteri yaşı)
- **Davranış Profili**: Müşteri alışveriş geçmişi ve tercihleri
- **Dark Theme**: Profesyonel görsel tasarım



## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

**Muhammet Güneri**
- GitHub: [@muhammetguneri](https://github.com/muhammetguneri)
- LinkedIn: [Muhammet Güneri](https://linkedin.com/in/muhammetguneri)

## 🙏 Teşekkürler

- Perakende müşteri segmentasyonu veri seti
- Scikit-learn ve Yellowbrick kütüphaneleri
- Microsoft Power BI platformu
- Python data science topluluğu

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**