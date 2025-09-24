###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
"""
Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )
müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.
"""

###############################################################
# Veri Seti Hikayesi
###############################################################
"""
Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline)
olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

"""
#ilgili kütüphanelerin yüklenmesi

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

# çıktıyı daha okunur hale getirmek için
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


# Veri okuma: proje klasöründeki 'data' alt klasöründen CSV'yi yükle
df_ = pd.read_csv("data/flo_data_20k.csv")
# Orijinali korumak için bir kopya üzerinde çalış (daha güvenli pratik)
df = df_.copy()

print(df.shape) #19945 gözlem, 12 değişken


def check_dataset(dataframe, head=5):
    """
    Veri setini hızlıca tanımak için yardımcı fonksiyon.
    """
    print("### Veri Seti Şekli ###")
    print(dataframe.shape)
    print("\n### Veri Tipleri ###")
    print(dataframe.dtypes)
    print("\n### İlk {} Gözlem ###".format(head))
    print(dataframe.head(head))
    print("\n### Eksik Değerler ###")
    na_cols = dataframe.isnull().sum()
    print(na_cols[na_cols > 0])
    print("\n### Betimsel İstatistikler ###")
    print(dataframe.describe().T)
    print("\n### Kategorik Değişkenlerin Dağılımı ###")
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    for col in cat_cols:
        print(f"\n{col} sütunu dağılımı:")
        print(dataframe[col].value_counts().head())


check_dataset(df)

# tarih değişkeni kategorik gözüküyor, içinde tarih geçen değişkenleri çevirelim
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Bugünün tarihi - Recency hesaplamak için
today_date = df["last_order_date"].max() + pd.Timedelta(days=2)
#bugünün tarihini son siparişten 2 gün sonrasını belirliyorum

# Toplam sipariş sayısı
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
# Toplam müşteri değeri
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Recency (en son alışveriş üzerinden geçen gün sayısı)
df["recency"] = (today_date - pd.to_datetime(df["last_order_date"])).dt.days

# Tenure (müşteri ilk alışverişini yapalı kaç gün olmuş)
df["tenure"] = (today_date - pd.to_datetime(df["first_order_date"])).dt.days

# Ortalama sepet tutarı
df["avg_order_value"] = df["total_customer_value"] / df["total_order_num"]

# Online alışveriş oranı
df["online_ratio"] = df["order_num_total_ever_online"] / df["total_order_num"]

check_dataset(df, head=10)

###########################################
# K-Means ile Müşteri Segmentasyonu
###########################################
#Çarpıklık Analizi
def check_skew(df_skew, column):
    """
    Belirli bir sütunun çarpıklığını (skewness) hesaplar, test eder ve görselleştirir.
    Ayrıca sonuçları konsola yazdırır.
    """
    # Çarpıklık (skewness) hesapla
    skew = stats.skew(df_skew[column])
    # Çarpıklık testi yap (istatistiksel olarak anlamlı mı?)
    skewtest = stats.skewtest(df_skew[column])

    # Grafik
    plt.figure(figsize=(6, 4))  # Her sütun için ayrı pencere
    sns.histplot(df_skew[column], kde=True, color="g")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show(block=True)

    # Sonuçları ekrana yazdır
    print(f"{column} -> Skew: {skew:.4f}, SkewTest: {skewtest}")
    return


# İncelenecek sayısal sütunlar
num_features = ['order_num_total_ever_online',
 'order_num_total_ever_offline',
 'customer_value_total_ever_offline',
 'customer_value_total_ever_online',
 'recency',
 'total_order_num',
 'total_customer_value',
 'tenure',
 'avg_order_value',
 'online_ratio']

# Her sütun için ayrı ayrı grafik çiz
for col in num_features:
    check_skew(df, col)

"""
Neredeyse tüm değişkenler çarpık;
recency ve online_ratio log. dönüşümü yapmaya gerek yok.

order_num_total_ever_online -> Skew: 10.4877, 
order_num_total_ever_offline -> Skew: 20.3281, 
customer_value_total_ever_offline -> Skew: 16.2995, 
customer_value_total_ever_online -> Skew: 20.0843,
recency -> Skew: 0.6172,
total_order_num -> Skew: 9.1422, 
total_customer_value -> Skew: 17.4053,
tenure -> Skew: 1.7780, 
avg_order_value -> Skew: 14.0266,
online_ratio -> Skew: -0.0114,

"""
# Normal dağılımın sağlanması için Log transformation uygulanması
# Log dönüşümü yapılacak kolonlar
log_transform_cols = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online",
    "total_order_num",
    "total_customer_value",
    "tenure",
    "avg_order_value"
]

# Hariç tutulacak kolonlar
exclude_cols = ["recency", "online_ratio"]

model_df = df[log_transform_cols + exclude_cols].copy()
print(model_df.columns)

# Sadece log dönüşümü yapılacak kolonlarda uygula
model_df[log_transform_cols] = model_df[log_transform_cols].apply(np.log1p)

# Dönüşümden önce ve sonra verileri kontrol
print("Orijinal df ilk 5 satır:")
print(df[log_transform_cols + exclude_cols].head())

print("\nLog dönüşümü yapılmış model_df ilk 5 satır:")
print(model_df.head())

# Scaling İşlemleri
# Mesafeye dayalı KNN işlemi için ölçeklendirme işlemi yapıyoruz
scaler = MinMaxScaler()
scaled_model = scaler.fit_transform(model_df)

scaled_model_df = pd.DataFrame(scaled_model, columns=model_df.columns)
scaled_model_df.head()

# Optimum küme sayısını bulmak için
# KMeans nesnesi
kmeans = KMeans(random_state=42)

# Elbow yöntemi - 2'den 10'a kadar küme sayısını deneyecek
elbow = KElbowVisualizer(kmeans, k=(2, 10),
                                 timings = False,
                                 locate_elbow = True
)
elbow.fit(scaled_model_df)
elbow.show(outpath="elbow_curve.png")

"""
Optimum küme sayısı 4 olduğunu gözlemiyoruz.
"""


# KMeans modeli oluşturma
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")

# scaled_model_df üzerinde modeli fit et ve cluster etiketlerini al
clusters = kmeans.fit_predict(scaled_model_df)

# Orijinal df'ye cluster etiketlerini ekle
df["cluster"] = clusters

# Kontrol
print(df["cluster"].value_counts())
"""
cluster
0    5548
2    5462
3    4729
1    4206
"""

df.head()

segment_summary = df.groupby("cluster").agg({
    "order_num_total_ever_online": ["mean", "min", "max"],
    "order_num_total_ever_offline": ["mean", "min", "max"],
    "customer_value_total_ever_online": ["mean", "min", "max"],
    "customer_value_total_ever_offline": ["mean", "min", "max"],
    "recency": ["mean", "min", "max"],
    "tenure": ["mean", "min", "max"],
    "total_order_num": ["mean", "min", "max"],
    "total_customer_value": ["mean", "min", "max"],
    "avg_order_value": ["mean", "min", "max"],
    "online_ratio": ["mean", "min", "max"],
    "master_id": "count"  # Segmentteki müşteri sayısı
})

# Segmentleri değerli müşterilerden başlayarak sırala
segment_summary = segment_summary.sort_values(
    ("total_customer_value", "mean"), ascending=False
)

print(segment_summary)

"""
Kümeleri incelediğimizde en değerli küme / cluster = 3

============================================
 MÜŞTERİ SEGMENT ÖZET TABLOSU (ORTALAMALARLA)
============================================

Cluster | Müşteri Sayısı | Recency (Aktiflik) | Tenure (Sadakat) | Ortalama Toplam Değer | Ortalama Toplam Sipariş | Online Oranı | Genel Yorum
------------------------------------------------------------------------------------------------------------------------------------------------------------
   3    |     4729       | 87.51 (Orta)       | 1045.12 (Yüksek) | 1441.05 (Çok Yüksek)    | 9.25                  | 0.79         | VIP, sadık, değerli müşteriler
   1    |     4206       | 87.54 (Orta)       | 764.23 (Orta)    | 751.02 (Orta)           | 5.10                  | 0.32         | Terk riski olan eski offline ağırlıklı
   0    |     5548       | 270.32 (Yüksek)    | 838.42 (Orta)    | 514.17 (Düşük)          | 3.62                  | 0.53         | İnaktif, düşük harcama gücü
   2    |     5462       | 73.23 (Orta)       | 581.06 (Orta)    | 395.00 (Düşük)          | 2.74                  | 0.56         | Yeni, düşük hacimli müşteriler

"""


####################################################
# Hierarchical Clustering ile Müşteri Segmentasyonu
####################################################

# Ward yöntemi kullanarak bağlantı matrisi oluşturma
linkage_matrix = linkage(scaled_model_df, method='ward')

dists = linkage_matrix[:, 2]         # Birleşme mesafeleri
diffs = np.diff(dists)               # Mesafe farkları
jump_idx = np.argmax(diffs)          # En büyük sıçrama
y_cut = (dists[jump_idx] + dists[jump_idx + 1]) / 2

print(f"Önerilen y_cut: {y_cut:.2f}")

# Dendrogram çizimi
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='lastp',
                           p=20, leaf_rotation=45,
                           leaf_font_size=10,
                           show_contracted=True)

plt.axhline(y=y_cut, color='red', linestyle='--', linewidth=1.5 )
plt.title("Hiyerarşik Kümelenme Dendogramı")
plt.xlabel("Gözlem birimleri")
plt.ylabel("Uzaklık (Distance)")

# Grafiği kaydet
plt.tight_layout()  # kenar boşluklarını otomatik düzenler
plt.savefig("dendrogram.png", dpi=300, bbox_inches='tight')
plt.show(block=True)

"""
Önerilen y_cut = 30.77 göre optimum küme sayısı 3 olarak görülüyor.
"""

# k=3, Ward Linkage ile model oluşturma
ward_cluster = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

# scaled_model_df: log ve MinMaxScaler uygulanmış numerik veri
df["ward_cluster"] = ward_cluster.fit_predict(scaled_model_df)

print("Ward linkage küme dağılımı:")
print(df["ward_cluster"].value_counts())

df.head()

#Segmentleri inceliyoruz
segment_summary2 = df.groupby("ward_cluster").agg({
    "order_num_total_ever_online": ["mean", "min", "max"],
    "order_num_total_ever_offline": ["mean", "min", "max"],
    "customer_value_total_ever_online": ["mean", "min", "max"],
    "customer_value_total_ever_offline": ["mean", "min", "max"],
    "recency": ["mean", "min", "max"],
    "tenure": ["mean", "min", "max"],
    "total_order_num": ["mean", "min", "max"],
    "total_customer_value": ["mean", "min", "max"],
    "avg_order_value": ["mean", "min", "max"],
    "online_ratio": ["mean", "min", "max"],
    "master_id": "count"
})

# Segmentleri değerli müşterilerden başlayarak sırala
segment_summary2 = segment_summary2.sort_values(
    ("total_customer_value", "mean"), ascending=False
)

print(segment_summary2)

"""
 Kümeleri incelediğimizde en değerli küme / cluster = 2

============================================
  MÜŞTERİ SEGMENT ÖZET TABLOSU (ORTALAMALARLA)
============================================

 Cluster | Müşteri Sayısı | Recency (Aktiflik)   | Tenure (Sadakat)   | Ortalama Toplam Değer | Ortalama Toplam Sipariş | Online Oranı | Genel Yorum
 ------------------------------------------------------------------------------------------------------------------------------------------------------------
    2    |     6226       | 119.90 (Orta)        | 949.86 (~2.6 yıl)  | 1072.20 (Yüksek)       | 7.02                     | 0.77         | VIP, sadık, değerli müşteriler
    0    |    10204       | 89.62 (İyi)          | 690.35 (~1.9 yıl)  | 639.75 (Orta)          | 4.26                     | 0.46         | Orta düzey, potansiyeli yüksek müşteriler
    1    |     3515       | 290.40 (Çok düşük)   | 860.26 (~2.4 yıl)  | 506.41 (Düşük)         | 3.72                     | 0.45         | İnaktif, geri kazanılması gereken müşteriler

"""

# DataFrame'i CSV olarak kaydetme
df.to_csv("musteri_segmentleri.csv", index=False, encoding="utf-8-sig")

print("Dosya başarıyla 'musteri_segmentleri.csv' olarak kaydedildi.")