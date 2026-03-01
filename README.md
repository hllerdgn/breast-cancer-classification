# 🔬 Meme Kanseri Teşhisinde Random Forest ile İkili Sınıflandırma

Bu çalışmada, Breast Cancer Wisconsin veri seti kullanılarak iyi huylu (benign) ve kötü huylu (malignant) tümörleri birbirinden ayırt eden bir makine öğrenmesi modeli geliştirilmiştir. Sınıflandırma işlemi için **Random Forest (Rastgele Orman)** algoritması tercih edilmiştir.

---


```
breast-cancer-classification/
│
├── breastcancerclassification.py   # Ana model kodu
├── data.csv                        # Breast Cancer Wisconsin veri seti
└── README.md
```

---

## 📊 Veri Seti

- **Kaynak:** Breast Cancer Wisconsin Dataset
- **Toplam Örnek:** 569
- **Öznitelik Sayısı:** 30
- **Sınıflar:** Benign (İyi Huylu) — 357 | Malignant (Kötü Huylu) — 212

Her öznitelik; ortalama (mean), standart hata (se) ve en kötü değer (worst) olmak üzere üç farklı istatistiksel ölçüm biçiminde hesaplanmıştır.

---

## ⚙️ Kullanılan Teknolojiler

| Kütüphane      | Kullanım Amacı                 |
| -------------- | ------------------------------ |
| `pandas`       | Veri okuma ve işleme           |
| `numpy`        | Sayısal hesaplamalar           |
| `scikit-learn` | Model eğitimi ve değerlendirme |

---

## 🔄 İşlem Adımları

1. `id` sütunu veri setinden çıkarıldı
2. `diagnosis` sütunu LabelEncoder ile sayısal değere dönüştürüldü (B=0, M=1)
3. Veri %80 eğitim / %20 test olarak ayrıldı (`stratify=y`)
4. Random Forest modeli `n_estimators=100` ile eğitildi
5. Model performansı 4 metrik ile değerlendirildi

---

## 📈 Model Performansı

| Metrik               | Değer  |
| -------------------- | ------ |
| Accuracy (Doğruluk)  | %96,49 |
| Precision (Kesinlik) | %100   |
| Recall (Duyarlılık)  | %90.47 |
| F1 Score             | %95    |

---

## 🚀 Çalıştırma

```bash
# Gerekli kütüphaneleri yükle
pip install pandas numpy scikit-learn

# Modeli çalıştır
python breastcancerclassification.py
```
