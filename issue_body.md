## 📋 Perencanaan Analisis Data dan Pemodelan Random Forest

Proyek ini bertujuan membangun pipeline analisis data lengkap mulai dari eksplorasi dataset hasil panen, pembersihan data, analisis eksploratif (EDA), hingga pemodelan prediktif menggunakan **Random Forest Regressor**. Target prediksi adalah `hg/ha_yield` (hasil panen), dengan fitur input mencakup variabel iklim dan pertanian.

---

## 🎯 Tujuan Proyek

| No | Tujuan |
|----|--------|
| 1 | Memahami struktur dan kualitas dataset hasil panen |
| 2 | Melakukan analisis eksploratif untuk menemukan pola penting pada data |
| 3 | Membersihkan data agar siap dipakai untuk pemodelan |
| 4 | Membangun model Random Forest Regressor untuk memprediksi yield |
| 5 | Mengevaluasi performa model dan menarik insight yang relevan |

---

## ❓ Rumusan Masalah

1. Faktor apa yang paling berpengaruh terhadap hasil panen?
2. Apakah variabel iklim dan pertanian memiliki hubungan yang cukup kuat terhadap yield?
3. Seberapa baik Random Forest mampu memprediksi hasil panen?
4. Fitur mana yang paling dominan dalam model?

---

## 🗂️ Scope Pekerjaan

**Yang Dikerjakan:**
- Pengumpulan dan pemahaman dataset
- Data cleaning
- Exploratory Data Analysis (EDA)
- Encoding variabel kategorikal
- Training dan evaluasi model Random Forest
- Interpretasi hasil model
- Penyusunan laporan/presentasi

**Yang Tidak Difokuskan:**
- Deployment aplikasi
- Perbandingan banyak algoritma sekaligus
- Integrasi real-time data pertanian

---

## 🔄 Alur Kerja Proyek (7 Tahap)

### Tahap 1 — Pemahaman Dataset
- Pelajari arti setiap kolom
- Tentukan target (`hg/ha_yield`) dan fitur input
- Cek ukuran data, tipe data, cakupan tahun, negara, dan komoditas

### Tahap 2 — EDA
- Distribusi target yield
- Distribusi curah hujan, pestisida, suhu rata-rata
- Boxplot yield per jenis tanaman
- Tren yield per tahun
- Heatmap korelasi antar fitur numerik
- Scatter plot fitur numerik vs yield

---

### Tahap 3 — Data Preprocessing

> **Temuan EDA:** dataset memiliki **28.242 baris**, **missing value = 0** (terkonfirmasi), dan **2.310 data duplikat**

#### 3.1 — Cek Missing Value
- **Hasil EDA**: `Total missing: 0` — semua kolom bersih
- Tetap konfirmasi ulang dengan `df.isnull().sum()`
- Dokumentasikan: _"Tidak ditemukan missing value pada seluruh 7 kolom dataset"_

#### 3.2 — Deteksi dan Tangani Data Duplikat
- Ditemukan **2.310 baris duplikat** (≈8,2%)
- Hapus dengan `df.drop_duplicates(keep='first', inplace=True)`

| Kondisi | Jumlah Baris |
|---------|-------------|
| Sebelum `drop_duplicates` | **28.242** baris |
| Setelah `drop_duplicates` | **±25.932** baris |

#### 3.3 — Deteksi dan Tangani Outlier (Metode IQR)

| Kolom | Distribusi (EDA) | Strategi |
|-------|-----------------|----------|
| `hg/ha_yield` | Right-skewed | **Tetap dipertahankan** — yield ekstrem valid per jenis tanaman |
| `pesticides_tonnes` | **Sangat** right-skewed | **Capping** IQR — distribusi ekstrem berpotensi distorsi |
| `average_rain_fall_mm_per_year` | Right-skewed | **Capping** IQR — nilai >2.500mm tidak representatif |
| `avg_temp` | Mendekati normal | **Tetap dipertahankan** — suhu ekstrem valid geografis |

> **Capping (Winsorizing)** dipilih dibanding hapus baris agar jumlah data tidak berkurang lagi.

#### 3.4 — Konsistensi Nama Kolom
- Rename `hg/ha_yield` → `yield_hg_ha` (karakter `/` bermasalah)
- Standarisasi: `df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')`

#### 3.5 — Pisahkan Fitur Numerik dan Kategorikal

| Tipe | Kolom |
|------|-------|
| Numerik | `year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp` |
| Kategorikal | `area`, `item` |
| Target | `yield_hg_ha` |

---

### Tahap 4 — Persiapan Model

#### 4.1 — Encoding (DIPERLUKAN)
**Metode: Label Encoding** untuk `area` dan `item`

| Metode | Keputusan | Alasan |
|--------|-----------|--------|
| **Label Encoding** | ✅ Digunakan | Efisien, tidak menambah dimensi |
| One-Hot Encoding | ⚠️ Hindari | Area punya banyak negara unik → dimensionality explosion |

#### 4.2 — Scaling (TIDAK DIPERLUKAN)
> **Min-Max Scaler / Standard Scaler TIDAK DIPERLUKAN** — Random Forest bekerja berdasarkan threshold pada split, bukan jarak antar data.

#### 4.3 — SMOTE (TIDAK RELEVAN)
> **SMOTE TIDAK RELEVAN** — proyek ini adalah **regresi**, bukan klasifikasi. Tidak ada konsep class imbalance.

#### 4.4 — Train-Test Split (80:20)

| Split | Proporsi | Jumlah |
|-------|----------|--------|
| Train | 80% | ~22.593 baris |
| Test  | 20% | ~5.649 baris |

---

### Tahap 5 — Modelling

Algoritma: **Random Forest Regressor**

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### Tahap 6 — Evaluasi

| Metrik | Keterangan |
|--------|------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Koefisien Determinasi |

Analisis lanjutan:
- Bandingkan skor train vs test (deteksi overfitting)
- Feature Importance — fitur dominan apa?

### Tahap 7 — Insight dan Kesimpulan
- Faktor dominan berdasarkan feature importance
- Interpretasi dalam konteks pertanian nyata
- Simpulan dan saran pengembangan

---

## 📅 Timeline Pengerjaan (7 Hari)

| Hari | Aktivitas |
|------|-----------|
| Hari 1 | Load dataset, cek tipe data, missing, duplikat |
| Hari 2 | Data cleaning, EDA distribusi awal |
| Hari 3 | Analisis korelasi, tren per tanaman & tahun |
| Hari 4 | Encoding, split data, training RF awal |
| Hari 5 | Evaluasi & tuning parameter, feature importance |
| Hari 6 | Susun hasil, interpretasi, kesimpulan |
| Hari 7 | Finalisasi laporan & slide presentasi |

---

## 👥 Pembagian Tugas Kelompok

| Anggota | Tanggung Jawab |
|---------|----------------|
| **Anggota 1** | Data Understanding & Preprocessing |
| **Anggota 2** | EDA & Visualisasi |
| **Anggota 3** | Modelling & Encoding |
| **Anggota 4** | Evaluasi & Dokumentasi |

---

## 🛠️ Tools

| Library | Fungsi |
|---------|--------|
| `Pandas` | Olah data tabular |
| `NumPy` | Operasi numerik |
| `Matplotlib` / `Seaborn` | Visualisasi |
| `Scikit-learn` | Preprocessing, RF, evaluasi |
| Google Colab / Jupyter | Lingkungan pengerjaan |

---

## ✅ Checklist Output

- [ ] Notebook analisis lengkap (EDA + Preprocessing + Modelling)
- [ ] Dataset bersih (`yield_clean.csv`)
- [ ] Visualisasi EDA (distribusi, boxplot, heatmap, scatter)
- [ ] Model Random Forest terlatih
- [ ] Laporan evaluasi (MAE, RMSE, R²)
- [ ] Visualisasi feature importance
- [ ] Laporan akhir
- [ ] Slide presentasi (PPT)

---

## ⚠️ Risiko dan Antisipasi

| Risiko | Antisipasi |
|--------|------------|
| 2.310 data duplikat | Bersihkan sejak awal, dokumentasikan sebelum & sesudah |
| Fitur kategorikal banyak | Label Encoding (bukan OHE) |
| Model overfitting | Cross-validation, bandingkan train vs test score |
| Interpretasi lemah | Fokus feature importance & konteks domain |

---

## 📌 Catatan Penting untuk Dosen

> Karena dataset tercatat **missing value = 0** namun ditemukan **2.310 data duplikat**, bagian preprocessing perlu ditonjolkan secara eksplisit dalam laporan. Tunjukkan jumlah data sebelum dan sesudah cleaning — ini menjadi indikator kualitas analisis yang dinilai tinggi.
