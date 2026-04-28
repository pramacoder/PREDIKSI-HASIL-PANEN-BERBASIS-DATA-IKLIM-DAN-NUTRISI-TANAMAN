# Hasil EDA Visualisasi - Crop Yield Prediction Dataset

## Ringkasan Eksekusi

Notebook `eda_visualisasi_extended.ipynb` telah berhasil dijalankan. Berikut adalah ringkasan hasilnya:

---

## 1. Dataset Overview

| Atribut | Nilai |
|---------|-------|
| Total Baris | 28.242 |
| Total Kolom | 7 |
| Missing Values | 0 |
| Duplikat | 2.310 |
| Negara Unik | 101 |
| Tanaman Unik | 10 |
| Rentang Tahun | 1990 - 2013 |

---

## 2. Statistik Deskriptif

| Variabel | Mean | Std | Min | 25% | 50% | 75% | Max |
|----------|------|-----|-----|-----|-----|-----|------|
| Yield (hg/ha) | 77.053 | 84.957 | 50 | 19.919 | 38.295 | 104.677 | 501.412 |
| Curah Hujan (mm/thn) | 1.149 | 710 | 51 | 593 | 1.083 | 1.668 | 3.240 |
| Pestisida (ton) | 37.077 | 59.959 | 0,04 | 1.702 | 17.529 | 48.688 | 367.778 |
| Suhu (°C) | 20,54 | 6,31 | 1,30 | 16,70 | 21,51 | 26,00 | 30,65 |

---

## 3. Hasil Korelasi

### A. Numeric-Numeric (Pearson Correlation)

| Variabel | Korelasi dengan Yield | Interpretasi |
|---------|---------------------|-------------|
| average_rain_fall_mm_per_year | r = 0.0010 | Lemah positif |
| pesticides_tonnes | r = 0.0641 | Lemah positif |
| avg_temp | r = -0.1148 | Lemah negatif |

### B. Categorical-Categorical (Cramér's V)

| Variabel | Nilai | Interpretasi |
|---------|-------|--------------|
| Area vs Item | V = 0.299 | Asosiasi sedang |

### C. Categorical-Numerical (Eta Squared η²)

| Variabel | η² | Interpretasi |
|---------|-----|--------------|
| Area | 0.1457 | Kuat |
| Item | 0.6088 | Kuat |

---

## 4. Interpretasi

### Skala Interpretasi:
- **η² > 0.14**: Hubungan Kuat
- **η² 0.06 - 0.14**: Hubungan Moderat
- **η² < 0.06**: Hubungan Lemah
- **Cramér's V → 1**: Asosiasi Kuat
- **Cramér's V → 0**: Asosiasi Lemah

---

## 5. Kesimpulan

1. **Item (jenis tanaman) adalah prediktor terkuat untuk Yield** dengan η² = 0.6088, menunjukkan bahwa jenis tanaman menjelaskan sekitar 60% varians dalam yield.

2. **Area (negara) juga memiliki pengaruh signifikan** dengan η² = 0.1457, termasuk dalam kategori kuat.

3. **Variabel numerik memiliki korelasi lemah dengan Yield**:
   - Curah hujan hampir tidak berkorelasi (r ≈ 0)
   - Pestisida berkorelasi positif lemah (r = 0.064)
   - Suhu berkorelasi negatif lemah (r = -0.115)

4. **Asosiasi antara Area dan Item sedang** (V = 0.299), menunjukkan bahwa kombinasi negara-tanaman tertentu lebih sering muncul.

---

## 6. Visualizations Generated

| File | Deskripsi |
|------|----------|
| gambar1_distribusi_yield.png | Distribusi Yield (histogram + boxplot per tanaman) |
| gambar2_distribusi_fitur.png | Distribusi 4 variabel numerik |
| gambar3_missing_values.png | Persentase missing values |
| gambar4_korelasi_pearson.png | Matriks korelasi Pearson |
| gambar5_scatter.png | Scatter plot fitur vs yield |
| gambar6_crop_trend.png | Rata-rata yield per tanaman + tren tahunan |
| gambar7_korelasi_cramers.png | Matriks Cramér's V |
| gambar8_korelasi_eta.png | Bar chart Eta Squared |
| gambar9_yield_per_crop.png | Proporsi varians yield per tanaman |

Semua gambar tersimpan di folder: `figures/`