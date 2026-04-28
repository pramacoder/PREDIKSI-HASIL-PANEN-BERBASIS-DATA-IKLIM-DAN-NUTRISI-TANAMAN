"""
EDA: Analisis Input Features untuk Sistem Prediksi
Versi Python - Output single string untuk AI consumption
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "dataset"
OUTPUT_DIR = SCRIPT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COL_YIELD = "hg/ha_yield"
COL_RAINFALL = "average_rain_fall_mm_per_year"
COL_PEST = "pesticides_tonnes"
COL_TEMP = "avg_temp"
COL_ITEM = "Item"
COL_AREA = "Area"
COL_YEAR = "Year"

preferred_files = ["yield_df.csv", "crop_production.csv", "yield.csv"]
csv_path = None
for name in preferred_files:
    candidate = DATASET_DIR / name
    if candidate.exists():
        csv_path = candidate
        break

if csv_path is None:
    all_csv = sorted(DATASET_DIR.glob("*.csv"))
    if not all_csv:
        raise FileNotFoundError("Tidak ada file CSV ditemukan di folder dataset/")
    csv_path = all_csv[0]

df = pd.read_csv(csv_path)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

col_labels = {
    COL_YIELD: "Yield",
    COL_RAINFALL: "Curah Hujan",
    COL_PEST: "Pestisida",
    COL_TEMP: "Suhu",
    COL_ITEM: "Jenis Tanaman",
    COL_AREA: "Negara",
}


def detect_outliers_iqr(data, multiplier=1.5):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (data < lower) | (data > upper), lower, upper


def detect_outliers_zscore(data, threshold=3):
    z = np.abs(stats.zscore(data))
    return z > threshold


output = []

output.append("=" * 70)
output.append("EDA: ANALISIS INPUT FEATURES UNTUK SISTEM PREDIKSI")
output.append("=" * 70)
output.append(f"\nDataset: {csv_path.name}")
output.append(f"Shape: {df.shape}")
output.append(f"Kolom: {list(df.columns)}")

# BAGIAN 1: OUTLIER
output.append("\n" + "=" * 70)
output.append("BAGIAN 1: ANALISIS OUTLIER")
output.append("=" * 70)

for col in [COL_YIELD, COL_RAINFALL, COL_PEST, COL_TEMP]:
    data = df[col].dropna()
    outlier_iqr, lower_iqr, upper_iqr = detect_outliers_iqr(data)
    output.append(f"\n{col_labels[col]} ({col}):")
    output.append(f"  Range: {data.min():.2f} - {data.max():.2f}")
    output.append(f"  IQR bounds: [{lower_iqr:.2f}, {upper_iqr:.2f}]")
    output.append(
        f"  Outliers (IQR): {outlier_iqr.sum()} ({outlier_iqr.sum() / len(data) * 100:.2f}%)"
    )

output.append("\n" + "-" * 40)
output.append("OUTLIER PER JENIS TANAMAN (Yield):")
for item in sorted(df[COL_ITEM].unique()):
    item_data = df[df[COL_ITEM] == item][COL_YIELD].dropna()
    outlier_iqr, _, _ = detect_outliers_iqr(item_data)
    pct = outlier_iqr.sum() / len(item_data) * 100
    output.append(f"  {item}: {outlier_iqr.sum()}/{len(item_data)} ({pct:.1f}%)")

extreme_yield = df[df[COL_YIELD] > 400000][[COL_ITEM, COL_AREA, COL_YEAR, COL_YIELD]]
output.append(f"\nOUTLIER EXTREME (Yield > 400,000): {len(extreme_yield)} records")
if len(extreme_yield) > 0:
    for _, row in extreme_yield.iterrows():
        output.append(
            f"  {row[COL_ITEM]} di {row[COL_AREA]} ({row[COL_YEAR]}): {row[COL_YIELD]:,.0f} hg/ha"
        )

output.append("\n" + "-" * 40)
output.append("REKOMENDASI PENANGANAN OUTLIER:")
output.append(
    "  1. Outlier Yield (Kentang >400k): JANGAN dihapus, gunakan log-transform"
)
output.append("  2. Outlier Pesticides: Log-transform atau standardization")
output.append("  3. Outlier Suhu/Hujan: Sangat kecil, bisa di-cap jika tidak valid")

# BAGIAN 2: VALIDASI INPUT USER
output.append("\n" + "=" * 70)
output.append("BAGIAN 2: VALIDASI INPUT USER")
output.append("=" * 70)

output.append("\n[A] KOLOM KATEGORIKAL (User pilih dari dropdown):")
output.append(f"  - Area: {df[COL_AREA].nunique()} unique negara")
output.append(f"  - Item: {df[COL_ITEM].nunique()} unique tanaman")

output.append("\n[B] KOLOM NUMERIK (Range dalam dataset):")
for col in [COL_RAINFALL, COL_TEMP, COL_PEST]:
    output.append(
        f"  - {col_labels[col]} ({col}): Min={df[col].min():.1f}, Max={df[col].max():.1f}, Mean={df[col].mean():.1f}"
    )

valid_ranges = {
    COL_RAINFALL: {"min": 0, "max": 3500},
    COL_TEMP: {"min": -10, "max": 45},
    COL_PEST: {"min": 0, "max": 500000},
}

output.append("\n[C] RANGE INPUT YANG DISARANKAN UNTUK APP:")
for col, info in valid_ranges.items():
    data_range = df[col]
    outside = ((data_range < info["min"]) | (data_range > info["max"])).sum()
    output.append(
        f"  - {col_labels[col]}: {info['min']} - {info['max']} (data di luar range: {outside})"
    )

output.append("\n[D] SARAN VALIDASI INPUT DI APLIKASI:")
output.append("  - Tampilkan error jika user input di luar range")
output.append("  - Sediakan tooltip dengan range yang valid")

# BAGIAN 3: STRATEGI ENCODING
output.append("\n" + "=" * 70)
output.append("BAGIAN 3: STRATEGI ENCODING")
output.append("=" * 70)

for col in [COL_AREA, COL_ITEM]:
    n_unique = df[col].nunique()
    value_counts = df[col].value_counts()
    output.append(f"\n{col_labels[col]} ({col}): {n_unique} unique")
    output.append(f"  Min frequency: {value_counts.min()}")
    output.append(f"  Max frequency: {value_counts.max()}")

area_means = df.groupby(COL_AREA)[COL_YIELD].mean().sort_values(ascending=False)
item_means = df.groupby(COL_ITEM)[COL_YIELD].mean().sort_values(ascending=False)

output.append("\n" + "-" * 40)
output.append("STRATEGI ENCODING YANG DISARANKAN:")
output.append("\n[A] AREA (101 unique):")
output.append("  - Cardinality: TINGGI (101 unique)")
output.append("  - SOLUSI: Target Encoding (mean yield per negara)")
output.append("  - Handle unknown: Gunakan global mean untuk negara baru")
output.append("\n  Contoh Target Encoding (Top 5):")
for area in area_means.head(5).index:
    output.append(f"    {area}: {area_means[area]:,.0f} hg/ha")

output.append("\n[B] ITEM (10 unique):")
output.append("  - Cardinality: RENDAH (10 unique)")
output.append("  - SOLUSI: One-Hot Encoding (sederhana dan efektif)")

# BAGIAN 4: TRAIN-TEST SPLIT
output.append("\n" + "=" * 70)
output.append("BAGIAN 4: TRAIN-TEST SPLIT CONSIDERATION")
output.append("=" * 70)

years = sorted(df[COL_YEAR].unique())
output.append(f"\nRentang tahun: {min(years)} - {max(years)} ({len(years)} tahun)")

output.append("\n" + "-" * 40)
output.append("STRATEGI YANG MUNGKIN:")
output.append("\n[1] RANDOM SPLIT (70:30):")
output.append("  PRO: Menggunakan semua data")
output.append("  CONS: DATA LEAKAGE - tahun di test bisa muncul di train")

output.append("\n[2] TIME-BASED SPLIT (REKOMENDASI):")
output.append("  Train: tahun lama, Test: tahun terbaru")
output.append("  PRO: Tidak ada data leakage temporal")
output.append("  CONS: Model tidak lihat tren terbaru")

split_year = 2010
df_train = df[df[COL_YEAR] <= split_year]
df_test = df[df[COL_YEAR] > split_year]

output.append(
    f"\nContoh Time-Based Split (Train <= {split_year}, Test > {split_year}):"
)
output.append(f"  Train: {len(df_train):,} records (tahun 1990-{split_year})")
output.append(f"  Test: {len(df_test):,} records (tahun {split_year + 1}-2013)")
output.append(f"  Train yield mean: {df_train[COL_YIELD].mean():,.0f}")
output.append(f"  Test yield mean: {df_test[COL_YIELD].mean():,.0f}")

# BAGIAN 5: ANALISIS FITUR YEAR
output.append("\n" + "=" * 70)
output.append("BAGIAN 5: ANALISIS FITUR YEAR")
output.append("=" * 70)

correlation = df[COL_YEAR].corr(df[COL_YIELD])
spearman = df[COL_YEAR].corr(df[COL_YIELD], method="spearman")

output.append(f"\nKorelasi Year vs Yield:")
output.append(f"  Pearson: r = {correlation:.4f}")
output.append(f"  Spearman: rho = {spearman:.4f}")

output.append("\n[OPSI A] TIDAK MASUKKAN YEAR SEBAGAI FITUR:")
output.append("  - Alasan: User tidak tahu tahun depan seperti apa")
output.append("  - Gunakan kondisi lingkungan (suhu, hujan, pestisida)")
output.append("  - Ini lebih masuk akal untuk sistem rekomendasi")

output.append("\n[OPSI B] MASUKKAN YEAR SEBAGAI FITUR:")
output.append("  - Alasan: Ada tren temporal yang mungkin berguna")
output.append("  - Masalah: Model tidak bisa prediksi untuk tahun > 2013")

output.append("\n[KESIMPULAN]:")
output.append(f"  - Korelasi Year vs Yield lemah (r = {correlation:.4f})")
output.append("  - REKOMENDASI: JANGAN gunakan Year sebagai fitur input")
output.append("  - Gunakan fitur lingkungan (suhu, hujan, pestisida) saja")

# RINGKASAN
output.append("\n" + "=" * 70)
output.append("RINGKASAN FINAL - FEATURE SET UNTUK MODEL")
output.append("=" * 70)

output.append("\n[INPUT DARI USER]:")
output.append("  - Area (kategori) -> Target Encoding")
output.append("  - Item (kategori) -> One-Hot Encoding")
output.append("  - average_rain_fall_mm_per_year (numerik)")
output.append("  - avg_temp (numerik)")
output.append("  - pesticides_tonnes (numerik)")

output.append("\n[TARGET]:")
output.append("  - hg/ha_yield")

output.append("\n[TIDAK MASUKKAN]:")
output.append("  - Year (tidak masuk akal untuk sistem prediksi)")

output.append("\n" + "=" * 70)
output.append("END OF ANALYSIS")
output.append("=" * 70)

print("\n".join(output))
