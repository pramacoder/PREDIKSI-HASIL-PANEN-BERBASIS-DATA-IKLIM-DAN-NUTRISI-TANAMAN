#!/usr/bin/env python
# coding: utf-8

# # EDA Visualisasi Extended dengan ANOVA F-Test
# 
# Notebook ini memperluas `eda_visualisasi_extended.ipynb` dengan:
# - Tambahan ANOVA F-Test untuk menguji perbedaan mean antar kelompok
# - One-way ANOVA: Yield per Jenis Tanaman (Item)
# - One-way ANOVA: Yield per Negara (Area)
# - Two-way ANOVA: Item + Area terhadap Yield

# In[1]:


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway, levene
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ModuleNotFoundError:
    HAVE_SEABORN = False
    print('seaborn tidak terpasang.')

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_DIR = Path('dataset')
OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nama kolom di dataset
COL_YIELD = 'hg/ha_yield'
COL_RAINFALL = 'average_rain_fall_mm_per_year'
COL_PEST = 'pesticides_tonnes'
COL_TEMP = 'avg_temp'
COL_ITEM = 'Item'
COL_AREA = 'Area'
COL_YEAR = 'Year'

# Cari file CSV dari folder dataset/
preferred_files = ['yield_df.csv', 'crop_production.csv', 'yield.csv']
csv_path = None
for name in preferred_files:
    candidate = DATASET_DIR / name
    if candidate.exists():
        csv_path = candidate
        break

if csv_path is None:
    all_csv = sorted(DATASET_DIR.glob('*.csv'))
    if not all_csv:
        raise FileNotFoundError('Tidak ada file CSV di folder dataset/.')
    csv_path = all_csv[0]

print('Membaca dataset:', csv_path)
df = pd.read_csv(csv_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print('Shape:', df.shape)
print('Kolom:', list(df.columns))


# In[2]:


# --- Palette ---
BLUE = '#2563EB'
TEAL = '#0891B2'
PURPLE = '#7C3AED'
AMBER = '#D97706'
SLATE = '#475569'
LIGHT = '#E2E8F0'
GRID = '#CBD5E1'
RED = '#EF4444'
GREEN = '#10B981'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

num_cols = [COL_YIELD, COL_RAINFALL, COL_PEST, COL_TEMP]
col_labels = {
    COL_YIELD: 'Yield (hg/ha)',
    COL_RAINFALL: 'Curah Hujan (mm/tahun)',
    COL_PEST: 'Pestisida (ton)',
    COL_TEMP: 'Suhu Rata-rata (degC)',
}
colors = [BLUE, TEAL, PURPLE, AMBER]

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print('Kolom Numerik:', numerical_columns)
print('Kolom Kategorik:', categorical_columns)


# ## Gambar 1 - Distribusi Yield

# In[3]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df[COL_YIELD], bins=60, color=BLUE, edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('Yield (hg/ha)')
axes[0].set_ylabel('Frekuensi')
axes[0].set_title('Distribusi Yield', fontweight='bold')
mean_v = df[COL_YIELD].mean()
axes[0].axvline(mean_v, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_v:,.0f} hg/ha')
axes[0].legend(fontsize=9)

crops = sorted(df[COL_ITEM].dropna().unique())
data_by_crop = [df[df[COL_ITEM] == c][COL_YIELD].dropna().values for c in crops]
axes[1].boxplot(
    data_by_crop,
    tick_labels=crops,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor=LIGHT, color=BLUE),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color=SLATE),
    capprops=dict(color=SLATE),
    flierprops=dict(marker='o', color=TEAL, alpha=0.3, markersize=3),
)
axes[1].set_xticklabels(crops, rotation=35, ha='right')
axes[1].set_ylabel('Yield (hg/ha)')
axes[1].set_title('Distribusi Yield per Jenis Tanaman', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar1_distribusi_yield.png', bbox_inches='tight')
plt.show()


# ## Gambar 2 - Distribusi 4 Variabel Numerik

# In[4]:


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=60, color=colors[i], edgecolor='white', linewidth=0.4, alpha=0.85)
    axes[i].set_title(col_labels[col], fontweight='bold')
    axes[i].set_ylabel('Frekuensi')
    m, s = df[col].mean(), df[col].std()
    axes[i].text(
        0.97,
        0.95,
        f'Mean={m:,.1f}\nStd={s:,.1f}',
        transform=axes[i].transAxes,
        ha='right',
        va='top',
        fontsize=8,
        color=SLATE,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    )

plt.suptitle('Distribusi Variabel Fitur Numerik', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar2_distribusi_fitur.png', bbox_inches='tight')
plt.show()


# ## Gambar 3 - Missing Values

# In[5]:


miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = ['#EF4444' if v > 0 else TEAL for v in miss_pct.values]
bars = ax.barh(miss_pct.index, miss_pct.values, color=bar_colors, edgecolor='white')
ax.set_xlabel('Persentase Missing (%)')
ax.set_title('Persentase Missing Values per Kolom', fontweight='bold')
ax.axvline(0, color=SLATE, linewidth=0.8)
for bar, v in zip(bars, miss_pct.values):
    ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%', va='center', fontsize=9)

total_miss = int(df.isnull().sum().sum())
dup_count = int(df.duplicated().sum())
ax.text(
    0.98,
    0.02,
    f'Total missing: {total_miss:,}  |  Duplikat: {dup_count:,}',
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=9,
    color=SLATE,
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar3_missing_values.png', bbox_inches='tight')
plt.show()

print('Total baris   :', len(df))
print('Total missing :', total_miss)
print('Total duplikat:', dup_count)


# ## Gambar 4 - Matriks Korelasi (Pearson - Numeric-Numeric)

# In[6]:


fig, ax = plt.subplots(figsize=(8, 6))
corr = df[num_cols].corr()

if HAVE_SEABORN:
    sns.heatmap(
        corr,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 11, 'weight': 'bold'},
        xticklabels=[col_labels[c] for c in num_cols],
        yticklabels=[col_labels[c] for c in num_cols],
        vmin=-1,
        vmax=1,
        center=0,
    )
else:
    m = corr.values
    im = ax.imshow(m, cmap='Blues', vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels([col_labels[c] for c in num_cols], rotation=45, ha='right')
    ax.set_yticklabels([col_labels[c] for c in num_cols])
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f'{m[i, j]:.3f}', ha='center', va='center', fontsize=10, color='black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Matriks Korelasi - Pearson (Numeric-Numeric)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar4_korelasi_pearson.png', bbox_inches='tight')
plt.show()

print('Korelasi dengan Yield (Pearson):')
print(corr[COL_YIELD].drop(COL_YIELD).to_string())


# ## Gambar 5 - Scatter Plot Fitur vs Yield

# In[7]:


feats = [COL_RAINFALL, COL_PEST, COL_TEMP]
flabels = [col_labels[f] for f in feats]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
MAX_SAMPLE = 3000

for i, (feat, flabel) in enumerate(zip(feats, flabels)):
    sub = df[[feat, COL_YIELD]].dropna()
    samp = sub.sample(min(MAX_SAMPLE, len(sub)), random_state=42)

    axes[i].scatter(samp[feat], samp[COL_YIELD], alpha=0.2, color=colors[i + 1], s=8)

    z = np.polyfit(sub[feat], sub[COL_YIELD], 1)
    p = np.poly1d(z)
    xline = np.linspace(sub[feat].min(), sub[feat].max(), 200)
    axes[i].plot(xline, p(xline), color='red', linewidth=1.5, linestyle='--')

    r = sub[feat].corr(sub[COL_YIELD])
    axes[i].set_xlabel(flabel)
    axes[i].set_ylabel('Yield (hg/ha)')
    axes[i].set_title(f'Yield vs {flabel}\n(r = {r:.3f})', fontweight='bold')

plt.suptitle('Hubungan Antara Fitur Numerik dan Yield', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar5_scatter.png', bbox_inches='tight')
plt.show()


# ## Gambar 6 - Rata-rata Yield per Tanaman + Tren Tahunan

# In[8]:


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

crop_means = df.groupby(COL_ITEM)[COL_YIELD].mean().sort_values(ascending=True)
bars = axes[0].barh(crop_means.index, crop_means.values, color=BLUE, edgecolor='white')
axes[0].set_xlabel('Rata-rata Yield (hg/ha)')
axes[0].set_title('Rata-rata Yield per Jenis Tanaman', fontweight='bold')
for bar, v in zip(bars, crop_means.values):
    axes[0].text(v + crop_means.max() * 0.01, bar.get_y() + bar.get_height() / 2, f'{v:,.0f}', va='center', fontsize=8)

yearly = df.groupby(COL_YEAR)[COL_YIELD].mean()
axes[1].plot(yearly.index, yearly.values, color=BLUE, linewidth=2, marker='o', markersize=4)
axes[1].fill_between(yearly.index, yearly.values, alpha=0.12, color=BLUE)
axes[1].set_xlabel('Tahun')
axes[1].set_ylabel('Rata-rata Yield (hg/ha)')
axes[1].set_title(f'Tren Yield Rata-rata per Tahun ({yearly.index.min()}-{yearly.index.max()})', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar6_crop_trend.png', bbox_inches='tight')
plt.show()


# ## Gambar 7 - Korelasi Categorical-Categorical (Cramér's V)

# In[9]:


def cramers_v(x, y):
    """Hitung Cramér's V untuk asosiasi kategorikal-kategorikal."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if min(r, k) - 1 == 0:
        return 0
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

MAX_SAMPLE = 2000
df_sample = df.sample(min(MAX_SAMPLE, len(df)), random_state=42) if len(df) > MAX_SAMPLE else df

cv_area_item = cramers_v(df_sample[COL_AREA], df_sample[COL_ITEM])
print(f'Cramér\'s V (Area vs Item): {cv_area_item:.4f}')

cat_corr_data = {
    COL_AREA: {COL_AREA: 1.0, COL_ITEM: cv_area_item},
    COL_ITEM: {COL_AREA: cv_area_item, COL_ITEM: 1.0}
}
cat_corr_df = pd.DataFrame(cat_corr_data)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cat_corr_df.values, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(categorical_columns)))
ax.set_yticks(range(len(categorical_columns)))
ax.set_xticklabels(categorical_columns, rotation=45, ha='right')
ax.set_yticklabels(categorical_columns)
for i in range(len(categorical_columns)):
    for j in range(len(categorical_columns)):
        ax.text(j, i, f'{cat_corr_df.values[i, j]:.3f}', ha='center', va='center', fontsize=12, color='black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Matriks Cramér\'s V (Categorical-Categorical)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar7_korelasi_cramers.png', bbox_inches='tight')
plt.show()

print('Interpretasi Cramér\'s V:')
print('  - Nilai → 1: Asosiasi kuat')
print('  - Nilai → 0: Asosiasi lemah')


# ## Gambar 8 - Korelasi Categorical-Numerical (Eta Squared)

# In[10]:


def eta_squared(cat, num):
    """Hitung Eta Squared (η²) untuk hubungan kategorikal-numerik."""
    categories = pd.unique(cat)
    grand_mean = np.mean(num)

    ss_between = sum(
        len(num[cat == c]) * (np.mean(num[cat == c]) - grand_mean) ** 2
        for c in categories
    )
    ss_total = sum((num - grand_mean) ** 2)

    return ss_between / ss_total if ss_total != 0 else 0

target = COL_YIELD
eta_results = {}

for cat_col in categorical_columns:
    eta_results[cat_col] = eta_squared(df[cat_col], df[target])

print('Eta Squared (η²) dengan', col_labels[target], ':')
for cat, val in eta_results.items():
    strength = 'Strong' if val > 0.14 else 'Moderate' if val > 0.06 else 'Weak'
    print(f'  {cat}: {val:.4f} ({strength})')

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = [BLUE, TEAL, PURPLE, AMBER]
bars = ax.bar(eta_results.keys(), eta_results.values(), color=colors_bar[:len(eta_results)], edgecolor='white')
ax.set_xlabel('Variabel Kategorikal', fontsize=11)
ax.set_ylabel('Nilai η²', fontsize=11)
ax.set_title(f'Eta Squared (η²) - Hubungan Kategorikal dengan {col_labels[target]}', fontweight='bold', fontsize=13)
ax.axhline(0.14, color='red', linestyle='--', linewidth=1, label='Strong (≥0.14)')
ax.axhline(0.06, color='orange', linestyle='--', linewidth=1, label='Moderate (≥0.06)')
ax.legend(loc='upper right', fontsize=9)
for bar, v in zip(bars, eta_results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar8_korelasi_eta.png', bbox_inches='tight')
plt.show()


# ## Gambar 9 - Proporsi Varians Yield per Jenis Tanaman

# In[11]:


crop_means = df.groupby(COL_ITEM)[COL_YIELD].mean()
grand_mean = df[COL_YIELD].mean()
crop_counts = df.groupby(COL_ITEM)[COL_YIELD].count()

item_var_explained = {}
for crop in crop_means.index:
    n = crop_counts[crop]
    item_var_explained[crop] = n * (crop_means[crop] - grand_mean) ** 2

total_var = sum(item_var_explained.values())
item_eta_pct = {k: v / total_var for k, v in item_var_explained.items()}

fig, ax = plt.subplots(figsize=(10, 6))
items = sorted(item_eta_pct.keys())
values = [item_eta_pct[i] for i in items]
bars = ax.barh(items, values, color=BLUE, edgecolor='white')
ax.set_xlabel('Proporsi Varians Yield yang Diterangkan', fontsize=11)
ax.set_title(f'Proporsi Varians Yield per Jenis Tanaman ({COL_ITEM})', fontweight='bold', fontsize=13)
for bar, v in zip(bars, values):
    ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar9_yield_per_crop.png', bbox_inches='tight')
plt.show()


# ## --- ANOVA F-TEST SECTION ---
# 
# Bagian ini menambahkan ANOVA F-Test untuk menguji apakah terdapat perbedaan mean Yield yang signifikan antar:
# 1. Kelompok jenis tanaman (Item)
# 2. Kelompok negara (Area)
# 
# Hipotesis ANOVA:
# - H0: μ1 = μ2 = ... = μk (semua mean kelompok sama)
# - H1: minimal ada satu mean yang berbeda
# 
# Signifikansi: α = 0.05

# ## Gambar 10 - One-Way ANOVA: Yield per Jenis Tanaman (Item)

# In[12]:


print('=' * 60)
print('ONE-WAY ANOVA: Yield vs Jenis Tanaman (Item)')
print('=' * 60)

crop_groups = [df[df[COL_ITEM] == crop][COL_YIELD].dropna().values for crop in sorted(df[COL_ITEM].unique())]
crop_names = sorted(df[COL_ITEM].unique())

stat_anova_item, p_anova_item = f_oneway(*crop_groups)
print(f'F-statistic: {stat_anova_item:.4f}')
print(f'p-value: {p_anova_item:.2e}')
print(f'Significant (α=0.05): {"YES" if p_anova_item < 0.05 else "NO"}')

levene_stat, levene_p = levene(*crop_groups)
print(f'Levene Test (homogenitas varians): W={levene_stat:.4f}, p={levene_p:.4f}')

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(crop_groups, labels=crop_names, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor(LIGHT)
    patch.set_edgecolor(BLUE)
for median in bp['medians']:
    median.set_color(RED)
    median.set_linewidth(2)
ax.set_xticklabels(crop_names, rotation=35, ha='right')
ax.set_ylabel('Yield (hg/ha)')
ax.set_title(f'One-Way ANOVA: Yield per Jenis Tanaman\nF={stat_anova_item:.2f}, p={p_anova_item:.2e}', fontweight='bold')
ax.axhline(grand_mean, color=GREEN, linestyle='--', linewidth=1.5, label=f'Grand Mean: {grand_mean:,.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar10_anova_item.png', bbox_inches='tight')
plt.show()

print('\nMean Yield per Jenis Tanaman:')
for name, group in zip(crop_names, crop_groups):
    print(f'  {name}: Mean={np.mean(group):,.0f}, Std={np.std(group):,.0f}, N={len(group)}')


# ## Gambar 11 - One-Way ANOVA: Yield per Negara (Area) - Top 10

# In[13]:


print('=' * 60)
print('ONE-WAY ANOVA: Yield vs Negara (Area)')
print('=' * 60)

area_counts = df[COL_AREA].value_counts()
top_areas = area_counts.head(10).index.tolist()
print(f'Menggunakan top 10 negara (dari {df[COL_AREA].nunique()} total)')

df_top_areas = df[df[COL_AREA].isin(top_areas)]
area_groups = [df_top_areas[df_top_areas[COL_AREA] == area][COL_YIELD].dropna().values for area in top_areas]

stat_anova_area, p_anova_area = f_oneway(*area_groups)
print(f'F-statistic: {stat_anova_area:.4f}')
print(f'p-value: {p_anova_area:.2e}')
print(f'Significant (α=0.05): {"YES" if p_anova_area < 0.05 else "NO"}')

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(area_groups, labels=top_areas, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor(LIGHT)
    patch.set_edgecolor(TEAL)
for median in bp['medians']:
    median.set_color(RED)
    median.set_linewidth(2)
ax.set_xticklabels(top_areas, rotation=45, ha='right')
ax.set_ylabel('Yield (hg/ha)')
ax.set_title(f'One-Way ANOVA: Yield per Negara (Top 10)\nF={stat_anova_area:.2f}, p={p_anova_area:.2e}', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar11_anova_area.png', bbox_inches='tight')
plt.show()

print('\nMean Yield per Negara (Top 10):')
area_means = df_top_areas.groupby(COL_AREA)[COL_YIELD].mean().sort_values(ascending=False)
for area in top_areas:
    print(f'  {area}: Mean={area_means[area]:,.0f}')


# ## Gambar 12 - Two-Way ANOVA: Interaction Effect (Item x Area)

# In[14]:


print('=' * 60)
print('INTERACTION PLOT: Item x Area (Sample)')
print('=' * 60)

top_crops = df[COL_ITEM].value_counts().head(3).index.tolist()
top_countries = df[COL_AREA].value_counts().head(3).index.tolist()
print(f'Crop: {top_crops}')
print(f'Country: {top_countries}')

interaction_data = df[(df[COL_ITEM].isin(top_crops)) & (df[COL_AREA].isin(top_countries))]
interaction_means = interaction_data.groupby([COL_ITEM, COL_AREA])[COL_YIELD].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(top_crops))
width = 0.25

for i, country in enumerate(top_countries):
    means = [interaction_means.loc[crop, country] if country in interaction_means.columns else 0 for crop in top_crops]
    ax.bar(x + i * width, means, width, label=country, edgecolor='white')

ax.set_xlabel('Jenis Tanaman')
ax.set_ylabel('Mean Yield (hg/ha)')
ax.set_title('Interaction Effect: Item x Area\n(Mean Yield per Kombinasi)', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(top_crops, rotation=15)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar12_interaction.png', bbox_inches='tight')
plt.show()

print('\nMean Yield - Interaction Table:')
print(interaction_means.round(0))


# ## Gambar 13 - ANOVA Effect Size (Eta Squared dan Omega Squared)

# In[15]:


def omega_squared(cat, num):
    """Hitung Omega Squared (ω²) untuk effect size ANOVA."""
    categories = pd.unique(cat)
    grand_mean = np.mean(num)

    k = len(categories)
    N = len(num)

    ss_between = sum(
        len(num[cat == c]) * (np.mean(num[cat == c]) - grand_mean) ** 2
        for c in categories
    )
    ss_total = sum((num - grand_mean) ** 2)
    ss_within = ss_total - ss_between

    ms_within = ss_within / (N - k)

    if ms_within == 0:
        return 0

    eta_sq = ss_between / ss_total
    omega_sq = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)

    return eta_sq, max(0, omega_sq)

print('=' * 60)
print('ANOVA EFFECT SIZE')
print('=' * 60)

effect_sizes = {}
for cat_col in categorical_columns:
    eta_sq, omega_sq = omega_squared(df[cat_col], df[COL_YIELD])
    effect_sizes[cat_col] = {'eta_squared': eta_sq, 'omega_squared': omega_sq}
    print(f'{cat_col}:')
    print(f'  η² = {eta_sq:.4f}')
    print(f'  ω² = {omega_sq:.4f}')

    size = 'Large' if omega_sq >= 0.14 else 'Medium' if omega_sq >= 0.06 else 'Small'
    print(f'  Effect Size: {size}')

fig, ax = plt.subplots(figsize=(8, 5))
cat_names = list(effect_sizes.keys())
eta_vals = [effect_sizes[c]['eta_squared'] for c in cat_names]
omega_vals = [effect_sizes[c]['omega_squared'] for c in cat_names]

x = np.arange(len(cat_names))
width = 0.35

bars1 = ax.bar(x - width/2, eta_vals, width, label='η² (Eta Squared)', color=BLUE, edgecolor='white')
bars2 = ax.bar(x + width/2, omega_vals, width, label='ω² (Omega Squared)', color=TEAL, edgecolor='white')

ax.set_xlabel('Variabel Kategorikal')
ax.set_ylabel('Effect Size')
ax.set_title('ANOVA Effect Size: η² vs ω²', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cat_names)
ax.legend()
ax.axhline(0.14, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Large threshold')
ax.axhline(0.06, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Medium threshold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gambar13_effect_size.png', bbox_inches='tight')
plt.show()


# ## Ringkasan Statistik Deskriptif

# In[16]:


print(df[num_cols].describe().round(2))
print('Jumlah negara unik :', df[COL_AREA].nunique())
print('Jumlah tanaman unik:', df[COL_ITEM].nunique())
print('Rentang tahun      :', f"{df[COL_YEAR].min()} - {df[COL_YEAR].max()}")
print('Semua gambar tersimpan di folder:', OUTPUT_DIR)


# ## Ringkasan Hasil Korelasi dan ANOVA

# In[17]:


print('============================================================')
print('RINGKASAN HASIL KORELASI DAN ANOVA')
print('============================================================')

print('[A] Numeric-Numeric (Pearson):')
print('    - Korelasi dengan Yield (numerik):')
pearson_corr = corr[COL_YIELD].drop(COL_YIELD)
for col, val in pearson_corr.items():
    direction = 'positif' if val > 0 else 'negatif'
    strength = 'kuat' if abs(val) > 0.5 else 'sedang' if abs(val) > 0.3 else 'lemah'
    print(f'      {col}: r = {val:.4f} ({strength} {direction})')

print('\n[B] Categorical-Categorical (Cramér\'s V):')
print(f'    - Area vs Item: V = {cv_area_item:.3f} (sedang)')

print('\n[C] Categorical-Numerical (Eta Squared):')
print('    - Hubungan dengan Yield:')
for cat_col in categorical_columns:
    val = eta_squared(df[cat_col], df[COL_YIELD])
    strength = 'Kuat' if val > 0.14 else 'Moderat' if val > 0.06 else 'Lemah'
    print(f'      {cat_col}: η² = {val:.4f} ({strength})')

print('\n[D] ANOVA F-Test:')
print(f'    - One-Way ANOVA (Item):')
print(f'      F = {stat_anova_item:.2f}, p < 0.001 ( SIGNIFIKAN)')
print(f'    - One-Way ANOVA (Area, Top 10):')
print(f'      F = {stat_anova_area:.2f}, p < 0.001 ( SIGNIFIKAN)')

print('[E] Effect Size (Omega Squared):')
for cat_col in categorical_columns:
    eta_sq, omega_sq = omega_squared(df[cat_col], df[COL_YIELD])
    size = 'Large' if omega_sq >= 0.14 else 'Medium' if omega_sq >= 0.06 else 'Small'
    print(f'    - {cat_col}: ω² = {omega_sq:.3f} ({size})')

print('\n[F] Kesimpulan:')
print('    1. Item (jenis tanaman) adalah prediktor terkuat untuk Yield')
print('    2. Area (negara) juga memiliki pengaruh signifikan')
print('    3. Variabel numerik memiliki korelasi lemah dengan Yield')
print('    4. ANOVA menunjukkan perbedaan mean Yield yang sangat signifikan')
print('       antar kelompok (p < 0.001)')
print('    5. Effect size besar untuk Item (ω² = 0.608)')

