import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def p(*parts):
    return os.path.join(ROOT_DIR, *parts)

def main():
    print("Mulai proses training...")
    
    # 1. Buat folder backend/ jika belum ada
    os.makedirs(p('backend'), exist_ok=True)
    
    # 2. Muat dataset mentah untuk merekonstruksi LabelEncoder
    print("Memuat dataset mentah untuk mencocokkan LabelEncoder...")
    df_raw = pd.read_csv(p('dataset', 'yield_df.csv'))
    
    # Standarisasi kolom seperti di preprocessing.ipynb
    df_raw.rename(columns={'hg/ha_yield': 'yield_hg_ha'}, inplace=True)
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')
    df_raw.drop_duplicates(keep='first', inplace=True)
    
    # Fit LabelEncoder
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    
    le_area.fit(df_raw['area'])
    le_item.fit(df_raw['item'])
    
    print(f"LabelEncoder berhasil di-fit:")
    print(f"  Jumlah kelas Area (negara): {len(le_area.classes_)}")
    print(f"  Jumlah kelas Item (tanaman): {len(le_item.classes_)}")
    
    # Simpan LabelEncoder
    joblib.dump(le_area, p('backend', 'le_area.pkl'))
    joblib.dump(le_item, p('backend', 'le_item.pkl'))
    print("LabelEncoder disimpan di backend/le_area.pkl dan backend/le_item.pkl")
    
    # 3. Muat data train hasil preprocessing
    print("Memuat data train...")
    X_train = pd.read_csv(p('output', 'X_train.csv'))
    y_train = pd.read_csv(p('output', 'y_train.csv')).squeeze()
    
    # 4. Latih model RandomForestRegressor
    print("Melatih model RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model berhasil dilatih.")
    
    # 5. Simpan model
    joblib.dump(model, p('backend', 'model.pkl'))
    print("Model berhasil disimpan di backend/model.pkl")
    
    print("Semua proses training selesai dengan sukses!")

if __name__ == '__main__':
    main()
