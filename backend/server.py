import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Mengizinkan request dari frontend Next.js

# Load model dan encoders
model_path = os.path.join('backend', 'model.pkl')
knn_model_path = os.path.join('backend', 'knn_pipeline_model.pkl')
le_area_path = os.path.join('backend', 'le_area.pkl')
le_item_path = os.path.join('backend', 'le_item.pkl')

if not (os.path.exists(model_path) and os.path.exists(le_area_path) and os.path.exists(le_item_path)):
    raise RuntimeError("Model dan label encoder belum dilatih! Jalankan train.py terlebih dahulu.")

if not os.path.exists(knn_model_path):
    raise RuntimeError("Model KNN pipeline tidak ditemukan! Pastikan knn_pipeline_model.pkl ada di folder backend.")

model_rf = joblib.load(model_path)
model_knn = joblib.load(knn_model_path)
le_area = joblib.load(le_area_path)
le_item = joblib.load(le_item_path)

# Mengambil list area dan item yang valid (diurutkan secara alfabetis)
areas_list = sorted(list(le_area.classes_))
items_list = sorted(list(le_item.classes_))

@app.route('/options', methods=['GET'])
def get_options():
    """
    Endpoint untuk mendapatkan daftar opsi negara (area) dan tanaman (item) yang valid.
    """
    return jsonify({
        'areas': areas_list,
        'items': items_list
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk memprediksi hasil panen.
    Menerima data JSON dengan parameter:
      - area: nama negara (string)
      - item: nama tanaman (string)
      - year: tahun prediksi (int)
      - average_rain_fall_mm_per_year: curah hujan rata-rata (float)
      - pesticides_tonnes: pestisida yang digunakan dalam ton (float)
      - avg_temp: suhu rata-rata (float)
      - model_type: tipe model ('rf' atau 'knn')
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Format input harus berupa JSON'}), 400

    required_fields = ['area', 'item', 'year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Field berikut wajib diisi: {", ".join(missing_fields)}'}), 400

    try:
        # 1. Ekstrak input dan validasi tipe data
        area_str = str(data['area']).strip()
        item_str = str(data['item']).strip()
        year = int(data['year'])
        rain = float(data['average_rain_fall_mm_per_year'])
        pesticides = float(data['pesticides_tonnes'])
        temp = float(data['avg_temp'])
        model_type = str(data.get('model_type', 'rf')).strip().lower()
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Input tidak valid: pastikan tipe data angka diisi dengan benar. Detail: {str(e)}'}), 400

    if model_type not in ['rf', 'knn']:
        return jsonify({'error': 'Model type harus bernilai "rf" atau "knn"'}), 400

    # 2. Encode fitur kategorikal
    try:
        encoded_area = le_area.transform([area_str])[0]
    except ValueError:
        return jsonify({'error': f'Negara/Area "{area_str}" tidak terdaftar dalam model.'}), 400

    try:
        encoded_item = le_item.transform([item_str])[0]
    except ValueError:
        return jsonify({'error': f'Tanaman/Item "{item_str}" tidak terdaftar dalam model.'}), 400

    # 3. Susun dataframe sesuai urutan fitur modeling
    features_df = pd.DataFrame([{
        'area': encoded_area,
        'item': encoded_item,
        'year': year,
        'average_rain_fall_mm_per_year': rain,
        'pesticides_tonnes': pesticides,
        'avg_temp': temp
    }])

    # Pastikan urutan kolom sesuai dengan modeling
    features_df = features_df[['area', 'item', 'year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]

    # 4. Lakukan prediksi
    try:
        if model_type == 'knn':
            prediction_hg_ha = model_knn.predict(features_df)[0]
        else:
            prediction_hg_ha = model_rf.predict(features_df)[0]
            
        # 1 hg/ha = 0.1 kg/ha = 0.0001 tonnes/ha (1 ton = 10.000 hg)
        prediction_tonnes_ha = prediction_hg_ha / 10000.0
        
        return jsonify({
            'success': True,
            'input': {
                'area': area_str,
                'item': item_str,
                'year': year,
                'average_rain_fall_mm_per_year': rain,
                'pesticides_tonnes': pesticides,
                'avg_temp': temp
            },
            'model_type': model_type,
            'prediction': {
                'yield_hg_ha': round(prediction_hg_ha, 2),
                'yield_tonnes_ha': round(prediction_tonnes_ha, 4)
            }
        })
    except Exception as e:
        return jsonify({'error': f'Gagal melakukan prediksi dengan model {model_type.upper()}: {str(e)}'}), 500

if __name__ == '__main__':
    # Menjalankan server pada port 5001
    print("Memulai server Flask di http://localhost:5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)

