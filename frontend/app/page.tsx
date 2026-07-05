'use client';

import React, { useState, useEffect } from 'react';
import Image from 'next/image';

interface Options {
  areas: string[];
  items: string[];
}

interface PredictionResult {
  success: boolean;
  model_type?: string;
  prediction: {
    yield_hg_ha: number;
    yield_tonnes_ha: number;
  };
  input: {
    area: string;
    item: string;
    year: number;
    average_rain_fall_mm_per_year: number;
    pesticides_tonnes: number;
    avg_temp: number;
  };
}

export default function Home() {
  // Model yang dipilih: rf atau knn
  const [modelType, setModelType] = useState<'rf' | 'knn'>('rf');

  // Opsi input dinamis dari backend
  const [options, setOptions] = useState<Options>({ areas: [], items: [] });
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [optionsError, setOptionsError] = useState('');

  // Search filter untuk Area/Negara
  const [areaSearch, setAreaSearch] = useState('');
  const [showAreaDropdown, setShowAreaDropdown] = useState(false);

  // Form inputs
  const [formData, setFormData] = useState({
    area: '',
    item: '',
    year: 2026,
    average_rain_fall_mm_per_year: 1200.0,
    pesticides_tonnes: 500.0,
    avg_temp: 20.0,
  });

  // State untuk Prediksi
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [predictError, setPredictError] = useState('');

  // Tab di panel visualisasi/insights
  const [activeTab, setActiveTab] = useState<'prediction' | 'features' | 'model'>('prediction');

  // Load opsi dari backend saat halaman dimuat
  useEffect(() => {
    fetch('http://localhost:5001/options')
      .then((res) => {
        if (!res.ok) throw new Error('Gagal mengambil opsi dari server backend');
        return res.json();
      })
      .then((data: Options) => {
        setOptions(data);
        setLoadingOptions(false);
      })
      .catch((err) => {
        console.error(err);
        setOptionsError('Gagal tersambung ke backend API. Pastikan server Flask sudah berjalan di port 5001.');
        setLoadingOptions(false);
      });
  }, []);

  // Filter area berdasarkan search keyword
  const filteredAreas = options.areas.filter((area) =>
    area.toLowerCase().includes(areaSearch.toLowerCase())
  );

  const handleSelectArea = (area: string) => {
    setFormData({ ...formData, area });
    setAreaSearch(area);
    setShowAreaDropdown(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: name === 'item' || name === 'area' ? value : parseFloat(value) || value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.area) {
      setPredictError('Silakan pilih Negara / Area terlebih dahulu.');
      return;
    }
    if (!formData.item) {
      setPredictError('Silakan pilih Jenis Tanaman terlebih dahulu.');
      return;
    }

    setPredicting(true);
    setPredictError('');
    setPredictionResult(null);

    try {
      const res = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          model_type: modelType
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Terjadi kesalahan saat memprediksi.');
      }

      setPredictionResult(data);
      setActiveTab('prediction'); // Pindah ke tab prediksi untuk menampilkan hasil
    } catch (err: any) {
      setPredictError(err.message || 'Gagal terhubung ke API.');
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 dark:bg-zinc-950 dark:text-zinc-100 flex flex-col font-sans transition-colors duration-300">
      {/* Navbar Premium */}
      <header className="sticky top-0 z-40 w-full border-b border-slate-200/80 bg-white/80 backdrop-blur-md dark:border-zinc-800/80 dark:bg-zinc-900/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-600 flex items-center justify-center text-white shadow-lg shadow-emerald-600/30">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m0-12.728l.707.707m12.728 12.728l.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-emerald-600 to-teal-500 bg-clip-text text-transparent">
                AgriPredict
              </h1>
              <p className="text-[10px] font-medium text-slate-500 dark:text-zinc-400">
                Sistem Prediksi Hasil Panen Tanaman
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${
              modelType === 'knn'
                ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-400'
                : 'bg-blue-100 text-blue-800 dark:bg-blue-950/50 dark:text-blue-400'
            }`}>
              <span className={`w-1.5 h-1.5 mr-1.5 rounded-full ${modelType === 'knn' ? 'bg-emerald-500' : 'bg-blue-500'} animate-pulse`}></span>
              Model Aktif: {modelType === 'knn' ? 'Optimized KNN (R²: 98.96%)' : 'Random Forest (R²: 98.56%)'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Form Inputs */}
        <section className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-white dark:bg-zinc-900 rounded-3xl border border-slate-200/80 dark:border-zinc-800/80 p-6 shadow-xl shadow-slate-100/50 dark:shadow-none">
            <h2 className="text-xl font-bold mb-1 flex items-center gap-2 text-slate-900 dark:text-zinc-100">
              <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              Input Parameter
            </h2>
            <p className="text-xs text-slate-500 dark:text-zinc-400 mb-6">
              Masukkan parameter kondisi iklim, nutrisi (pestisida), dan jenis tanaman.
            </p>

            {/* Model Type Selector */}
            <div className="mb-6">
              <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400 mb-2">
                Metode Algoritma ML
              </label>
              <div className="grid grid-cols-2 gap-3 p-1 bg-slate-100 dark:bg-zinc-950/60 rounded-xl border border-slate-200/50 dark:border-zinc-800/80">
                <button
                  type="button"
                  onClick={() => setModelType('rf')}
                  className={`py-2.5 px-3 rounded-lg text-xs font-bold transition-all duration-200 flex flex-col items-center justify-center gap-1 cursor-pointer ${
                    modelType === 'rf'
                      ? 'bg-white dark:bg-zinc-900 text-emerald-600 dark:text-emerald-400 shadow-md border border-slate-100 dark:border-zinc-800'
                      : 'text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-200'
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    <span className={`w-1.5 h-1.5 rounded-full ${modelType === 'rf' ? 'bg-emerald-500' : 'bg-transparent'}`}></span>
                    Random Forest
                  </span>
                  <span className="text-[10px] opacity-70 font-normal">R²: 98.56%</span>
                </button>
                {/* <button
                  type="button"
                  onClick={() => setModelType('knn')}
                  className={`py-2.5 px-3 rounded-lg text-xs font-bold transition-all duration-200 flex flex-col items-center justify-center gap-1 cursor-pointer relative ${
                    modelType === 'knn'
                      ? 'bg-white dark:bg-zinc-900 text-emerald-600 dark:text-emerald-400 shadow-md border border-slate-100 dark:border-zinc-800'
                      : 'text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-200'
                  }`}
                >
                  <span className="absolute -top-2 -right-1 bg-gradient-to-r from-amber-500 to-orange-500 text-white text-[8px] font-black px-2 py-0.5 rounded-full shadow-sm animate-pulse tracking-wide">
                    TERBAIK
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className={`w-1.5 h-1.5 rounded-full ${modelType === 'knn' ? 'bg-emerald-500' : 'bg-transparent'}`}></span>
                    Optimized KNN
                  </span>
                  <span className="text-[10px] opacity-70 font-normal">R²: 98.96%</span>
                </button> */}
              </div>
            </div>

            {optionsError && (
              <div className="p-4 mb-5 rounded-2xl bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/30 text-amber-800 dark:text-amber-400 text-xs leading-relaxed">
                <span className="font-semibold">Perhatian:</span> {optionsError}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-5">
              
              {/* Searchable Area (Country) Select */}
              <div className="relative">
                <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400 mb-1.5">
                  Negara / Area
                </label>
                <div className="relative">
                  <input
                    type="text"
                    placeholder={loadingOptions ? "Loading..." : "Cari & pilih negara..."}
                    disabled={loadingOptions || options.areas.length === 0}
                    value={areaSearch}
                    onChange={(e) => {
                      setAreaSearch(e.target.value);
                      setShowAreaDropdown(true);
                      if (formData.area !== e.target.value) {
                        setFormData({ ...formData, area: '' }); // reset selected area
                      }
                    }}
                    onFocus={() => setShowAreaDropdown(true)}
                    className="w-full h-11 pl-4 pr-10 rounded-xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950/50 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all duration-200"
                  />
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>

                {/* Dropdown Options */}
                {showAreaDropdown && filteredAreas.length > 0 && (
                  <ul className="absolute z-50 w-full mt-1.5 max-h-60 overflow-y-auto rounded-xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-1 shadow-2xl focus:outline-none text-sm">
                    {filteredAreas.map((area) => (
                      <li
                        key={area}
                        onClick={() => handleSelectArea(area)}
                        className="cursor-pointer select-none rounded-lg px-4 py-2 hover:bg-slate-100 dark:hover:bg-zinc-800 text-slate-700 dark:text-zinc-300 font-medium hover:text-slate-900 dark:hover:text-white"
                      >
                        {area}
                      </li>
                    ))}
                  </ul>
                )}
                {showAreaDropdown && filteredAreas.length === 0 && (
                  <div className="absolute z-50 w-full mt-1.5 rounded-xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-4 shadow-2xl text-xs text-slate-500 dark:text-zinc-400 text-center">
                    Tidak ditemukan area cocok
                  </div>
                )}
                {/* Backdrop to close dropdown */}
                {showAreaDropdown && (
                  <div className="fixed inset-0 z-40 cursor-default" onClick={() => setShowAreaDropdown(false)} />
                )}
              </div>

              {/* Crop / Item Select */}
              <div>
                <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400 mb-1.5">
                  Jenis Tanaman (Item)
                </label>
                <select
                  name="item"
                  value={formData.item}
                  onChange={handleInputChange}
                  disabled={loadingOptions || options.items.length === 0}
                  className="w-full h-11 px-4 rounded-xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950/50 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all duration-200 cursor-pointer"
                >
                  <option value="">Pilih tanaman...</option>
                  {options.items.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>

              {/* Grid 2 Columns: Year & Temp */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400 mb-1.5">
                    Tahun
                  </label>
                  <input
                    type="number"
                    name="year"
                    value={formData.year}
                    onChange={handleInputChange}
                    min="1990"
                    max="2100"
                    className="w-full h-11 px-4 rounded-xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950/50 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all duration-200"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400 mb-1.5">
                    Suhu Rata-rata (°C)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    name="avg_temp"
                    value={formData.avg_temp}
                    onChange={handleInputChange}
                    className="w-full h-11 px-4 rounded-xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950/50 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all duration-200"
                  />
                </div>
              </div>

              {/* Slider / Inputs: Rain Fall */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400">
                    Curah Hujan Tahunan (mm/tahun)
                  </label>
                  <span className="text-xs font-bold text-emerald-600">{formData.average_rain_fall_mm_per_year} mm</span>
                </div>
                <input
                  type="range"
                  name="average_rain_fall_mm_per_year"
                  min="50"
                  max="3500"
                  step="10"
                  value={formData.average_rain_fall_mm_per_year}
                  onChange={handleInputChange}
                  className="w-full h-2 rounded-lg bg-slate-200 dark:bg-zinc-800 accent-emerald-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 dark:text-zinc-500 mt-1">
                  <span>50 mm</span>
                  <span>1.800 mm (Optimal)</span>
                  <span>3.500 mm</span>
                </div>
              </div>

              {/* Slider / Inputs: Pesticides */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="block text-xs font-semibold text-slate-600 dark:text-zinc-400">
                    Penggunaan Pestisida (Ton)
                  </label>
                  <span className="text-xs font-bold text-emerald-600">{formData.pesticides_tonnes} Ton</span>
                </div>
                <input
                  type="range"
                  name="pesticides_tonnes"
                  min="0.1"
                  max="5000"
                  step="1"
                  value={formData.pesticides_tonnes}
                  onChange={handleInputChange}
                  className="w-full h-2 rounded-lg bg-slate-200 dark:bg-zinc-800 accent-emerald-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 dark:text-zinc-500 mt-1">
                  <span>0.1 Ton</span>
                  <span>2.500 Ton</span>
                  <span>5.000 Ton</span>
                </div>
              </div>

              {predictError && (
                <div className="p-3 text-xs rounded-xl bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/30 text-rose-600 dark:text-rose-400">
                  {predictError}
                </div>
              )}

              <button
                type="submit"
                disabled={predicting || loadingOptions}
                className="w-full h-12 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 disabled:bg-emerald-600/40 text-white font-semibold rounded-xl shadow-lg shadow-emerald-600/20 hover:shadow-emerald-700/30 transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer"
              >
                {predicting ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Menghitung Prediksi...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    Hitung Prediksi Hasil Panen
                  </>
                )}
              </button>
            </form>
          </div>
        </section>

        {/* Right Column: Visualization & Prediction Results */}
        <section className="lg:col-span-7 flex flex-col gap-6">
          <div className="bg-white dark:bg-zinc-900 rounded-3xl border border-slate-200/80 dark:border-zinc-800/80 p-6 shadow-xl shadow-slate-100/50 dark:shadow-none flex flex-col flex-1">
            
            {/* Tabs Header */}
            <div className="flex border-b border-slate-200 dark:border-zinc-800 mb-6">
              <button
                onClick={() => setActiveTab('prediction')}
                className={`py-3 px-4 text-sm font-semibold transition-all border-b-2 flex items-center gap-2 cursor-pointer ${
                  activeTab === 'prediction'
                    ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
                    : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-200'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
                </svg>
                Hasil Prediksi
              </button>
              
              <button
                onClick={() => setActiveTab('features')}
                className={`py-3 px-4 text-sm font-semibold transition-all border-b-2 flex items-center gap-2 cursor-pointer ${
                  activeTab === 'features'
                    ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
                    : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-200'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Pentingnya Fitur (Importance)
              </button>

              <button
                onClick={() => setActiveTab('model')}
                className={`py-3 px-4 text-sm font-semibold transition-all border-b-2 flex items-center gap-2 cursor-pointer ${
                  activeTab === 'model'
                    ? 'border-emerald-500 text-emerald-600 dark:text-emerald-400'
                    : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-200'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                </svg>
                Evaluasi Model
              </button>
            </div>

            {/* Tab: Prediction Results */}
            {activeTab === 'prediction' && (
              <div className="flex-1 flex flex-col items-center justify-center min-h-[350px]">
                {predictionResult ? (
                  <div className="w-full space-y-6 animate-fade-in">
                    
                    {/* Big Result Card */}
                    <div className="bg-gradient-to-br from-emerald-600/90 to-teal-700/90 dark:from-emerald-950/80 dark:to-teal-950/80 text-white rounded-3xl p-8 text-center shadow-xl shadow-emerald-700/10 border border-emerald-500/20 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full blur-2xl -translate-y-6 translate-x-6"></div>
                      <div className="absolute bottom-0 left-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-2xl translate-y-6 -translate-x-6"></div>
                      
                      {/* Model Type Used Badge */}
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-[10px] font-bold bg-white/20 text-white mb-3 backdrop-blur-sm shadow-sm gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span>
                        Model: {predictionResult.model_type === 'knn' ? 'Optimized KNN' : 'Random Forest'}
                      </span>

                      <span className="text-[11px] font-bold uppercase tracking-widest text-emerald-200/90 block mb-2">
                        Estimasi Yield Hasil Panen
                      </span>
                      
                      <div className="space-y-1">
                        <h3 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
                          {predictionResult.prediction.yield_tonnes_ha.toLocaleString('id-ID', { minimumFractionDigits: 4 })}
                        </h3>
                        <p className="text-sm font-semibold text-emerald-200/80">
                          Ton per Hektar (tonnes/ha)
                        </p>
                      </div>

                      <div className="my-6 border-t border-white/10 w-24 mx-auto"></div>

                      <div className="text-lg font-medium text-slate-100">
                        {predictionResult.prediction.yield_hg_ha.toLocaleString('id-ID')} <span className="text-xs text-emerald-300 font-semibold">hg/ha</span>
                      </div>
                      <p className="text-[10px] text-emerald-200/60 mt-1">
                        1 hg/ha = 100 gram per hektar
                      </p>
                    </div>

                    {/* Meta/Input Summary */}
                    <div className="bg-slate-50 dark:bg-zinc-950/40 rounded-2xl p-5 border border-slate-100 dark:border-zinc-800 grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
                      <div>
                        <span className="text-slate-400 dark:text-zinc-500 block mb-0.5">Lokasi</span>
                        <strong className="text-slate-700 dark:text-zinc-300 font-bold">{predictionResult.input.area}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 dark:text-zinc-500 block mb-0.5">Komoditas</span>
                        <strong className="text-slate-700 dark:text-zinc-300 font-bold">{predictionResult.input.item}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 dark:text-zinc-500 block mb-0.5">Tahun Target</span>
                        <strong className="text-slate-700 dark:text-zinc-300 font-bold">{predictionResult.input.year}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 dark:text-zinc-500 block mb-0.5">Curah Hujan</span>
                        <strong className="text-slate-700 dark:text-zinc-300 font-bold">{predictionResult.input.average_rain_fall_mm_per_year} mm/tahun</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 dark:text-zinc-500 block mb-0.5">Pestisida</span>
                        <strong className="text-slate-700 dark:text-zinc-300 font-bold">{predictionResult.input.pesticides_tonnes} Ton</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 dark:text-zinc-500 block mb-0.5">Suhu</span>
                        <strong className="text-slate-700 dark:text-zinc-300 font-bold">{predictionResult.input.avg_temp} °C</strong>
                      </div>
                    </div>

                  </div>
                ) : predicting ? (
                  <div className="space-y-4 w-full text-center py-10">
                    <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto shadow-md shadow-emerald-500/10"></div>
                    <p className="text-sm font-semibold text-slate-600 dark:text-zinc-400 animate-pulse">
                      Menghubungi server ML & memproses hasil prediksi...
                    </p>
                  </div>
                ) : (
                  <div className="text-center py-10 max-w-sm">
                    <div className="w-20 h-20 bg-emerald-50 dark:bg-emerald-950/20 rounded-full flex items-center justify-center mx-auto mb-4 text-emerald-600 dark:text-emerald-500 shadow-inner">
                      <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.782 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-800 dark:text-zinc-200 mb-1">
                      Siap Melakukan Prediksi
                    </h3>
                    <p className="text-xs text-slate-500 dark:text-zinc-400 leading-relaxed">
                      Silakan isi form parameter iklim, negara, dan komoditas tanaman di sebelah kiri, lalu klik <strong>Hitung Prediksi Hasil Panen</strong>.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Tab: Features Importance */}
            {activeTab === 'features' && (
              <div className="flex-1 flex flex-col min-h-[350px]">
                <h3 className="text-base font-bold mb-3 text-slate-800 dark:text-zinc-200">
                  Feature Importance
                </h3>
                <p className="text-xs text-slate-500 dark:text-zinc-400 mb-5 leading-relaxed">
                  Berdasarkan pemodelan Random Forest Regressor, diagram ini menunjukkan seberapa signifikan pengaruh tiap variabel terhadap volume panen (hg/ha).
                </p>
                <div className="flex-1 relative min-h-[250px] bg-slate-100 dark:bg-zinc-950/50 rounded-2xl overflow-hidden border border-slate-200/50 dark:border-zinc-800/80 flex items-center justify-center">
                  <Image
                    src="/modeling_feature_importance.png"
                    alt="Modeling Feature Importance"
                    width={500}
                    height={300}
                    className="object-contain"
                  />
                </div>
              </div>
            )}

            {/* Tab: Model Evaluation */}
            {activeTab === 'model' && (
              <div className="flex-1 flex flex-col min-h-[350px] space-y-6">
                <div>
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                    <h3 className="text-base font-bold text-slate-800 dark:text-zinc-200">
                      Evaluasi & Performa Model
                    </h3>
                    
                    {/* Switcher within evaluation tab */}
                    <div className="flex bg-slate-100 dark:bg-zinc-950/50 p-0.5 rounded-lg border border-slate-200/50 dark:border-zinc-800/80 text-xs font-semibold self-start sm:self-auto">
                      <button
                        onClick={() => setModelType('rf')}
                        className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                          modelType === 'rf'
                            ? 'bg-white dark:bg-zinc-900 text-slate-900 dark:text-white shadow-sm'
                            : 'text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200'
                        }`}
                      >
                        Random Forest
                      </button>
                      <button
                        onClick={() => setModelType('knn')}
                        className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                          modelType === 'knn'
                            ? 'bg-white dark:bg-zinc-900 text-emerald-600 dark:text-emerald-400 shadow-sm'
                            : 'text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200'
                        }`}
                      >
                        Optimized KNN
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-2xl bg-emerald-50/50 dark:bg-emerald-950/10 border border-emerald-100 dark:border-emerald-900/20 text-center">
                      <span className="text-[10px] uppercase font-bold text-slate-400 dark:text-zinc-500 block mb-0.5">R-Squared (R²)</span>
                      <strong className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
                        {modelType === 'knn' ? '98.96%' : '98.56%'}
                      </strong>
                      <p className="text-[9px] text-slate-400 mt-0.5">Akurasi prediksi test data</p>
                    </div>
                    <div className="p-4 rounded-2xl bg-emerald-50/50 dark:bg-emerald-950/10 border border-emerald-100 dark:border-emerald-900/20 text-center">
                      <span className="text-[10px] uppercase font-bold text-slate-400 dark:text-zinc-500 block mb-0.5">MAE (Test)</span>
                      <strong className="text-xl font-bold text-slate-700 dark:text-zinc-300">
                        {modelType === 'knn' ? '3.271,57' : '3.763,01'}
                      </strong>
                      <p className="text-[9px] text-slate-400 mt-0.5">Mean Absolute Error (hg/ha)</p>
                    </div>
                    <div className="p-4 rounded-2xl bg-emerald-50/50 dark:bg-emerald-950/10 border border-emerald-100 dark:border-emerald-900/20 text-center col-span-2 md:col-span-1">
                      <span className="text-[10px] uppercase font-bold text-slate-400 dark:text-zinc-500 block mb-0.5">RMSE (Test)</span>
                      <strong className="text-xl font-bold text-slate-700 dark:text-zinc-300">
                        {modelType === 'knn' ? '8.666,63' : '10.205,88'}
                      </strong>
                      <p className="text-[9px] text-slate-400 mt-0.5">Root Mean Square Error</p>
                    </div>
                  </div>
                </div>

                {/* Model Specific Visualization */}
                {modelType === 'rf' ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-100 dark:bg-zinc-950/50 rounded-2xl p-4 border border-slate-200/50 dark:border-zinc-800/80 flex flex-col">
                      <span className="text-xs font-bold text-slate-700 dark:text-zinc-300 mb-2 block">
                        Prediksi vs Nilai Aktual (Random Forest)
                      </span>
                      <div className="flex-1 relative min-h-[180px] flex items-center justify-center">
                        <Image
                          src="/modeling_pred_vs_actual.png"
                          alt="Prediction vs Actual RF"
                          width={300}
                          height={180}
                          className="object-contain"
                        />
                      </div>
                    </div>

                    <div className="bg-slate-100 dark:bg-zinc-950/50 rounded-2xl p-4 border border-slate-200/50 dark:border-zinc-800/80 flex flex-col">
                      <span className="text-xs font-bold text-slate-700 dark:text-zinc-300 mb-2 block">
                        Analisis Residual RF (Error)
                      </span>
                      <div className="flex-1 relative min-h-[180px] flex items-center justify-center">
                        <Image
                          src="/modeling_residual.png"
                          alt="Model Residuals RF"
                          width={300}
                          height={180}
                          className="object-contain"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-100 dark:bg-zinc-950/50 rounded-2xl p-4 border border-slate-200/50 dark:border-zinc-800/80 flex flex-col">
                    <span className="text-xs font-bold text-slate-700 dark:text-zinc-300 mb-2 block text-center">
                      Prediksi vs Aktual & Analisis Residual (Optimized KNN)
                    </span>
                    <div className="flex-1 relative min-h-[220px] flex items-center justify-center">
                      <Image
                        src="/knn_improved_evaluation.png"
                        alt="KNN Evaluation Plots"
                        width={600}
                        height={250}
                        className="object-contain rounded-xl"
                      />
                    </div>
                  </div>
                )}

                {/* Overall Accuracy Comparison Graph */}
                <div className="bg-slate-100 dark:bg-zinc-950/50 rounded-2xl p-4 border border-slate-200/50 dark:border-zinc-800/80 flex flex-col">
                  <span className="text-xs font-bold text-slate-700 dark:text-zinc-300 mb-2 block text-center">
                    Perbandingan Akurasi R² Antara Model Awal vs Model Baru
                  </span>
                  <div className="flex-1 relative min-h-[220px] flex items-center justify-center animate-fade-in">
                    <Image
                      src="/model_accuracy_comparison.png"
                      alt="Model Accuracy Comparison"
                      width={550}
                      height={250}
                      className="object-contain rounded-xl"
                    />
                  </div>
                </div>
              </div>
            )}

          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-slate-200 dark:border-zinc-800/80 bg-white dark:bg-zinc-900/30 py-6 mt-12 text-center text-xs text-slate-500 dark:text-zinc-500">
        <p>© 2026 AgriPredict. Project Hasil Panen Berbasis Data Iklim dan Nutrisi Tanaman.</p>
      </footer>
    </div>
  );
}
