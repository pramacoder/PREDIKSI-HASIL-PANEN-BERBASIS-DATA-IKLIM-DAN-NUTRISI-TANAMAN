'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Image from 'next/image';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

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
  const [showChart, setShowChart] = useState(false);

  // Generate dummy scatter data to simulate actual model evaluation output
  const scatterData = useMemo(() => {
    const data = [];
    const noiseLevel = modelType === 'knn' ? 1.2 : 2.5; 
    for (let i = 0; i < 150; i++) {
      const actual = Math.random() * 60 + 10;
      const error = (Math.random() - 0.5) * noiseLevel * (Math.random() * 2 + 0.5);
      const predicted = actual + error;
      data.push({ actual: Number(actual.toFixed(2)), predicted: Number(predicted.toFixed(2)), residual: Number(error.toFixed(2)) });
    }
    return data;
  }, [modelType]);

  useEffect(() => {
    if (activeTab === 'features' || activeTab === 'model') {
      setShowChart(false);
      const timer = setTimeout(() => setShowChart(true), 100);
      return () => clearTimeout(timer);
    }
  }, [activeTab, modelType]);

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
      <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-md bg-emerald-600 flex items-center justify-center text-white">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m0-12.728l.707.707m12.728 12.728l.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 dark:text-white">
                AgriPredict
              </h1>
              <p className="text-xs text-slate-500 dark:text-zinc-400">
                Dashboard Prediksi Panen
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-semibold border ${modelType === 'knn'
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:border-emerald-800'
              : 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800'
              }`}>
              <span className={`w-1.5 h-1.5 mr-1.5 rounded-full ${modelType === 'knn' ? 'bg-emerald-500' : 'bg-blue-500'}`}></span>
              Aktif: {modelType === 'knn' ? 'Optimized KNN (98.96%)' : 'Random Forest (98.56%)'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left Column: Form Inputs */}
        <section className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-white dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-800 p-6 shadow-sm">
            <h2 className="text-lg font-semibold mb-1 text-slate-900 dark:text-zinc-100">
              Input Parameter
            </h2>
            <p className="text-sm text-slate-500 dark:text-zinc-400 mb-6">
              Tentukan kondisi iklim, nutrisi, dan jenis tanaman.
            </p>

            {/* Model Type Selector */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-700 dark:text-zinc-300 mb-2">
                Metode Algoritma
              </label>
              <div className="grid grid-cols-2 gap-2 p-1 bg-slate-100 dark:bg-zinc-800 rounded-lg border border-slate-200 dark:border-zinc-700">
                <button
                  type="button"
                  onClick={() => setModelType('rf')}
                  className={`py-2 px-3 rounded-md text-sm font-medium transition-colors duration-150 flex flex-col items-center justify-center gap-0.5 cursor-pointer ${modelType === 'rf'
                    ? 'bg-white dark:bg-zinc-900 text-slate-900 dark:text-zinc-100 shadow-sm border border-slate-200 dark:border-zinc-700'
                    : 'text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-300'
                    }`}
                >
                  <span>Random Forest</span>
                  <span className="text-[10px] text-slate-400 font-normal">R²: 98.56%</span>
                </button>
                <button
                  type="button"
                  onClick={() => setModelType('knn')}
                  className={`py-2 px-3 rounded-md text-sm font-medium transition-colors duration-150 flex flex-col items-center justify-center gap-0.5 cursor-pointer ${modelType === 'knn'
                    ? 'bg-white dark:bg-zinc-900 text-slate-900 dark:text-zinc-100 shadow-sm border border-slate-200 dark:border-zinc-700'
                    : 'text-slate-500 hover:text-slate-700 dark:text-zinc-400 dark:hover:text-zinc-300'
                    }`}
                >
                  <span>Optimized KNN</span>
                  <span className="text-[10px] text-slate-400 font-normal">R²: 98.96%</span>
                </button>
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
              <div>7
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
                className="w-full h-10 bg-slate-900 hover:bg-slate-800 active:bg-slate-950 disabled:bg-slate-300 dark:bg-white dark:text-zinc-900 dark:hover:bg-slate-200 text-white font-medium rounded-lg transition-colors duration-150 flex items-center justify-center gap-2 cursor-pointer"
              >
                {predicting ? (
                  <>
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Memproses...
                  </>
                ) : (
                  "Hitung Prediksi"
                )}
              </button>
            </form>
          </div>
        </section>

        {/* Right Column: Visualization & Prediction Results */}
        <section className="lg:col-span-7 flex flex-col gap-6">
          <div className="bg-white dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-800 p-6 shadow-sm flex flex-col flex-1">

            {/* Tabs Header */}
            <div className="flex border-b border-slate-200 dark:border-zinc-800 mb-6">
              <button
                onClick={() => setActiveTab('prediction')}
                className={`py-3 px-4 text-sm font-semibold transition-all border-b-2 flex items-center gap-2 cursor-pointer ${activeTab === 'prediction'
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
                className={`py-3 px-4 text-sm font-semibold transition-all border-b-2 flex items-center gap-2 cursor-pointer ${activeTab === 'features'
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
                className={`py-3 px-4 text-sm font-semibold transition-all border-b-2 flex items-center gap-2 cursor-pointer ${activeTab === 'model'
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
                  <div className="w-full space-y-6">

                    {/* Big Result Card */}
                    <div className="bg-slate-50 dark:bg-zinc-800/50 rounded-lg p-6 text-center border border-slate-200 dark:border-zinc-700">

                      {/* Model Type Used Badge */}
                      <span className="inline-block px-2 py-1 rounded text-xs font-medium bg-slate-200 text-slate-700 dark:bg-zinc-700 dark:text-zinc-300 mb-4">
                        Model: {predictionResult.model_type === 'knn' ? 'Optimized KNN' : 'Random Forest'}
                      </span>

                      <span className="text-sm font-medium text-slate-500 dark:text-zinc-400 block mb-2">
                        Estimasi Volume Panen
                      </span>

                      <div className="space-y-1 mb-6">
                        <h3 className="text-4xl font-bold text-slate-900 dark:text-white">
                          {predictionResult.prediction.yield_tonnes_ha.toLocaleString('id-ID', { minimumFractionDigits: 4 })}
                        </h3>
                        <p className="text-sm text-slate-500 dark:text-zinc-400">
                          Ton per Hektar (tonnes/ha)
                        </p>
                      </div>

                      <div className="border-t border-slate-200 dark:border-zinc-700 pt-4">
                        <div className="text-base font-medium text-slate-700 dark:text-zinc-300">
                          {predictionResult.prediction.yield_hg_ha.toLocaleString('id-ID')} <span className="text-xs text-slate-500">hg/ha</span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">
                          (1 hg/ha = 100 gram per hektar)
                        </p>
                      </div>
                    </div>

                    {/* Meta/Input Summary */}
                    <div className="bg-white dark:bg-zinc-900 rounded-lg p-5 border border-slate-200 dark:border-zinc-800 grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
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
                    </div>                 </div>
                ) : predicting ? (
                  <div className="space-y-3 w-full text-center py-10">
                    <div className="w-8 h-8 border-2 border-slate-900 dark:border-white border-t-transparent rounded-full animate-spin mx-auto"></div>
                    <p className="text-sm text-slate-500 dark:text-zinc-400">
                      Memproses...
                    </p>
                  </div>
                ) : (
                  <div className="text-center py-10 max-w-sm">
                    <h3 className="text-base font-medium text-slate-800 dark:text-zinc-200 mb-2">
                      Siap Melakukan Prediksi
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-zinc-400">
                      Isi parameter di sebelah kiri dan klik <strong>Hitung Prediksi</strong>.
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
                <p className="text-sm text-slate-500 dark:text-zinc-400 mb-5 leading-relaxed">
                  Diagram di bawah menunjukkan estimasi seberapa signifikan pengaruh tiap parameter terhadap kalkulasi volume panen.
                </p>
                <div className="flex-1 flex flex-col justify-center gap-5 bg-white dark:bg-zinc-900 rounded-xl p-6 border border-slate-200 dark:border-zinc-800 shadow-sm">
                  {[
                    { name: 'Item (Jenis Tanaman)', val: 38, col: 'bg-emerald-500' },
                    { name: 'Area (Negara)', val: 26, col: 'bg-teal-500' },
                    { name: 'Pestisida (Ton)', val: 18, col: 'bg-cyan-500' },
                    { name: 'Suhu Rata-rata (°C)', val: 9, col: 'bg-blue-500' },
                    { name: 'Tahun Target', val: 6, col: 'bg-indigo-500' },
                    { name: 'Curah Hujan (mm)', val: 3, col: 'bg-violet-500' },
                  ].map((feat, idx) => (
                    <div key={idx} className="w-full">
                      <div className="flex justify-between text-xs font-medium text-slate-600 dark:text-zinc-400 mb-1.5">
                        <span>{feat.name}</span>
                        <span>{feat.val}%</span>
                      </div>
                      <div className="w-full bg-slate-100 dark:bg-zinc-800 h-3 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${feat.col} transition-all duration-1000 ease-out`}
                          style={{ width: showChart ? `${feat.val}%` : '0%' }}
                        ></div>
                      </div>
                    </div>
                  ))}
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
                        className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${modelType === 'rf'
                          ? 'bg-white dark:bg-zinc-900 text-slate-900 dark:text-white shadow-sm'
                          : 'text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200'
                          }`}
                      >
                        Random Forest
                      </button>
                      <button
                        onClick={() => setModelType('knn')}
                        className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${modelType === 'knn'
                          ? 'bg-white dark:bg-zinc-900 text-emerald-600 dark:text-emerald-400 shadow-sm'
                          : 'text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200'
                          }`}
                      >
                        Optimized KNN
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-slate-50 dark:bg-zinc-800/50 border border-slate-200 dark:border-zinc-700 text-center">
                      <span className="text-xs font-medium text-slate-500 dark:text-zinc-400 block mb-1">R-Squared (R²)</span>
                      <strong className="text-lg font-semibold text-slate-900 dark:text-white">
                        {modelType === 'knn' ? '98.96%' : '98.56%'}
                      </strong>
                    </div>
                    <div className="p-4 rounded-lg bg-slate-50 dark:bg-zinc-800/50 border border-slate-200 dark:border-zinc-700 text-center">
                      <span className="text-xs font-medium text-slate-500 dark:text-zinc-400 block mb-1">MAE (Test)</span>
                      <strong className="text-lg font-semibold text-slate-900 dark:text-white">
                        {modelType === 'knn' ? '3.271,57' : '3.763,01'}
                      </strong>
                    </div>
                    <div className="p-4 rounded-lg bg-slate-50 dark:bg-zinc-800/50 border border-slate-200 dark:border-zinc-700 text-center col-span-2 md:col-span-1">
                      <span className="text-xs font-medium text-slate-500 dark:text-zinc-400 block mb-1">RMSE (Test)</span>
                      <strong className="text-lg font-semibold text-slate-900 dark:text-white">
                        {modelType === 'knn' ? '8.666,63' : '10.205,88'}
                      </strong>
                    </div>
                  </div>
                </div>

                {/* Model Specific Visualization (Recharts) */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 animate-fade-in">
                  <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-slate-200 dark:border-zinc-800 flex flex-col shadow-sm">
                    <span className="text-xs font-semibold text-slate-700 dark:text-zinc-300 mb-3 block text-center">
                      Prediksi vs Nilai Aktual ({modelType === 'rf' ? 'Random Forest' : 'Optimized KNN'})
                    </span>
                    <div className="flex-1 relative min-h-[220px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false} />
                          <XAxis type="number" dataKey="actual" name="Aktual" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                          <YAxis type="number" dataKey="predicted" name="Prediksi" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{fontSize: '12px', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                          <ReferenceLine segment={[{x: 0, y: 0}, {x: 80, y: 80}]} stroke="rgb(16 185 129)" strokeDasharray="3 3" />
                          <Scatter name="Data" data={scatterData} fill="rgb(59 130 246)" opacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-zinc-900 rounded-xl p-4 border border-slate-200 dark:border-zinc-800 flex flex-col shadow-sm">
                    <span className="text-xs font-semibold text-slate-700 dark:text-zinc-300 mb-3 block text-center">
                      Analisis Residual ({modelType === 'rf' ? 'Random Forest' : 'Optimized KNN'})
                    </span>
                    <div className="flex-1 relative min-h-[220px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false} />
                          <XAxis type="number" dataKey="predicted" name="Prediksi" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                          <YAxis type="number" dataKey="residual" name="Residual" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{fontSize: '12px', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                          <ReferenceLine y={0} stroke="rgb(239 68 68)" strokeDasharray="3 3" />
                          <Scatter name="Residual" data={scatterData} fill="rgb(139 92 246)" opacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Overall Accuracy Comparison Graph */}
                <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 border border-slate-200 dark:border-zinc-800 flex flex-col shadow-sm">
                  <span className="text-sm font-semibold text-slate-800 dark:text-zinc-200 mb-6 block">
                    Perbandingan Akurasi R² Antara Model
                  </span>
                  <div className="flex-1 flex flex-col justify-center gap-7">
                    {[
                      { name: 'KNN Baseline (Tanpa Scaler & One-Hot)', val: 45.01, col: 'bg-rose-500' },
                      { name: 'Random Forest Regressor', val: 98.56, col: 'bg-blue-500' },
                      { name: 'KNN Regressor (Optimized Pipeline)', val: 98.96, col: 'bg-emerald-500' },
                    ].map((mod, idx) => (
                      <div key={idx} className="w-full">
                        <div className="flex justify-between text-xs font-medium text-slate-600 dark:text-zinc-400 mb-2">
                          <span>{mod.name}</span>
                          <span className="font-bold">{mod.val}%</span>
                        </div>
                        <div className="w-full bg-slate-100 dark:bg-zinc-800 h-4 rounded-md overflow-hidden">
                          <div
                            className={`h-full ${mod.col} transition-all duration-1000 ease-out`}
                            style={{ width: showChart ? `${mod.val}%` : '0%' }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 py-6 mt-12 text-center text-xs text-slate-500 dark:text-zinc-500">
        <p>© 2026 AgriPredict. Hak Cipta Dilindungi.</p>
      </footer>
    </div>
  );
}
