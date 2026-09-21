# Global Superstore Intelligence: Customer Segmentation & Profitability Prediction

Final Project - LnT Camp 2026  
Tema: "Bridging the Gap: Empowering Future Talent through Machine Learning for Industry Innovation"

Author: Christian Immanuel Valerio - Tech Enthusiast

---

## Ringkasan Proyek

### 1. Situation
Dalam operasional bisnis ritel skala global, perusahaan menghadapi tantangan efisiensi biaya logistik serta tingginya transaksi yang merugikan akibat diskon berlebihan dan margin yang tergerus. Data mentah tersimpan dalam basis data relasional SQLite (`superstore.sqlite`). Diperlukan sistem analitik terintegrasi mulai dari ekstraksi basis data, pengelompokan pola belanja pelanggan, hingga inferensi model *machine learning* secara *real-time* melalui antarmuka pengguna interaktif.

### 2. Task
Proyek ini menyelesaikan dua tugas pemodelan *Machine Learning* utama:
* **Customer Segmentation (Clustering)**: Mengelompokkan pelanggan berdasarkan indikator RFM (Recency, Frequency, Monetary) tanpa label historis guna merancang strategi pemasaran terarah.
* **Order Profitability Prediction (Classification)**: Memprediksi potensi profitabilitas suatu transaksi pesanan (untung vs rugi) sebelum order diproses di tingkat operasional.

### 3. Action 
* **Data Extraction & Preprocessing**:
  * Mengekstraksi multi-tabel dari `superstore.sqlite` menggunakan `sqlite3` dan `pandas`.
  * Rekayasa fitur: Durasi pengiriman (`delivery_duration`), tier diskon (`discount_tier`), serta agregasi metrik pelanggan RFM.
* **Machine Learning Pipeline**:
  * **Clustering**: Model **K-Means** yang dioptimasi menggunakan **Elbow Method (Inertia)** dan divalidasi dengan nilai puncak **Silhouette Score** ($k=3$).
  * **Classification**: Membandingkan **Logistic Regression** (baseline) dengan **Random Forest Classifier** yang dilengkapi pipeline `ColumnTransformer` (penskalaan numerik dan *One-Hot Encoding* kategorikal).
* **Deployment & Serving**:
  * Serialisasi artefak pipeline model terlatih ke dalam format `.joblib` di folder `model/`.
  * Membangun REST API modular menggunakan **FastAPI** dengan endpoint pemantauan `/health` serta endpoint inferensi `/predict/cluster` dan `/predict/profitability`.
  * Membangun dasbor analitik berbasis **Streamlit** yang memuat visualisasi EDA, inferensi segmentasi pelanggan, dan simulasi keuntungan transaksi secara terpisah dari logika model.

### 4. Result
* **Evaluasi Clustering**: Berhasil membagi 51.290 transaksi ke dalam 3 segmen terpisah secara tegas: *High-Value Champions*, *Active Moderate Spenders*, dan *At-Risk Customers*.
* **Evaluasi Classification**: Model klasifikasi mencapai performa $F1\text{-score} > 0.94$ dengan kemampuan memisahkan ribuan transaksi merugi secara presisi.
* **Key Business Insights**:
  * Variabel diskon (`discount`) merupakan faktor pendorong kerugian paling dominan (*Feature Importance* > 0.31), disusul oleh beban biaya pengiriman (`shipping_cost`).
  * Penetapan batas maksimal diskon otomatis (*discount capping*) sebesar 20% menjadi strategi kunci mitigasi risiko finansial.

---

## Spesifikasi Teknis

* **Bahasa & Runtime**: Python 3.10 / 3.11 / 3.14
* **Algoritma Pemodelan**:
  * Task 1: K-Means Clustering (`scikit-learn`)
  * Task 2: Random Forest Classifier & Logistic Regression (`scikit-learn`)
* **Backend Framework**: FastAPI + Uvicorn + Pydantic
* **Frontend Framework**: Streamlit
* **Persistence Model**: Joblib

---

## Struktur Repositori

```text
.
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── schemas.py
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── model/
│   ├── customer_clustering_pipeline.joblib
│   └── order_profitability_pipeline.joblib
├── notebook/
│   ├── exploration_and_modelling.ipynb
│   ├── requirements.txt
│   └── superstore.sqlite
└── README.md

## Panduan Instalasi & Eksekusi

Pastikan sistem telah terpasang **Python 3.10+** dan **Git**. Disarankan menggunakan *virtual environment* agar dependensi antar modul tidak berbenturan.

---

### 1. Kloning Repositori

Buka terminal (PowerShell / Command Prompt / Bash) dan jalankan perintah berikut:

```bash
git clone [https://github.com/username/LnT-Camp-Final-Project.git](https://github.com/username/LnT-Camp-Final-Project.git)
cd LnT-Camp-Final-Project
```