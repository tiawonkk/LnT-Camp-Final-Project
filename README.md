# Global Superstore Intelligence: Customer Segmentation & Profitability Prediction

Final Project - LnT Camp 2026  
Tema: "Bridging the Gap: Empowering Future Talent through Machine Learning for Industry Innovation"

Author: Christian Immanuel Valerio - Tech Enthusiat

---

## Ringkasan Proyek

### 1. Situation
Dalam operasional bisnis retail global, perusahaan menghadapi tantangan dalam efisiensi biaya pemasaran dan tingginya transaksi yang merugikan akibat diskon serta biaya operasional yang tidak terukur. Data mentah tersimpan dalam database relasional SQLite yang terdiri dari beberapa tabel terpisah (orders, order_items, customers, products, locations). Diperlukan sistem machine learning terintegrasi dari database hingga antarmuka pengguna untuk memetakan perilaku pelanggan dan memitigasi pesanan berisiko rugi.

### 2. Task
Tugas utama yang diselesaikan dalam proyek ini mencakup 2 pemodelan Machine Learning:
* **Clustering (Customer Segmentation)**: Mengelompokkan pelanggan berdasarkan perilaku transaksi tanpa label sebelumnya untuk menentukan strategi retensi yang tepat.
* **Classification (Order Profitability)**: Memprediksi apakah suatu pesanan akan menghasilkan laba (Profitable) atau mengalami kerugian (Unprofitable) sebelum transaksi diproses.

### 3. Action 
* **Data Extraction**: Menghubungkan dan memuat database SQLite secara terprogram menggunakan `sqlite3` dan `pandas` melalui query SQL JOIN multi-tabel.
* **Exploratory Data Analysis (EDA) & Preprocessing**:
  * Menganalisis distribusi data, korelasi diskon terhadap laba, dan missing values.
  * Rekayasa fitur: Durasi pengiriman (ship_date - order_date), profit margin, discount tier, serta agregasi fitur RFM (Recency, Frequency, Monetary).
* **Modeling**:
  * **Clustering**: Menggunakan algoritma **K-Means** yang dioptimasi dengan analisis **Elbow Method (Inertia)** dan validasi **Silhouette Score**.
  * **Classification**: Menggunakan algoritma **Random Forest Classifier** dengan penanganan class imbalance (`class_weight='balanced'`).
* **Deployment & Integration**:
  * Mengekspor pipeline model terlatih ke format `.joblib` di folder `model/`.
  * Membangun REST API menggunakan **FastAPI** dengan endpoint inferensi `/predict/segmentation` dan `/predict/profitability`.
  * Membangun antarmuka simulasi interaktif berbasis **Streamlit** yang terhubung langsung ke backend API.

### 4. Result 
* **Hasil Evaluasi**:
  * Model clustering berhasil memetakan segmentasi pelanggan ke dalam kelompok yang jelas (High-Value Loyalists, Regular Buyers, dan Discount Seekers) dibuktikan dengan Silhouette Score yang optimal.
  * Model klasifikasi berhasil mendeteksi transaksi merugi dengan evaluasi komprehensif pada metrik Accuracy, Precision, Recall, F1-Score, dan ROC-AUC.
* **Kesimpulan Bisnis**:
  * Pemberian diskon di atas batas tertentu pada kategori produk spesifik merupakan kontributor terbesar terhadap pesanan merugi.
  * Sistem peringatan dini ini dapat digunakan tim sales dan operasional sebagai validasi kelayakan order secara otomatis.

---

## Spesifikasi Teknis

* **Bahasa & Runtime**: Python 3.10 / 3.11
* **Algoritma Machine Learning**:
  * Task 1 (Clustering): K-Means Clustering (`scikit-learn`)
  * Task 2 (Classification): Random Forest Classifier (`scikit-learn`)
* **Backend Framework**: FastAPI + Uvicorn
* **Frontend Framework**: Streamlit
* **Penyimpanan Model**: Joblib

---

## Tautan Proyek

* Deployed Frontend: [Tautan menyusul]
* Deployed Backend API: [Tautan menyusul]
* Repositori GitHub: [Tautan menyusul]
* Publikasi LinkedIn: [Tautan menyusul]

---

## Struktur Repositori

```text
.
├── notebook/
│   └── exploration_and_modelling.ipynb
├── model/
│   ├── customer_clustering_pipeline.joblib
│   └── order_profitability_model.joblib
├── backend/
│   ├── app.py
│   ├── schemas.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
└── README.md