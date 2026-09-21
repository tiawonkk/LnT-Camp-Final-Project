import streamlit as st
import requests
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(
    page_title="Superstore Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000"

# Header Utama
st.title("📊 Superstore Retail Analytics & ML Platform")
st.markdown(
    "Solusi *end-to-end* untuk analisis transaksi ritel: visualisasi eksplorasi data (EDA), "
    "segmentasi profil pelanggan (Clustering), dan prediksi profitabilitas pesanan (Classification)."
)

# Sidebar: Status Sistem & Kontrol
with st.sidebar:
    st.header("⚙️ Status Backend API")
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if res.status_code == 200:
            st.success("Backend API: Terhubung 🟢")
        else:
            st.error("Backend API: Error 🔴")
    except requests.exceptions.RequestException:
        st.error("Backend API: Tidak Terhubung 🔴")
        st.caption("Pastikan server FastAPI telah dijalankan di `http://127.0.0.1:8000`.")

    st.markdown("---")
    st.subheader("📌 Arsitektur Sistem")
    st.markdown(
        "- **Modeling:** K-Means & Random Forest\n"
        "- **Pipeline:** Scikit-Learn + Joblib\n"
        "- **Backend:** FastAPI\n"
        "- **Frontend:** Streamlit"
    )

# Tab Navigasi
tab_eda, tab_cluster, tab_class = st.tabs([
    "📈 Exploratory Data Analysis (EDA)",
    "🎯 Customer Segmentation (Clustering)",
    "💰 Order Profitability (Classification)"
])

# =========================================================
# TAB 1: EDA & BUSINESS INSIGHTS
# =========================================================
with tab_eda:
    st.subheader("Eksplorasi Data & Metrik Bisnis Utama")
    st.markdown("Ringkasan temuan kunci dari 51.290 transaksi Superstore.")

    # KPI Metrik Agregat
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Transaksi", value="51,290")
    with kpi2:
        st.metric(label="Rasio Pesanan Untung", value="80.6%", delta="Profitable")
    with kpi3:
        st.metric(label="Rasio Pesanan Rugi", value="19.4%", delta="-Unprofitable", delta_color="inverse")
    with kpi4:
        st.metric(label="Optimal Cluster (k)", value="3", help="Berdasarkan Elbow Method & Silhouette Score")

    st.markdown("---")

    # Bagian 1: Clustering (Atas)
    st.markdown("#### 1. Karakteristik Segmentasi Pelanggan (RFM)")
    st.write(
        "Melalui evaluasi **Elbow Method** dan **Silhouette Score** ($k=3$), "
        "pelanggan terbagi menjadi tiga perilaku belanja yang tegas:"
    )

    cluster_data = {
        "Cluster": ["Cluster 1: High-Value Champions", "Cluster 0: Regular Spenders", "Cluster 2: At-Risk / Lapsed"],
        "Recency (Hari)": ["< 400 hari (Sangat Aktif)", "< 400 hari (Aktif)", "400 - 1.400+ hari (Lama)"],
        "Frequency (Order)": ["> 6 - 18 kali", "1 - 7 kali", "1 - 5 kali"],
        "Monetary (Sales)": ["> $10.000 (Hingga $25.000)", "< $5.000", "< $3.000"],
        "Strategi Bisnis": ["Loyalty VIP & Layanan Prioritas", "Cross-selling & Upselling", "Win-back Re-engagement Campaign"]
    }
    st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Bagian 2: Classification (Bawah)
    st.markdown("#### 2. Penentu Profitabilitas Transaksi (Classification)")
    st.write(
        "Berdasarkan analisis *Feature Importance* model ensemble Random Forest, "
        "berikut proporsi faktor yang menentukan apakah transaksi untung atau rugi:"
    )

    importance_data = {
        "Fitur Prediktor": ["Discount", "Discount Tier: No Discount", "Shipping Cost", "Sales", "Discount Tier: Medium", "Quantity"],
        "Importance Score": [0.315, 0.160, 0.122, 0.108, 0.089, 0.040],
        "Dampak Margin": ["Negatif Ekstrem", "Sangat Positif", "Negatif Signifikan", "Positif", "Negatif Tinggi", "Netral / Positif"]
    }
    st.dataframe(pd.DataFrame(importance_data), use_container_width=True, hide_index=True)

    st.info(
        "💡 **Rekomendasi Operasional:** Kebijakan pemotongan harga (`discount`) memiliki pengaruh paling dominan terhadap margin kotor. "
        "Pembatasan diskon maksimal 20% pada produk margin tipis sangat dianjurkan untuk menekan rasio 19.4% pesanan yang merugi."
    )
# =========================================================
# TAB 2: CLUSTERING (RFM INFERENCE)
# =========================================================
with tab_cluster:
    st.subheader("Prediksi Segmen Pelanggan Baru")
    st.write("Masukkan indikator transaksi pelanggan untuk menentukan penempatan segmen K-Means secara instan.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "Recency (Hari sejak transaksi terakhir)",
            min_value=0.0,
            max_value=2000.0,
            value=35.0,
            step=1.0
        )
    with col2:
        frequency = st.number_input(
            "Frequency (Jumlah riwayat order)",
            min_value=1,
            max_value=100,
            value=8,
            step=1
        )
    with col3:
        monetary = st.number_input(
            "Monetary (Total nilai belanja kumulatif / $)",
            min_value=0.0,
            max_value=100000.0,
            value=4500.0,
            step=50.0
        )

    if st.button("Jalankan Inferensi Klaster", type="primary", use_container_width=True):
        payload = {
            "recency": float(recency),
            "frequency": int(frequency),
            "monetary": float(monetary)
        }
        try:
            with st.spinner("Memproses ke backend FastAPI..."):
                response = requests.post(f"{BACKEND_URL}/predict/cluster", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediksi Segmentasi Berhasil!")

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Cluster ID", f"Cluster {result['cluster_id']}")
                    st.info(f"**Nama Segmen:**\n\n{result['segment_name']}")
                with r2:
                    st.markdown("**Deskripsi Perilaku:**")
                    st.write(result["description"])
                    st.warning(f"**Rekomendasi Tindakan Bisnis:**\n\n{result['actionable_recommendation']}")
            else:
                st.error(f"Gagal memproses prediksi: {response.text}")
        except Exception as e:
            st.error(f"Koneksi backend gagal: {str(e)}")

# =========================================================
# TAB 3: CLASSIFICATION (PROFITABILITY INFERENCE)
# =========================================================
with tab_class:
    st.subheader("Simulasi & Prediksi Profitabilitas Transaksi")
    st.write("Uji kombinasi order untuk memproyeksikan apakah order tersebut berisiko merugikan atau aman diproses.")

    c1, c2, c3 = st.columns(3)
    with c1:
        sales = st.number_input("Sales ($)", min_value=0.0, max_value=50000.0, value=250.0, step=10.0)
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2, step=1)
        discount = st.slider("Discount", min_value=0.0, max_value=0.9, value=0.1, step=0.05)

    with c2:
        shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, max_value=2000.0, value=15.0, step=1.0)
        delivery_duration = st.number_input("Delivery Duration (Hari)", min_value=0, max_value=30, value=3, step=1)
        ship_mode = st.selectbox(
            "Ship Mode",
            ["Standard Class", "Second Class", "First Class", "Same Day"]
        )

    with c3:
        order_priority = st.selectbox(
            "Order Priority",
            ["Low", "Medium", "High", "Critical"]
        )
        category = st.selectbox(
            "Category",
            ["Technology", "Furniture", "Office Supplies"]
        )

        # Mapping default tier diskon
        if discount == 0.0:
            default_tier_idx = 0
        elif discount <= 0.2:
            default_tier_idx = 1
        elif discount <= 0.5:
            default_tier_idx = 2
        else:
            default_tier_idx = 3

        discount_tier = st.selectbox(
            "Discount Tier",
            ["No Discount", "Low (<=20%)", "Medium (21-50%)", "High (>50%)"],
            index=default_tier_idx
        )

    if st.button("Prediksi Margin Keuntungan", type="primary", use_container_width=True):
        payload = {
            "sales": float(sales),
            "quantity": int(quantity),
            "discount": float(discount),
            "shipping_cost": float(shipping_cost),
            "delivery_duration": int(delivery_duration),
            "ship_mode": ship_mode,
            "order_priority": order_priority,
            "category": category,
            "discount_tier": discount_tier
        }
        try:
            with st.spinner("Menghitung estimasi klasifikasi..."):
                response = requests.post(f"{BACKEND_URL}/predict/profitability", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.markdown("---")

                cr1, cr2 = st.columns([1, 2])
                with cr1:
                    if result["is_profitable"] == 1:
                        st.success(f"### Status: {result['status']}")
                    else:
                        st.error(f"### Status: {result['status']}")
                    st.metric("Tingkat Keyakinan (Probabilitas Untung)", f"{result['profitability_probability'] * 100:.2f}%")
                with cr2:
                    st.info(f"**Rekomendasi Manajerial:**\n\n{result['recommendation']}")
            else:
                st.error(f"Gagal memproses respons backend: {response.text}")
        except Exception as e:
            st.error(f"Koneksi backend gagal: {str(e)}")