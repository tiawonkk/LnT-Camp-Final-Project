import streamlit as st
import requests
import pandas as pd
import os


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="Superstore Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("📊 Superstore Retail Analytics & ML Platform")
st.markdown(
    "An end-to-end retail transaction analytics suite: Exploratory Data Analysis (EDA), "
    "customer behavioral profiling (Clustering), and real-time order profitability inference (Classification)."
)

# Sidebar: System Status & Controls
with st.sidebar:
    st.header("⚙️ Backend API Status")
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if res.status_code == 200:
            st.success("Backend API: Connected 🟢")
        else:
            st.error("Backend API: Error 🔴")
    except requests.exceptions.RequestException:
        st.error("Backend API: Disconnected 🔴")
        st.caption(f"Ensure the FastAPI server is running at `{BACKEND_URL}`.")

    st.markdown("---")
    st.subheader("📌 System Architecture")
    st.markdown(
        "- **Modeling:** K-Means & Random Forest\n"
        "- **Pipeline:** Scikit-Learn + Joblib\n"
        "- **Backend:** FastAPI\n"
        "- **Frontend:** Streamlit"
    )

# Navigation Tabs
tab_eda, tab_cluster, tab_class = st.tabs([
    "📈 Exploratory Data Analysis (EDA)",
    "🎯 Customer Segmentation (Clustering)",
    "💰 Order Profitability (Classification)"
])

# =========================================================
# TAB 1: EDA & BUSINESS INSIGHTS
# =========================================================
with tab_eda:
    st.subheader("Data Exploration & Key Business Metrics")
    st.markdown("High-level empirical findings extracted across 51,290 Superstore transactions.")

    # Aggregate KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Transactions", value="51,290")
    with kpi2:
        st.metric(label="Profitable Order Ratio", value="80.6%", delta="Profitable")
    with kpi3:
        st.metric(label="Unprofitable Order Ratio", value="19.4%", delta="-Unprofitable", delta_color="inverse")
    with kpi4:
        st.metric(label="Optimal Clusters (k)", value="3", help="Validated via Elbow Method & Silhouette Score")

    st.markdown("---")

    # Section 1: Clustering
    st.markdown("#### 1. Customer Segmentation Profiles (RFM)")
    st.write(
        "Validated by the **Elbow Method** and peak **Silhouette Score** ($k=3$), "
        "the customer base is partitioned into three distinct behavioral cohorts:"
    )

    cluster_data = {
        "Cluster": ["Cluster 1: High-Value Champions", "Cluster 0: Regular Spenders", "Cluster 2: At-Risk / Lapsed"],
        "Recency (Days)": ["< 400 days (Highly Active)", "< 400 days (Active)", "400 - 1,400+ days (Dormant)"],
        "Frequency (Orders)": ["> 6 - 18 orders", "1 - 7 orders", "1 - 5 orders"],
        "Monetary (Sales)": ["> $10,000 (Up to $25,000)", "< $5,000", "< $3,000"],
        "Business Strategy": ["VIP Loyalty & Priority Service", "Cross-selling & Upselling", "Win-Back Re-engagement Campaigns"]
    }
    st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 2: Classification
    st.markdown("#### 2. Order Profitability Determinants (Classification)")
    st.write(
        "Based on the *Feature Importance* analysis from the ensemble Random Forest model, "
        "the following factors dictate transaction-level profit margins:"
    )

    importance_data = {
        "Predictor Feature": ["Discount", "Discount Tier: No Discount", "Shipping Cost", "Sales", "Discount Tier: Medium", "Quantity"],
        "Importance Score": [0.315, 0.160, 0.122, 0.108, 0.089, 0.040],
        "Margin Impact": ["Extreme Negative", "Strong Positive", "Significant Negative", "Positive", "High Negative", "Neutral / Positive"]
    }
    st.dataframe(pd.DataFrame(importance_data), use_container_width=True, hide_index=True)

    st.info(
        "💡 **Operational Recommendation:** Markdown policy (`discount`) exerts the strongest downward pressure on gross margins. "
        "Enforcing an automated 20% discount cap on slim-margin categories is critical to curb the 19.4% share of unprofitable orders."
    )

# =========================================================
# TAB 2: CLUSTERING (RFM INFERENCE)
# =========================================================
with tab_cluster:
    st.subheader("New Customer Segment Inference")
    st.write("Input customer behavioral metrics to compute instant K-Means cohort assignments.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "Recency (Days since last purchase)",
            min_value=0.0,
            max_value=2000.0,
            value=35.0,
            step=1.0
        )
    with col2:
        frequency = st.number_input(
            "Frequency (Lifetime unique orders)",
            min_value=1,
            max_value=100,
            value=8,
            step=1
        )
    with col3:
        monetary = st.number_input(
            "Monetary (Cumulative spend in USD)",
            min_value=0.0,
            max_value=100000.0,
            value=4500.0,
            step=50.0
        )

    if st.button("Run Cluster Inference", type="primary", use_container_width=True):
        payload = {
            "recency": float(recency),
            "frequency": int(frequency),
            "monetary": float(monetary)
        }
        try:
            with st.spinner("Submitting request to FastAPI backend..."):
                response = requests.post(f"{BACKEND_URL}/predict/cluster", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success("Customer Segmentation Successful!")

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Cluster ID", f"Cluster {result['cluster_id']}")
                    st.info(f"**Segment Name:**\n\n{result['segment_name']}")
                with r2:
                    st.markdown("**Behavioral Profile:**")
                    st.write(result["description"])
                    st.warning(f"**Actionable Recommendation:**\n\n{result['actionable_recommendation']}")
            else:
                st.error(f"Inference execution failed: {response.text}")
        except Exception as e:
            st.error(f"Backend connection error: {str(e)}")

# =========================================================
# TAB 3: CLASSIFICATION (PROFITABILITY INFERENCE)
# =========================================================
with tab_class:
    st.subheader("Order Profitability Simulation & Prediction")
    st.write("Simulate order attributes to evaluate transaction profitability before processing.")

    c1, c2, c3 = st.columns(3)
    with c1:
        sales = st.number_input("Sales ($)", min_value=0.0, max_value=50000.0, value=250.0, step=10.0)
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2, step=1)
        discount = st.slider("Discount", min_value=0.0, max_value=0.9, value=0.1, step=0.05)

    with c2:
        shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, max_value=2000.0, value=15.0, step=1.0)
        delivery_duration = st.number_input("Delivery Duration (Days)", min_value=0, max_value=30, value=3, step=1)
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

        # Map default discount tier
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

    if st.button("Predict Profitability Margin", type="primary", use_container_width=True):
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
            with st.spinner("Evaluating order risk via ensemble model..."):
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
                    st.metric("Confidence Score (Profit Probability)", f"{result['profitability_probability'] * 100:.2f}%")
                with cr2:
                    st.info(f"**Managerial Recommendation:**\n\n{result['recommendation']}")
            else:
                st.error(f"Failed to parse backend response: {response.text}")
        except Exception as e:
            st.error(f"Backend connection error: {str(e)}")