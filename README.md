# Global Superstore Intelligence: Customer Segmentation & Profitability Prediction

Final Project - LnT Camp 2026  
Theme: "Bridging the Gap: Empowering Future Talent through Machine Learning for Industry Innovation"

Author: Christian Immanuel Valerio - Tech Enthusiast

---

## Project Overview

### 1. Situation
In global retail operations, businesses struggle with logistics cost efficiency and an influx of unprofitable orders caused by excessive discounting and eroded margins. The raw transaction records are stored across multi-table relational structures in an SQLite database (`superstore.sqlite`). An end-to-end analytical system is required—spanning database extraction, customer behavioral segmentation, and real-time machine learning inference via an interactive user interface.

### 2. Task
This project implements two core Machine Learning tasks:
* **Customer Segmentation (Clustering)**: Grouping customers based on RFM (Recency, Frequency, Monetary) metrics without historical labels to drive targeted marketing and retention strategies.
* **Order Profitability Prediction (Classification)**: Predicting whether a newly configured order will yield a profit or incur a loss before processing at the operational level.

### 3. Action
* **Data Extraction & Preprocessing**:
  * Programmatically extracted records from `superstore.sqlite` using `sqlite3` and `pandas`.
  * Feature engineering: Delivery turnaround (`delivery_duration`), discount brackets (`discount_tier`), and aggregated customer-level RFM metrics.
* **Machine Learning Pipeline**:
  * **Clustering**: Optimized **K-Means** using the **Elbow Method (Inertia)** and validated with the peak **Silhouette Score** ($k=3$).
  * **Classification**: Benchmarked **Logistic Regression** against a **Random Forest Classifier** wrapped in a `ColumnTransformer` pipeline (numeric standard scaling and categorical One-Hot Encoding).
* **Deployment & Serving**:
  * Serialized trained pipeline artifacts into `.joblib` format inside the `model/` directory.
  * Developed a modular REST API using **FastAPI** with health monitoring (`/health`) and inference endpoints (`/predict/cluster` and `/predict/profitability`).
  * Built an interactive analytics dashboard with **Streamlit** displaying EDA metrics, customer segmentation inference, and profit simulation fully decoupled from model training logic.

### 4. Result
* **Clustering Evaluation**: Successfully segmented 51,290 transactions into 3 distinct behavioral cohorts: *High-Value Champions*, *Active Moderate Spenders*, and *At-Risk Customers*.
* **Classification Evaluation**: The Random Forest model achieved an $F1\text{-score} > 0.94$, accurately distinguishing profitable orders from margin-negative transactions.
* **Key Business Insights**:
  * The `discount` variable is the single strongest driver of negative margins (*Feature Importance* > 0.31), followed by freight overhead (`shipping_cost`).
  * Enforcing an automated discount cap of 20% on slim-margin categories serves as the primary financial safeguard against operational losses.

---

## Technical Specifications

* **Language & Runtime**: Python 3.10 / 3.11 / 3.14
* **Modeling Algorithms**:
  * Task 1: K-Means Clustering (`scikit-learn`)
  * Task 2: Random Forest Classifier & Logistic Regression (`scikit-learn`)
* **Backend Framework**: FastAPI + Uvicorn + Pydantic
* **Frontend Framework**: Streamlit
* **Model Persistence**: Joblib

---

## Repository Structure

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
```

--- 

## Installation & Execution Guide
Ensure Python 3.10+ and Git are installed on your system. Using a virtual environment is recommended to prevent dependency conflicts.

1. Clone the Repository
Open your terminal (PowerShell / Command Prompt / Bash) and run:

```bash
git clone [https://github.com/username/LnT-Camp-Final-Project.git](https://github.com/username/LnT-Camp-Final-Project.git)
cd LnT-Camp-Final-Project
```