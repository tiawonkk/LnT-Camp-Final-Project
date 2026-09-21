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

### 1. Clone the Repository
Open your terminal (PowerShell / Command Prompt / Bash) and run:

```bash
git clone [https://github.com/username/LnT-Camp-Final-Project.git](https://github.com/username/LnT-Camp-Final-Project.git)
cd LnT-Camp-Final-Project
```

### 2. Model Exploration & Notebook
To inspect or re-run the end-to-end data processing and model training pipeline:

```bash
cd notebook
pip install -r requirements.txt
jupyter notebook exploration_and_modelling.ipynb
```
Trained pipeline artifacts will automatically export to the ```model/``` folder (```customer_clustering_pipeline.joblib``` and ```order_profitability_pipeline.joblib```).

### 3. Run the Backend API (FastAPI)
The backend serves model inference without retraining on startup.

1. Open a terminal and navigate to the ```backend``` folder:
```bash
cd backend
```

2. Install backend dependencies:
```bash
pip install -r requirement.txt
```

3. Start the Uvicorn server:
```bash
python -m uvicorn app:app --reload --port 8000
```

4. Access endpoints:
  - Base URL: http://127.0.0.1:8000
  - Health Check: http://127.0.0.1:8000/health
  - Interactive Swagger UI Docs: http://127.0.0.1:8000/docs

### 4. Run the Frontend Dashboard (Streamlit)
The frontend provides an interactive UI for exploring data metrics, inferring customer clusters, and simulating order profitability.

1. Open a separate terminal (keeping the backend running) and navigate to the frontend folder:
```bash
cd frontend
```

2. Install UI dependencies:
```bash
pip install -r requirement.txt
```

3. Start the Streamlit application:
```bash
python -m streamlit run app.py
```

4. Access the web dashboard at: http://localhost:8501

## API Documentation
Models are loaded directly from the model/ directory upon application startup.

### 1. Health Check
  - Endpoint: ```GET /health```
  - Response Schema:
  ```json
  {
  "status": "healthy",
  "message": "Backend service dan model inference siap beroperasi."
  }
  ```
### 2. Customer Segmentation
  - Endpoint: ```POST /predict/cluster```
  - Request Schema:
  ```json
  {
  "recency": 35.0,
  "frequency": 8,
  "monetary": 4500.0
  }
  ```

  - Response Schema:
  ```json
  {
  "cluster_id": 1,
  "segment_name": "High-Value Champions",
  "description": "Pelanggan prioritas dengan frekuensi belanja tinggi dan kontribusi sales terbesar.",
  "actionable_recommendation": "Berikan loyalty program VIP, reward eksklusif, dan early access katalog baru."
  }
  ```
### 3. Order Profitability Prediction
  - Endpoint: POST /predict/profitability
  - Request Schema:
  ```json
  {
  "sales": 250.0,
  "quantity": 2,
  "discount": 0.1,
  "shipping_cost": 15.0,
  "delivery_duration": 3,
  "ship_mode": "Standard Class",
  "order_priority": "Medium",
  "category": "Office Supplies",
  "discount_tier": "Low (<=20%)"
  }
  ```
  - Response Schema:
  ```json
  {
  "is_profitable": 1,
  "status": "Profitable",
  "profitability_probability": 0.9421,
  "recommendation": "Order aman diproses. Margin kotor diproyeksikan mampu menutup shipping cost."
  }
  ```

## Troubleshooting Tips
- Command Not Found (uvicorn or streamlit):
  If PowerShell does not recognize global commands, invoke them directly via the active Python module:
  - Backend: ```python -m uvicorn app:app --reload --port 8000```
  - Frontend: ```python -m streamlit run app.py```
- Backend Connection Error in Streamlit:
Ensure the backend terminal displays ```Application startup complete on``` port ```8000``` before triggering predictions on the dashboard.

Project Links
- Deployed Frontend: [Link pending]
- Deployed Backend API: [Link pending]
- GitHub Repository: [Link pending]
- LinkedIn Post: [Link pending]