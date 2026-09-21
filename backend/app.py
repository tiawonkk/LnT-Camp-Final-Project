import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    HealthResponse,
    ClusteringInput,
    ClusteringOutput,
    ClassificationInput,
    ClassificationOutput,
)

app = FastAPI(
    title="Retail Analytics Backend API",
    description="API Inference untuk Customer Segmentation dan Order Profitability Prediction",
    version="1.0.0"
)

# Setup CORS agar Frontend dapat melakukan HTTP request tanpa terblokir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolusi path absolut ke direktori model/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

CLUSTERING_MODEL_PATH = os.path.join(MODEL_DIR, "customer_clustering_pipeline.joblib")
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "order_profitability_pipeline.joblib")

# Load model saat startup (tanpa retraining)
try:
    clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    print("Berhasil memuat customer_clustering_pipeline.joblib dan order_profitability_pipeline.joblib")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model dari {MODEL_DIR}: {e}")

# Mapping profil bisnis untuk segmen K-Means
CLUSTER_PROFILES = {
    0: {
        "segment_name": "Active Moderate Spenders",
        "description": "Pelanggan aktif berbelanja moderat dengan frekuensi berkala.",
        "actionable": "Terapkan rekomendasi produk komplementer (cross-selling) untuk menaikkan basket size."
    },
    1: {
        "segment_name": "High-Value Champions",
        "description": "Pelanggan prioritas dengan frekuensi belanja tinggi dan kontribusi sales terbesar.",
        "actionable": "Berikan loyalty program VIP, reward eksklusif, dan early access katalog baru."
    },
    2: {
        "segment_name": "At-Risk / Lapsed Customers",
        "description": "Pelanggan dengan jeda transaksi sangat lama dan nilai moneter rendah.",
        "actionable": "Jalankan win-back email campaign dengan voucher diskon bersyarat min. belanja."
    }
}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "message": "Backend service dan model inference siap beroperasi."
    }


@app.post("/predict/cluster", response_model=ClusteringOutput, tags=["Inference"])
def predict_cluster(payload: ClusteringInput):
    try:
        input_data = pd.DataFrame([{
            "recency": payload.recency,
            "frequency": payload.frequency,
            "monetary": payload.monetary
        }])
        
        pred_cluster = int(clustering_model.predict(input_data)[0])
        profile = CLUSTER_PROFILES.get(pred_cluster, {
            "segment_name": "Unknown",
            "description": "Segmen tidak teridentifikasi.",
            "actionable": "Lakukan analisis manual."
        })
        
        return {
            "cluster_id": pred_cluster,
            "segment_name": profile["segment_name"],
            "description": profile["description"],
            "actionable_recommendation": profile["actionable"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error pada clustering: {str(e)}")


@app.post("/predict/profitability", response_model=ClassificationOutput, tags=["Inference"])
def predict_profitability(payload: ClassificationInput):
    try:
        input_data = pd.DataFrame([payload.model_dump()])
        
        prediction = int(classification_model.predict(input_data)[0])
        probabilities = classification_model.predict_proba(input_data)[0]
        profit_prob = float(probabilities[1])
        
        if prediction == 1:
            status = "Profitable"
            rec = "Order aman diproses. Margin kotor diproyeksikan mampu menutup shipping cost."
        else:
            status = "Not Profitable"
            rec = "Peringatan kerugian: Rasio diskon atau shipping cost terlalu tinggi relatif terhadap nilai sales."
            
        return {
            "is_profitable": prediction,
            "status": status,
            "profitability_probability": round(profit_prob, 4),
            "recommendation": rec
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error pada classification: {str(e)}")