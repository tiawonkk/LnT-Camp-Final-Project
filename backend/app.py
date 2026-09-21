import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    ClusteringInput,
    ClusteringOutput,
    ClassificationInput,
    ClassificationOutput,
)

app = FastAPI(
    title="Retail Analytics Backend API",
    description="Inference API for Customer Segmentation and Order Profitability Prediction",
    version="1.0.0"
)

# Configure CORS to allow the frontend to execute HTTP requests without restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve absolute path to the model/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

CLUSTERING_MODEL_PATH = os.path.join(MODEL_DIR, "customer_clustering_pipeline.joblib")
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "order_profitability_pipeline.joblib")

# Load serialized pipelines at startup (decoupled from training)
try:
    clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    print("Successfully loaded customer_clustering_pipeline.joblib and order_profitability_pipeline.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts from {MODEL_DIR}: {e}")

# Business persona profiles for K-Means cohorts
CLUSTER_PROFILES = {
    0: {
        "segment_name": "Active Moderate Spenders",
        "description": "Active customers with consistent order frequency and moderate basket size.",
        "actionable": "Deploy cross-selling recommendations with complementary items to increase basket size."
    },
    1: {
        "segment_name": "High-Value Champions",
        "description": "Priority tier with the highest purchase frequency and primary revenue contribution.",
        "actionable": "Provide VIP loyalty rewards, exclusive perks, and early catalog access."
    },
    2: {
        "segment_name": "At-Risk / Lapsed Customers",
        "description": "Dormant accounts with prolonged inter-purchase dormancy and low monetary value.",
        "actionable": "Launch automated win-back email workflows with minimum-spend conditional vouchers."
    }
}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return {
        "status": "healthy",
        "message": "Backend service and model inference ready."
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
            "description": "Unidentified customer segment.",
            "actionable": "Manual review required."
        })
        
        return {
            "cluster_id": pred_cluster,
            "segment_name": profile["segment_name"],
            "description": profile["description"],
            "actionable_recommendation": profile["actionable"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering inference error: {str(e)}")


@app.post("/predict/profitability", response_model=ClassificationOutput, tags=["Inference"])
def predict_profitability(payload: ClassificationInput):
    try:
        input_data = pd.DataFrame([payload.model_dump()])
        
        prediction = int(classification_model.predict(input_data)[0])
        probabilities = classification_model.predict_proba(input_data)[0]
        profit_prob = float(probabilities[1])
        
        if prediction == 1:
            status = "Profitable"
            rec = "Order approved. Gross margins are projected to absorb fulfillment and shipping overhead."
        else:
            status = "Not Profitable"
            rec = "Loss alert: Discount rate or logistics overhead exceeds acceptable thresholds relative to order value."
            
        return {
            "is_profitable": prediction,
            "status": status,
            "profitability_probability": round(profit_prob, 4),
            "recommendation": rec
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification inference error: {str(e)}")