from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    message: str

# --- Clustering Schemas ---
class ClusteringInput(BaseModel):
    recency: float = Field(..., example=30.0, description="Jumlah hari sejak transaksi terakhir")
    frequency: int = Field(..., example=8, description="Total frekuensi pemesanan")
    monetary: float = Field(..., example=4500.0, description="Total nilai transaksi/sales")

class ClusteringOutput(BaseModel):
    cluster_id: int
    segment_name: str
    description: str
    actionable_recommendation: str

# --- Classification Schemas ---
class ClassificationInput(BaseModel):
    sales: float = Field(..., example=250.0, description="Nilai penjualan kotor")
    quantity: int = Field(..., example=2, description="Jumlah item produk")
    discount: float = Field(..., example=0.1, description="Besaran diskon (0.0 - 1.0)")
    shipping_cost: float = Field(..., example=15.0, description="Biaya pengiriman")
    delivery_duration: int = Field(..., example=3, description="Durasi pengiriman dalam hari")
    ship_mode: str = Field(..., example="Standard Class", description="Metode pengiriman")
    order_priority: str = Field(..., example="Medium", description="Prioritas order")
    category: str = Field(..., example="Office Supplies", description="Kategori produk")
    discount_tier: str = Field(..., example="Low (<=20%)", description="Tier diskon")

class ClassificationOutput(BaseModel):
    is_profitable: int
    status: str
    profitability_probability: float
    recommendation: str