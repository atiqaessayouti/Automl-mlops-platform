from contextlib import asynccontextmanager
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import load_pipeline, predict_churn


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()
    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API MLOps de prédiction du risque de résiliation client (Champion : SVM)",
    version="1.0.0",
    lifespan=lifespan,
)


class ClientData(BaseModel):
    gender: str = Field(default="Female")
    SeniorCitizen: int = Field(default=0)
    Partner: str = Field(default="No")
    Dependents: str = Field(default="No")
    tenure: int = Field(default=2)
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="No")
    StreamingMovies: str = Field(default="No")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=70.70)
    TotalCharges: float = Field(default=151.65)


@app.get("/")
def root():
    return {"message": "API Telco Churn opérationnelle. Ouvre /docs pour tester."}


@app.get("/health")
def health_check():
    artifact = load_pipeline()
    return {
        "status": "healthy",
        "model": artifact["model_name"],
        "threshold": artifact["threshold"],
    }


@app.post("/predict")
def predict(client: ClientData):
    try:
        return predict_churn(client.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))