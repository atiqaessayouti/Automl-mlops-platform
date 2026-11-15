from fastapi import FastAPI

app = FastAPI(
    title="AutoML MLOps Platform API",
    description="API for the AutoML prediction platform",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "AutoML MLOps Platform API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "automl-api"
    }