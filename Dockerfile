
FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir --default-timeout=120 -r requirements-deploy.txt

COPY .streamlit/ .streamlit/
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
