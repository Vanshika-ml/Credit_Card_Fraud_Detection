FROM python:3.11-slim
 
WORKDIR /app
 
# Install dependencies first (better layer caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
 
# Copy app code and model artifacts
COPY main.py .
COPY fraud_model.pkl .
COPY columns.pkl .
COPY encoders.pkl .
 
EXPOSE 8000
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
 
