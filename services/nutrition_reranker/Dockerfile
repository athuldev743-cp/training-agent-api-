FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    ORT_INTRA_OP_THREADS=1 \
    ORT_INTER_OP_THREADS=1

WORKDIR /app
COPY services/nutrition_reranker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/nutrition_reranker/app.py ./app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
