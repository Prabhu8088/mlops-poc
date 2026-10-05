FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN set -ex; \
  pip install --no-cache-dir -r requirements.txt && \
  mkdir -p /app/data /app/model;

COPY src/ ./src/

CMD ["sh", "-c", "python src/generate.py && python src/preprocess.py && python src/train.py && python src/evaluate.py"]