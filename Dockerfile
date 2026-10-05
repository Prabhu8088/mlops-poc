FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["sh", "-c", "python generate.py && python preprocess.py && python train.py && python evaluate.py"]