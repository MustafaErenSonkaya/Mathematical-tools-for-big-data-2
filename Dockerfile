FROM python:3.13.1-slim-bookworm
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY cmapss.py injector.py pipeline.py params.yaml config.yaml calibration.json ./
COPY detectors ./detectors
COPY app ./app
COPY data/train_FD001.txt ./data/train_FD001.txt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
