FROM python:3.11-slim
WORKDIR /workspace
ENV PYTHONPATH=/workspace PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir paho-mqtt==2.1.0
COPY app /workspace/app
