# Omni-MRL / Omni-Lindy Kubernetes Scheduler HTTP Extender (v11.0) 경량 컨테이너 이미지
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 외부 의존성은 NumPy 단독으로 유지
RUN pip install --no-cache-dir "numpy>=2.0"

COPY simulator.py baselines.py workloads.py adaptive_lindy.py bigtech_baselines.py omni_mrl.py ./
COPY k8s_extender/ ./k8s_extender/

EXPOSE 8888

ENTRYPOINT ["python", "-m", "k8s_extender.server", "--host", "0.0.0.0", "--port", "8888", "--engine", "omni_mrl"]
