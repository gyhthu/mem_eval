FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt requirements-full.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-full.txt && \
    python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-zh-v1.5')"

COPY src /app/src
COPY results/feishu_generator/README.md /app/results/feishu_generator/README.md
COPY results/feishu_generator/v1 /app/results/feishu_generator/v1
COPY results/feishu_generator/v2 /app/results/feishu_generator/v2
COPY results/eval_platform/runs /app/results/eval_platform/runs

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "src/eval_platform/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
