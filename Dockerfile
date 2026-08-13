FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps kept minimal; faiss-cpu + torch ship as wheels.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501
# Default: launch the demo UI. Override to run ingestion:
#   docker run ... python main.py
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
