FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime@sha256:3c0b174a1339193a5bcc4b37d737c9f5ca3c4782a2293d3888e4271367eddf44
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 fonts-dejavu-core espeak-ng python3-venv git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements-mcp.txt /app/requirements-mcp.txt
RUN python3 -m venv --system-site-packages /opt/venv && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt -r requirements-mcp.txt
ENV PATH="/opt/venv/bin:${PATH}"
COPY . /app
EXPOSE 8080
CMD ["python", "server.py"]
