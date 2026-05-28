FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements-dev.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-dev.txt

COPY . .

CMD ["bash"]
