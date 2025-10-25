# ===== STAGE 1: Build =====
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ===== STAGE 2: Runtime =====
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY . .

# RUN pip install grpcio-tools && \
#     python -m grpc_tools.protoc -I ../contracts --python_out=. --grpc_python_out=. ../contracts/calc.proto

EXPOSE 50051

CMD ["python", "server.py"]