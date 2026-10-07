FROM python:3.14-slim

WORKDIR /app

# Times are stored as naive local time and handed to the assistant as such, so
# the container has to run on the users' clock. Slim images ship without the
# zone database, and without it TZ silently falls back to UTC.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Ho_Chi_Minh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# auto = CUDA build if nvidia-smi works on the build host, else CPU.
# Force with: docker build --build-arg TORCH_VARIANT=cu126 .
ARG TORCH_VARIANT=auto
COPY scripts/install_torch.sh scripts/
RUN TORCH_VARIANT=$TORCH_VARIANT sh scripts/install_torch.sh

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
