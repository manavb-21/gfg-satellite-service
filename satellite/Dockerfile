FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    libgeos-dev \
    libproj-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
