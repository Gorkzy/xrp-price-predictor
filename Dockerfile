FROM python:3.10-slim

# Zajistíme, aby Python nebufroval (volitelné, ale doporučené)
ENV PYTHONUNBUFFERED=1

# Nastavíme pracovní adresář v kontejneru
WORKDIR /app

# Nainstalujeme potřebné systémové balíčky
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Zkopírujeme soubor s požadavky a nainstalujeme Python závislosti
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Zkopírujeme celý zdrojový kód do pracovního adresáře
COPY . .

# Nastavíme proměnnou PORT (Google Cloud Run ji očekává)
ENV PORT=8080

# Spustíme aplikaci pomocí gunicorn
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "full_app:app"]

# Nastavíme port, na kterém bude aplikace běžet
ENV PORT=8080

# Spustíme aplikaci pomocí gunicorn, kde entrypoint je "full_app:app"
CMD exec gunicorn --bind :$PORT --workers 1 full_app:app
# FROM python:3.10-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libatlas-base-dev \
#     libopenblas-dev \
#     liblapack-dev \
#     && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# COPY . .
# ENV PORT 8080
# CMD exec gunicorn --bind :$PORT --workers 1 full_app:app