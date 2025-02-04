FROM python:3.10.16

# Nastavíme pracovní adresář v kontejneru
WORKDIR /app

# Zkopírujeme soubory aplikace do kontejneru
COPY . .

# Nainstalujeme závislosti
RUN pip install --no-cache-dir -r requirements.txt

# Exponujeme port 8080 (Google Cloud Run používá tento port)
EXPOSE 8080

# Spustíme aplikaci pomocí Gunicornu
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "opp:app"]