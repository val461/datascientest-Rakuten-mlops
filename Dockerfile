FROM python:3.12-slim

WORKDIR /app

# Copie uniquement les dépendances d'abord (pour le cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
