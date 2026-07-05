FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download fr_core_news_sm

FROM base AS runtime

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
