FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Persistent data directory (for SQLite DB + predictions CSV)
RUN mkdir -p /app/data

EXPOSE 5005

ENTRYPOINT ["./entrypoint.sh"]
