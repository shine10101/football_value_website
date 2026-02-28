#!/bin/bash
set -e

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Refreshing predictions..."
python manage.py refresh_predictions || echo "WARNING: Prediction refresh failed, continuing startup..."

echo "Starting Gunicorn..."
exec gunicorn --config gunicorn-cfg.py core.wsgi
