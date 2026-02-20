#!/bin/bash
set -e

# ============================================================
# VPS Setup Script for value-ator.org
# Run this once on a fresh Ubuntu VPS as root:
#   ssh root@139.59.188.187
#   bash setup-server.sh
# ============================================================

DOMAIN="value-ator.org"
REPO="https://github.com/shine10101/football_value_website.git"
EMAIL=""

echo "==========================================="
echo "  Server Setup for $DOMAIN"
echo "==========================================="

# Prompt for email (needed for SSL certs)
read -p "Enter your email (for Let's Encrypt SSL): " EMAIL
if [ -z "$EMAIL" ]; then
    echo "Email is required for SSL certificates. Exiting."
    exit 1
fi

# ----- Step 1: System updates & Docker -----
echo ""
echo "[1/6] Installing Docker..."
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# ----- Step 2: Create shared Docker network -----
echo ""
echo "[2/6] Creating traefik-public network..."
docker network create traefik-public 2>/dev/null || echo "Network already exists"

# ----- Step 3: Set up Traefik -----
echo ""
echo "[3/6] Setting up Traefik reverse proxy..."
mkdir -p ~/traefik
cat > ~/traefik/docker-compose.yml << 'TRAEFIKEOF'
services:
  traefik:
    image: traefik:v3.3
    container_name: traefik
    restart: always
    command:
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--providers.docker.network=traefik-public"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--entrypoints.web.http.redirections.entrypoint.to=websecure"
      - "--entrypoints.web.http.redirections.entrypoint.scheme=https"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.letsencrypt.acme.email=${ACME_EMAIL}"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - traefik-certs:/letsencrypt
    networks:
      - traefik-public

volumes:
  traefik-certs:

networks:
  traefik-public:
    external: true
TRAEFIKEOF

cat > ~/traefik/.env << EOF
ACME_EMAIL=$EMAIL
EOF

cd ~/traefik
docker compose up -d
echo "Traefik is running."

# ----- Step 4: Clone the project -----
echo ""
echo "[4/6] Cloning project..."
cd ~
if [ -d "football_value_website" ]; then
    echo "Project directory already exists, pulling latest..."
    cd football_value_website && git pull
else
    git clone "$REPO"
    cd football_value_website
fi

# ----- Step 5: Generate .env -----
echo ""
echo "[5/6] Creating production .env..."
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")

cat > .env << EOF
DEBUG=False
SECRET_KEY=$SECRET_KEY
DOMAIN=$DOMAIN
DATA_DIR=/app/data
PREDICTIONS_CSV=predictions.csv
EOF

echo "Generated .env with a secure SECRET_KEY."

# ----- Step 6: Build & start the app -----
echo ""
echo "[6/6] Building and starting the app..."
docker compose up -d --build

echo ""
echo "==========================================="
echo "  DONE! Your site is deploying."
echo "==========================================="
echo ""
echo "  URL:  https://$DOMAIN"
echo ""
echo "  It may take 1-2 minutes for the SSL"
echo "  certificate to be issued on first start."
echo ""
echo "  To create a Django admin user, run:"
echo "    docker exec -it football_value python manage.py createsuperuser"
echo ""
echo "  To check logs:"
echo "    docker compose logs -f"
echo ""
