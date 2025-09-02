#!/bin/bash

# SuperNova RunPod Deployment Script
# This script deploys and runs SuperNova on a RunPod GPU instance

echo "======================================"
echo "SuperNova RunPod Deployment"
echo "======================================"

# 1. Update system and install dependencies
echo "[1/6] Installing system dependencies..."
apt-get update
apt-get install -y git nano curl python3-pip

# 2. Clone the repository (if not already cloned)
if [ ! -d "/workspace/nova_signal" ]; then
    echo "[2/6] Cloning repository..."
    cd /workspace
    git clone https://github.com/YOUR_USERNAME/nova_signal.git
    cd nova_signal/supernova
else
    echo "[2/6] Repository already exists, pulling latest..."
    cd /workspace/nova_signal/supernova
    git pull
fi

# 3. Install Python dependencies
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate bitsandbytes
pip install fastapi uvicorn httpx python-dotenv
pip install pandas yfinance finnhub-python
pip install requests

# 4. Set up environment variables
echo "[4/6] Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Please edit .env file with your API keys"
    nano .env
fi

# 5. Start FinGPT server
echo "[5/6] Starting FinGPT server..."
cd FinGPT
nohup python3 supernova_fingpt_server.py > fingpt.log 2>&1 &
FINGPT_PID=$!
echo "FinGPT server started with PID: $FINGPT_PID"

# Wait for FinGPT to initialize
sleep 10

# 6. Start SuperNova backend
echo "[6/6] Starting SuperNova backend..."
cd ../backend
nohup python3 app.py > supernova.log 2>&1 &
SUPERNOVA_PID=$!
echo "SuperNova backend started with PID: $SUPERNOVA_PID"

# Wait for services to start
sleep 5

# Check health status
echo ""
echo "======================================"
echo "Checking service health..."
echo "======================================"

# Check FinGPT
echo -n "FinGPT Health: "
curl -s http://localhost:8080/health | python3 -m json.tool || echo "Not responding"

echo ""
echo -n "SuperNova Health: "
curl -s http://localhost:8081/health | python3 -m json.tool || echo "Not responding"

echo ""
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo "FinGPT Server: http://localhost:8080"
echo "SuperNova Backend: http://localhost:8081"
echo ""
echo "To monitor logs:"
echo "  FinGPT: tail -f /workspace/nova_signal/supernova/FinGPT/fingpt.log"
echo "  SuperNova: tail -f /workspace/nova_signal/supernova/backend/supernova.log"
echo ""
echo "To stop services:"
echo "  kill $FINGPT_PID $SUPERNOVA_PID"
echo "======================================"