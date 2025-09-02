# SuperNova + FinGPT Deployment Repository

This repository contains the complete SuperNova trading analysis backend with FinGPT integration for deployment on RunPod GPU instances.

## 🚀 Quick Start on RunPod

### 1. Clone and Setup
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/supernova-deployment.git
cd supernova-deployment
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
nano .env
# Add your API keys
```

### 3. Start Services
```bash
python runpod_start.py
```

## 📁 Repository Structure

```
supernova-deployment/
├── backend/                  # SuperNova backend API
│   ├── app.py               # Main FastAPI application
│   ├── core/                # Core functionality
│   ├── llm/                 # LLM adapters (Claude, OpenAI)
│   ├── routes/              # API endpoints
│   ├── services/            # Service integrations
│   └── tools/               # Analysis tools
├── FinGPT/                  # FinGPT server
│   ├── supernova_fingpt_server.py  # Main FinGPT server
│   └── test_fingpt_client.py       # Test client
├── scripts/                 # Utility scripts
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── runpod_start.py        # Startup script
└── RUNPOD_DEPLOYMENT_GUIDE.md  # Detailed guide
```

## 🔧 Key Components

### SuperNova Backend (Port 8083)
- FastAPI-based REST API
- Orchestrates analysis using multiple tools
- Integrates with Claude or OpenAI for intelligence
- Provides market data, indicators, and forecasting

### FinGPT Server (Port 8080)
- Local financial LLM based on Llama-2
- Provides financial analysis and forecasting
- GPU-accelerated for fast inference
- No API keys required

## 🔑 Environment Variables

Required in `.env`:
```env
# LLM Configuration
LLM_PROVIDER=auto  # auto, anthropic, or openai
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# Service Ports
FINGPT_LOCAL_PORT=8080
SUPERNOVA_PORT=8083

# GPU Settings
FINGPT_GPU_ENABLED=true
```

## 📊 API Endpoints

### Health Check
- `GET http://localhost:8080/health` - FinGPT health
- `GET http://localhost:8083/health` - SuperNova health

### Analysis
- `POST http://localhost:8083/analyze` - Main analysis endpoint

Example request:
```json
{
  "symbols": ["AAPL"],
  "timeframe": "1d",
  "tasks": ["forecast", "news"],
  "query": "Analyze Apple stock"
}
```

## 🐛 Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Port Already in Use
```bash
pkill -f "supernova_fingpt_server"
pkill -f "app.py"
```

### API Key Issues
- Claude: Check at https://console.anthropic.com/
- OpenAI: Check at https://platform.openai.com/

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Support

For issues or questions, please open an issue on GitHub.