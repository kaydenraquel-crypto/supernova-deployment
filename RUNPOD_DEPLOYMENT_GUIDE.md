# RunPod Deployment Guide for SuperNova + FinGPT

## ✅ Status
- **Claude API**: ✅ WORKING (Fixed message format issue)
- **OpenAI API**: ⚠️ 404 Error (May need different model or API key validation)
- **FinGPT Server**: ✅ WORKING (Fixed tensor device issue)
- **SuperNova Backend**: ✅ WORKING

## 🚀 Quick Deploy to RunPod

### Step 1: SSH into RunPod
```bash
ssh root@your-runpod-ip
```

### Step 2: Clone Repository
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/nova_signal.git
cd nova_signal/supernova
```

### Step 3: Install Dependencies
```bash
# System packages
apt-get update
apt-get install -y nano curl git python3-pip

# Python packages
pip install --upgrade pip
pip install torch transformers peft accelerate bitsandbytes
pip install fastapi uvicorn httpx python-dotenv
pip install pandas yfinance finnhub-python requests
```

### Step 4: Configure Environment
```bash
# Create .env file
nano .env
```

Add the following (with your API keys):
```env
# Model config
LLM_PROVIDER=auto  # Will try Claude first, then OpenAI
ANTHROPIC_API_KEY=your-claude-key-here
ANTHROPIC_MODEL=claude-3-haiku-20240307
OPENAI_API_KEY=your-openai-key-here
OPENAI_MODEL=gpt-4-0125-preview
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.2

# FinGPT configuration
FINGPT_MODE=local
FINGPT_LOCAL_PORT=8080
FINGPT_BASE_URL=http://localhost:8080
FINGPT_GPU_ENABLED=true

# Backend settings
SUPERNOVA_PORT=8083
PORT=8081
```

### Step 5: Start Services

#### Option A: Using the startup script
```bash
python3 runpod_start.py
```

#### Option B: Manual startup
```bash
# Terminal 1 - Start FinGPT
cd /workspace/nova_signal/supernova/FinGPT
python3 supernova_fingpt_server.py

# Terminal 2 - Start SuperNova Backend
cd /workspace/nova_signal/supernova/backend
python3 app.py
```

### Step 6: Verify Services
```bash
# Check FinGPT
curl http://localhost:8080/health

# Check SuperNova
curl http://localhost:8083/health

# Test full system
cd /workspace/nova_signal/supernova
python3 test_simple.py
```

## 🔧 Fixed Issues

### 1. Claude API Message Format
**Problem**: tool_result blocks without corresponding tool_use blocks
**Solution**: Added assistant message with tool_use before user message with tool_result

### 2. FinGPT Tensor Device Mismatch
**Problem**: Tensors on different devices (CPU vs CUDA)
**Solution**: Move inputs to CUDA when GPU is enabled

### 3. Multiple .env Files
**Problem**: Backend loading wrong .env file
**Solution**: Update `/supernova/.env` not root `.env`

## 🎯 API Configuration

### Claude (Anthropic)
- **Working**: ✅ Fixed message format
- **Model**: claude-3-haiku-20240307
- **Note**: May hit rate limits (50k tokens/min)

### OpenAI
- **Status**: ⚠️ Getting 404 errors
- **Possible fixes**:
  - Try model: `gpt-3.5-turbo` or `gpt-4`
  - Verify API key at https://platform.openai.com/api-keys
  - Check organization/project settings

## 📊 Testing

### Simple Test (No Tools)
```python
payload = {
    'symbols': ['AAPL'],
    'timeframe': '1d',
    'tasks': [],
    'query': 'What is Apple stock?'
}
```

### Full Test (With Tools)
```python
payload = {
    'symbols': ['AAPL'],
    'timeframe': '1d',
    'tasks': ['forecast', 'news'],
    'query': 'Analyze Apple stock and provide forecast'
}
```

## 🚨 Troubleshooting

### Port Conflicts
```bash
# Kill existing processes
pkill -f "supernova_fingpt_server"
pkill -f "app.py"
pkill -f "uvicorn"
```

### GPU Issues
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

### API Key Issues
- Claude: Verify at https://console.anthropic.com/
- OpenAI: Verify at https://platform.openai.com/api-keys

## 📱 Access from Outside RunPod

If you want to access the services externally:

1. **RunPod Port Forwarding**:
   - Add ports 8080 and 8083 in RunPod console
   
2. **Use ngrok** (if allowed):
```bash
# Install ngrok
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip

# Expose SuperNova
./ngrok http 8083
```

## ✅ Success Checklist

- [ ] RunPod GPU instance running
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] .env configured with API keys
- [ ] FinGPT server running (port 8080)
- [ ] SuperNova backend running (port 8083)
- [ ] Health checks passing
- [ ] Test requests working

## 📝 Notes

- The system will automatically fallback between Claude and OpenAI
- Claude is preferred but may hit rate limits
- OpenAI needs valid API key and correct model name
- FinGPT runs locally and doesn't need API keys
- All logs are in `logs/supernova.log`

---

**Ready for deployment!** The system is tested and working with Claude API. OpenAI can be configured as backup once you verify the API key.