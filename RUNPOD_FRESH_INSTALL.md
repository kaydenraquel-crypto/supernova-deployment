# RunPod Fresh Installation Guide

## Step 1: Clean the GPU VM

```bash
# Remove old installation
cd /workspace
rm -rf nova_signal
rm -rf supernova-deployment

# Clean Python cache
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface

# Check available space
df -h /workspace
```

## Step 2: Clone the Updated Repository

```bash
# Clone the repository
cd /workspace
git clone https://github.com/kaydenraquel-crypto/supernova-deployment.git
cd supernova-deployment
```

## Step 3: Install PyTorch with CUDA Support

```bash
# Uninstall existing PyTorch if any
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## Step 4: Install Dependencies

```bash
# Use minimal requirements to avoid package conflicts
pip install -r requirements_minimal.txt

# Install additional required packages
pip install pydantic-settings
```

## Step 5: Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit the .env file and add your API keys
nano .env
```

Add these essential keys to your .env file:
```
LLM_PROVIDER=openai  # or anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here
```

## Step 6: Export HuggingFace Token

```bash
# Export HF_TOKEN for model download
export HF_TOKEN=your_huggingface_token_here
```

## Step 7: Start Services

### Option A: Use the startup script
```bash
# Fix the hardcoded paths in runpod_start.py first
sed -i 's|/workspace/nova_signal/supernova|/workspace/supernova-deployment|g' runpod_start.py

# Start both services
python runpod_start.py
```

### Option B: Start services manually
```bash
# Terminal 1: Start FinGPT
cd /workspace/supernova-deployment/FinGPT
export HF_TOKEN=your_huggingface_token_here
python supernova_fingpt_server.py

# Terminal 2: Start SuperNova Backend
cd /workspace/supernova-deployment/backend
python app.py
```

## Step 8: Verify Services

```bash
# Check SuperNova health
curl http://localhost:8083/health

# Check FinGPT health
curl http://localhost:8080/health

# Test analysis endpoint
curl -X POST http://localhost:8083/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "query": "What is the current price?"}'
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Port Already in Use
```bash
# Find process using port
lsof -i :8080
lsof -i :8083

# Kill process if needed
kill -9 <PID>
```

### API Rate Limits
- Switch between `anthropic` and `openai` in the .env file
- Claude API rate limit: 50,000 tokens/minute
- OpenAI has different rate limits based on your tier

### FinGPT Model Not Loading
```bash
# Ensure HF_TOKEN is set
echo $HF_TOKEN

# Check model download location
ls -la /workspace/supernova-deployment/FinGPT/models/
```

## Important Notes

1. **Storage Location**: Use `/workspace` which is persistent network storage
2. **GPU Memory**: The FinGPT model requires ~13GB VRAM
3. **API Keys**: Never commit API keys to the repository
4. **Ports**: 
   - SuperNova: 8083
   - FinGPT: 8080

## Support

If you encounter issues:
1. Check the logs in `/workspace/supernova-deployment/logs/`
2. Verify all API keys are correctly set in .env
3. Ensure GPU drivers are properly installed
4. Check available disk space and memory