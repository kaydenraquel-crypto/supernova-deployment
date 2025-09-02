# Push to GitHub Instructions

## Step 1: Create New Repository on GitHub
1. Go to https://github.com/new
2. Name it: `supernova-deployment` (or your preferred name)
3. Make it private (recommended for API keys safety)
4. Don't initialize with README (we have our own)

## Step 2: Initialize Git and Push

Run these commands in the `supernova` directory:

```bash
# Navigate to supernova directory
cd C:\Users\iseel\project01\nova_signal_v.0.1\supernova

# Initialize git repository
git init

# Add all necessary files
git add backend/
git add FinGPT/
git add scripts/
git add .env.example
git add .gitignore
git add requirements.txt
git add runpod_start.py
git add deploy_to_runpod.sh
git add RUNPOD_DEPLOYMENT_GUIDE.md
git add README_DEPLOYMENT.md
git add README.md

# Commit the files
git commit -m "Initial commit: SuperNova + FinGPT deployment for RunPod"

# Add your GitHub repository as origin
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/supernova-deployment.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify Files to Push

Before pushing, verify these essential directories and files are included:

### ✅ Required Directories:
- `backend/` - Complete SuperNova backend
  - `app.py`
  - `core/`
  - `llm/`
  - `routes/`
  - `services/`
  - `tools/`
- `FinGPT/` - FinGPT server
  - `supernova_fingpt_server.py`
- `scripts/` - Utility scripts

### ✅ Required Files:
- `.env.example` - Environment template
- `.gitignore` - Git ignore rules
- `requirements.txt` - Python dependencies
- `runpod_start.py` - Startup script
- `RUNPOD_DEPLOYMENT_GUIDE.md` - Deployment guide

### ❌ DO NOT Push:
- `.env` - Contains API keys (in .gitignore)
- `*.exe` - Windows executables
- `__pycache__/` - Python cache
- `logs/` - Log files
- `training/` - Training files (not needed)
- `config-gui/` - GUI tools (not needed)

## Step 4: Alternative - Push Specific Files Only

If you want to be more selective:

```bash
# Initialize repository
git init

# Add only essential files
git add backend/*.py
git add backend/core/*.py
git add backend/llm/*.py
git add backend/routes/*.py
git add backend/services/*.py
git add backend/tools/*.py
git add FinGPT/supernova_fingpt_server.py
git add runpod_start.py
git add requirements.txt
git add .env.example
git add .gitignore
git add RUNPOD_DEPLOYMENT_GUIDE.md

# Commit and push
git commit -m "SuperNova + FinGPT for RunPod deployment"
git remote add origin https://github.com/YOUR_USERNAME/supernova-deployment.git
git push -u origin main
```

## Step 5: On RunPod

Once pushed, on your RunPod instance:

```bash
# Clone the repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/supernova-deployment.git
cd supernova-deployment

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Start services
python runpod_start.py
```

## Important Notes

⚠️ **Security**: Never commit `.env` files with real API keys
⚠️ **Size**: If repo is too large, consider using Git LFS for model files
✅ **Testing**: Test the deployment on RunPod after pushing