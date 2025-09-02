#!/usr/bin/env python3
"""
RunPod Startup Script for SuperNova
Simple script to start both services with proper error handling
"""

import subprocess
import time
import os
import sys
import requests

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("[WARN] No GPU detected, will use CPU")
            return False
    except ImportError:
        print("[WARN] PyTorch not installed, skipping GPU check")
        return False

def start_fingpt():
    """Start FinGPT server."""
    print("\n[INFO] Starting FinGPT server...")
    os.chdir("/workspace/nova_signal/supernova/FinGPT")
    
    # Start the server
    process = subprocess.Popen(
        ["python3", "supernova_fingpt_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for startup
    time.sleep(10)
    
    # Check if it's running
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code == 200:
            print("[OK] FinGPT server is running on port 8080")
            return process
        else:
            print(f"[ERROR] FinGPT server returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] FinGPT server failed to start: {e}")
        return None

def start_supernova():
    """Start SuperNova backend."""
    print("\n[INFO] Starting SuperNova backend...")
    os.chdir("/workspace/nova_signal/supernova/backend")
    
    # Start the server
    process = subprocess.Popen(
        ["python3", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for startup
    time.sleep(5)
    
    # Check if it's running
    try:
        response = requests.get("http://localhost:8081/health")
        if response.status_code == 200:
            print("[OK] SuperNova backend is running on port 8081")
            return process
        else:
            print(f"[ERROR] SuperNova backend returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] SuperNova backend failed to start: {e}")
        return None

def main():
    """Main startup routine."""
    print("=" * 60)
    print("SuperNova RunPod Startup")
    print("=" * 60)
    
    # Check GPU
    check_gpu()
    
    # Start services
    fingpt_process = start_fingpt()
    supernova_process = start_supernova()
    
    if fingpt_process and supernova_process:
        print("\n" + "=" * 60)
        print("All services started successfully!")
        print("=" * 60)
        print("FinGPT API: http://localhost:8080")
        print("SuperNova API: http://localhost:8081")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 60)
        
        # Keep running
        try:
            while True:
                time.sleep(60)
                # Periodic health check
                try:
                    requests.get("http://localhost:8080/health", timeout=5)
                    requests.get("http://localhost:8081/health", timeout=5)
                except:
                    print("[WARN] Health check failed")
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down services...")
            fingpt_process.terminate()
            supernova_process.terminate()
            print("[OK] Services stopped")
    else:
        print("\n[ERROR] Failed to start all services")
        sys.exit(1)

if __name__ == "__main__":
    main()