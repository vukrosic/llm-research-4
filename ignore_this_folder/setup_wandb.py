#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration
Run this script to configure wandb for your project
"""

import os
import subprocess
import sys

def check_wandb_installed():
    """Check if wandb is installed"""
    try:
        import wandb
        print("✅ wandb is already installed")
        return True
    except ImportError:
        print("❌ wandb is not installed")
        return False

def install_wandb():
    """Install wandb package"""
    print("📦 Installing wandb...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ wandb installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install wandb")
        return False

def setup_wandb():
    """Setup wandb configuration"""
    print("\n�� Setting up Weights & Biases...")
    
    if not check_wandb_installed():
        if not install_wandb():
            return False
    
    print("\n�� To complete setup, you need to:")
    print("1. Create a free account at https://wandb.ai")
    print("2. Get your API key from your profile settings")
    print("3. Run: wandb login")
    print("4. Enter your API key when prompted")
    
    # Try to run wandb login
    try:
        print("\n🔑 Attempting to run wandb login...")
        subprocess.run([sys.executable, "-m", "wandb", "login"], check=True)
        print("✅ wandb login completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  wandb login failed. Please run 'wandb login' manually.")
        return False

def main():
    print("🚀 Weights & Biases Setup for LLM-Med Project")
    print("=" * 50)
    
    if setup_wandb():
        print("\n🎉 Setup completed successfully!")
        print("\n📊 You can now run your training script and it will automatically log to wandb!")
        print("   Run: python llm.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        print("\n💡 Manual setup:")
        print("   1. pip install wandb")
        print("   2. wandb login")
        print("   3. python llm.py")

if __name__ == "__main__":
    main()
