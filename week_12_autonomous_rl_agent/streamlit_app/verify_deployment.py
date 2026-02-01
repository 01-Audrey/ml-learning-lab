#!/usr/bin/env python3
"""
Deployment Helper Script
Verifies everything is ready for Streamlit Cloud deployment
"""

import os
import sys
import json
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath:50s} - {description}")
    return exists

def check_file_size(filepath):
    """Check file size (warn if > 100MB for GitHub)"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        if size > 100:
            print(f"   ⚠️  WARNING: File is {size:.2f}MB (GitHub limit is 100MB)")
            return False
    return True

def verify_requirements():
    """Verify requirements.txt has all dependencies"""
    required_packages = [
        'streamlit',
        'gymnasium',
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pillow'
    ]

    if not os.path.exists('streamlit_app/requirements.txt'):
        print("❌ requirements.txt not found!")
        return False

    with open('streamlit_app/requirements.txt', 'r') as f:
        content = f.read().lower()

    missing = []
    for package in required_packages:
        if package not in content:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages in requirements.txt: {', '.join(missing)}")
        return False

    print("✅ All required packages in requirements.txt")
    return True

def main():
    print("=" * 80)
    print("🚀 STREAMLIT CLOUD DEPLOYMENT VERIFICATION")
    print("=" * 80)

    print("\n📋 Checking required files...\n")

    # Check files
    files = {
        'streamlit_app/app.py': 'Main application',
        'streamlit_app/ppo_network.py': 'Model architecture',
        'streamlit_app/requirements.txt': 'Dependencies',
        'streamlit_app/README.md': 'Documentation',
        'streamlit_app/.streamlit/config.toml': 'Configuration',
        'results/day79/cartpole_best_model.pt': 'Trained model'
    }

    all_good = True
    for filepath, description in files.items():
        if not check_file(filepath, description):
            all_good = False
        else:
            if not check_file_size(filepath):
                all_good = False

    print("\n📦 Checking dependencies...\n")
    if not verify_requirements():
        all_good = False

    print("\n" + "=" * 80)

    if all_good:
        print("\n✅ ALL CHECKS PASSED! Ready for deployment! 🎉")
        print("\n📝 Next steps:")
        print("   1. Commit and push to GitHub")
        print("   2. Go to https://share.streamlit.io")
        print("   3. Click 'New app'")
        print("   4. Select your repository")
        print("   5. Set path: week_12_autonomous_rl_agent/streamlit_app/app.py")
        print("   6. Click Deploy!")
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("\n💡 Fix the issues above before deploying")

    print("\n" + "=" * 80)

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
