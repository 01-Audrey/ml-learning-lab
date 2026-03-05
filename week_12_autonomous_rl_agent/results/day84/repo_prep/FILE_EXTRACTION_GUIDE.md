# File Extraction Guide: ml-learning-lab to autonomous-rl-agent

## 🎯 Goal

Extract Week 12 files from `ml-learning-lab` and organize them into a clean, professional `autonomous-rl-agent` repository.

---

## 📋 Pre-Extraction Checklist

Before starting:
- [ ] Commit all changes in ml-learning-lab
- [ ] Verify all Week 12 files are present
- [ ] Ensure models are downloaded from Colab/Drive
- [ ] Have both repos ready (source and destination)

---

## 🗂️ File Mapping

### From ml-learning-lab to autonomous-rl-agent

#### Source Code Files

**Training Code:**
```
FROM: ml-learning-lab/week_12_autonomous_rl_agent/day_78-79_*.ipynb
TO:   autonomous-rl-agent/src/

Extract:
- PPO training function -> ppo_agent.py
- Network class -> network.py
- Training script -> train.py
- Helper functions -> utils.py
```

**Streamlit App:**
```
FROM: ml-learning-lab/week_12_autonomous_rl_agent/streamlit_app/
TO:   autonomous-rl-agent/streamlit_app/

Copy entire folder:
- app.py
- ppo_network.py
- requirements.txt
- .streamlit/config.toml
- README.md (update paths)
```

**Trained Models:**
```
FROM: ml-learning-lab/week_12_autonomous_rl_agent/results/day78/
      ml-learning-lab/week_12_autonomous_rl_agent/results/day79/
TO:   autonomous-rl-agent/models/

Copy:
- results/day79/cartpole_best_model.pt -> models/cartpole_best.pt
- results/day78/lunarlander_best_model.pt -> models/lunarlander_best.pt
```

**Documentation:**
```
FROM: ml-learning-lab/week_12_autonomous_rl_agent/results/day81/
TO:   autonomous-rl-agent/docs/

Copy and rename:
- COMPLETE_RESEARCH_PAPER.txt -> docs/RESEARCH_PAPER.md
```

**Visualizations:**
```
FROM: ml-learning-lab/week_12_autonomous_rl_agent/results/day83/
TO:   autonomous-rl-agent/docs/images/

Copy:
- cartpole_training_curve.png -> docs/images/cartpole_training_curve.png
- lunarlander_training_curve.png -> docs/images/lunarlander_training_curve.png
```

---

## 🔧 Step-by-Step Extraction Process

### Step 1: Create New Repository

**On GitHub:**
1. Go to https://github.com/new
2. Repository name: `autonomous-rl-agent`
3. Description: "Autonomous reinforcement learning agent using PPO"
4. Public repository
5. Add README: No (we'll create custom)
6. Add .gitignore: Yes (Python)
7. Add license: Yes (MIT License)
8. Click "Create repository"

**On Local Machine:**
```powershell
# Clone the new repo
cd C:\Users\audrey\Documents\
git clone https://github.com/YOUR_USERNAME/autonomous-rl-agent.git
cd autonomous-rl-agent
```

### Step 2: Create Folder Structure
```powershell
# Create all directories
mkdir src
mkdir models
mkdir streamlit_app
mkdir streamlit_app\.streamlit
mkdir docs
mkdir docs\images
mkdir notebooks
mkdir scripts

# Verify structure
tree /F
```

Expected output:
```
autonomous-rl-agent/
├── docs/
│   └── images/
├── models/
├── notebooks/
├── scripts/
├── src/
└── streamlit_app/
    └── .streamlit/
```

### Step 3: Extract Source Code

You'll need to manually extract and clean the code from notebooks into Python modules.

**Key files to create in src/:**
- network.py (PPONetwork class)
- ppo_agent.py (PPOAgent training class)
- train.py (CLI training script)
- utils.py (helper functions)
- __init__.py (package initialization)

This involves copying code from your Jupyter notebooks and organizing it into clean modules.

### Step 4: Copy Streamlit App
```powershell
# Copy entire streamlit_app folder
xcopy /E /I C:\Users\audrey\Documents\ml-learning-lab\week_12_autonomous_rl_agent\streamlit_app C:\Users\audrey\Documents\autonomous-rl-agent\streamlit_app

# Verify
ls streamlit_app
```

**Update streamlit_app/app.py paths:**

Find and replace in `streamlit_app/app.py`:
```python
# OLD:
'../results/day79/cartpole_best_model.pt'
'../results/day78/lunarlander_best_model.pt'

# NEW:
'../models/cartpole_best.pt'
'../models/lunarlander_best.pt'
```

### Step 5: Copy Models
```powershell
# Copy CartPole model
copy C:\Users\audrey\Documents\ml-learning-lab\week_12_autonomous_rl_agent\results\day79\cartpole_best_model.pt C:\Users\audrey\Documents\autonomous-rl-agent\models\cartpole_best.pt

# Copy LunarLander model
copy C:\Users\audrey\Documents\ml-learning-lab\week_12_autonomous_rl_agent\results\day78\lunarlander_best_model.pt C:\Users\audrey\Documents\autonomous-rl-agent\models\lunarlander_best.pt

# Verify
ls models
```

### Step 6: Copy Documentation
```powershell
# Copy research paper
copy C:\Users\audrey\Documents\ml-learning-lab\week_12_autonomous_rl_agent\results\day81\COMPLETE_RESEARCH_PAPER.txt C:\Users\audrey\Documents\autonomous-rl-agent\docs\RESEARCH_PAPER.md

# Copy training curves
copy C:\Users\audrey\Documents\ml-learning-lab\week_12_autonomous_rl_agent\results\day83\cartpole_training_curve.png C:\Users\audrey\Documents\autonomous-rl-agent\docs\images\

copy C:\Users\audrey\Documents\ml-learning-lab\week_12_autonomous_rl_agent\results\day83\lunarlander_training_curve.png C:\Users\audrey\Documents\autonomous-rl-agent\docs\images\

# Verify
ls docs
ls docs\images
```

### Step 7: Create Additional Files

**Create models/README.md**

**Create requirements.txt**

**Update main README.md**

### Step 8: Test Everything
```powershell
# Test imports (after creating src files)
python -c "from src import PPONetwork, PPOAgent"

# Test model loading
python -c "from src.network import PPONetwork; import torch; m = PPONetwork(4,2,128); m.load_state_dict(torch.load('models/cartpole_best.pt')); print('Model loads OK')"
```

---

## ✅ Verification Checklist

After extraction, verify:
```powershell
cd C:\Users\audrey\Documents\autonomous-rl-agent

# Check structure
tree /F

# Check file counts
ls src         # Should have 4-5 files
ls models      # Should have 2-3 files
ls streamlit_app  # Should have 4+ files
ls docs        # Should have 2+ files
ls docs\images # Should have 2+ files
```

---

## 📤 Commit to GitHub
```powershell
cd C:\Users\audrey\Documents\autonomous-rl-agent

# Check status
git status

# Add all files
git add .

# Commit
git commit -m "Initial commit: Complete PPO RL agent implementation"

# Push
git push origin main
```

---

## 🎯 Post-Extraction Tasks

After pushing:
- [ ] Verify repository looks good on GitHub
- [ ] Test cloning fresh copy
- [ ] Deploy Streamlit app from new repo
- [ ] Update Streamlit Cloud deployment path
- [ ] Add repository URL to portfolio
- [ ] Update LinkedIn with project link
- [ ] Share on social media

---

**Extraction complete! Your portfolio repo is ready!** 🎉
