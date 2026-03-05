# Deployment Guide - Streamlit Cloud

## 🚀 Deploy Your PPO Agent Demo to Streamlit Cloud

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at https://share.streamlit.io)
- Your code pushed to GitHub

---

## Step 1: Prepare Your Repository

### 1.1 Ensure All Files Are Committed
```bash
# Check status
git status

# Add all files
git add .

# Commit
git commit -m "feat: add Day 82 - Interactive Streamlit Demo"

# Push to GitHub
git push origin main
```

### 1.2 Verify File Structure

Your repository should have:
```
ml-learning-lab/
└── week_12_autonomous_rl_agent/
    ├── streamlit_app/
    │   ├── app.py
    │   ├── ppo_network.py
    │   ├── requirements.txt
    │   ├── README.md
    │   └── .streamlit/
    │       └── config.toml
    └── results/
        └── day79/
            └── cartpole_best_model.pt  (Important!)
```

---

## Step 2: Sign Up for Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "Sign up"
3. Sign in with your GitHub account
4. Authorize Streamlit to access your repositories

---

## Step 3: Deploy Your App

### 3.1 Create New App

1. Click "New app" button
2. Select your repository: `ml-learning-lab`
3. Select branch: `main`
4. Set main file path: `week_12_autonomous_rl_agent/streamlit_app/app.py`

### 3.2 Advanced Settings (Optional)

Click "Advanced settings" to:
- Set Python version (3.9 or 3.10 recommended)
- Add secrets (if needed)
- Configure resources

### 3.3 Deploy!

Click "Deploy!" and wait 2-5 minutes for:
- Dependencies to install
- App to build
- App to launch

---

## Step 4: Test Your Deployed App

Once deployed, you'll get a URL like:
```
https://your-username-ml-learning-lab-week12-app.streamlit.app
```

Test:
- [ ] App loads without errors
- [ ] All tabs are accessible
- [ ] Episode runs successfully
- [ ] Charts render correctly
- [ ] Model loads properly

---

## Step 5: Share Your App

### Add to README
```markdown
## 🌐 Live Demo

[Try the Interactive PPO Agent Demo](https://your-app-url.streamlit.app)
```

### Share on LinkedIn
```
🚀 Just deployed an interactive RL demo!

Watch my trained PPO agent solve CartPole in real-time:
[Your Streamlit URL]

Features:
✅ Live episode playback
✅ Performance metrics
✅ Algorithm comparison
✅ Statistical analysis

Part of Week 12 of my 168-day ML journey!

#MachineLearning #ReinforcementLearning #AI #Portfolio
```

---

## Troubleshooting

### Issue: Module Not Found

**Solution:** Check `requirements.txt` includes all dependencies
```txt
streamlit==1.29.0
gymnasium==0.29.1
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
pillow==10.0.0
```

### Issue: Model File Not Found

**Solution:** Ensure model file is committed to GitHub
```bash
# Check if file exists
git ls-files results/day79/cartpole_best_model.pt

# If not, add it
git add results/day79/cartpole_best_model.pt
git commit -m "add trained model"
git push
```

### Issue: App Crashes on Startup

**Solution:** Check logs in Streamlit Cloud dashboard
- Look for Python errors
- Verify all imports work
- Check file paths are relative

### Issue: Slow Performance

**Solution:** Optimize settings in `.streamlit/config.toml`
```toml
[server]
enableXsrfProtection = true
maxUploadSize = 200

[runner]
magicEnabled = true
fastReruns = true
```

---

## Updating Your App

Any push to GitHub automatically redeploys:
```bash
# Make changes to app.py
git add streamlit_app/app.py
git commit -m "update: improve UI"
git push origin main

# Streamlit Cloud auto-redeploys in ~2 minutes
```

---

## Cost

**Streamlit Cloud Free Tier:**
- ✅ Unlimited public apps
- ✅ 1 GB RAM per app
- ✅ Automatic SSL
- ✅ GitHub integration
- ❌ Limited to 3 apps

Perfect for portfolio projects! 🎉

---

## Next Steps

1. Deploy your app
2. Test thoroughly
3. Add URL to your resume
4. Share on LinkedIn
5. Include in portfolio
6. Mention in internship applications

**Your deployed app is now a live portfolio piece!** 🏆
