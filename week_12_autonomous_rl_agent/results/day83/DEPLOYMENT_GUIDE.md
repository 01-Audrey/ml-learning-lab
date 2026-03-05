# Step-by-Step Streamlit Cloud Deployment Guide

## 🎯 Goal

Deploy your PPO Agent Demo to Streamlit Cloud and get a public URL you can share!

**Time Required:** ~15 minutes

---

## 📋 Prerequisites

Before starting, ensure:
- ✅ All code is working locally
- ✅ All files committed to GitHub
- ✅ Model file is in the repository
- ✅ GitHub repository is public (or Streamlit has access)

---

## 🚀 Step 1: Verify GitHub Repository

### 1.1 Go to Your GitHub Repository

Navigate to: `https://github.com/YOUR_USERNAME/ml-learning-lab`

### 1.2 Verify Files Are Present

Check that these files exist:
```
ml-learning-lab/
└── week_12_autonomous_rl_agent/
    ├── streamlit_app/
    │   ├── app.py                    ✅
    │   ├── ppo_network.py            ✅
    │   ├── requirements.txt          ✅
    │   └── .streamlit/config.toml    ✅
    └── results/
        └── day79/
            └── cartpole_best_model.pt ✅
```

### 1.3 Check Latest Commit

- Click on "Commits"
- Verify your latest Day 82/83 commit is visible
- Make sure all files show up in the commit

**✅ Checkpoint:** All files visible on GitHub

---

## 🌐 Step 2: Sign Up for Streamlit Cloud

### 2.1 Go to Streamlit Cloud

Navigate to: https://share.streamlit.io

### 2.2 Sign In with GitHub

- Click "Sign in"
- Select "Continue with GitHub"
- Authorize Streamlit to access your repositories

### 2.3 Complete Profile (if first time)

- Add your name
- Choose a username (will be in your app URL)
- Complete any other required fields

**✅ Checkpoint:** Logged into Streamlit Cloud

---

## 🎨 Step 3: Create New App

### 3.1 Click "New App" Button

Look for the big blue "New app" button in the top right

### 3.2 Fill in Deployment Settings

**Repository:**
- Select: `YOUR_USERNAME/ml-learning-lab`

**Branch:**
- Select: `main` (or your default branch)

**Main file path:**
```
week_12_autonomous_rl_agent/streamlit_app/app.py
```

⚠️ **IMPORTANT:** Type this path exactly! Include the folder structure!

### 3.3 Advanced Settings (Optional)

Click "Advanced settings" if you need to:

**Python version:**
- Recommended: `3.9` or `3.10`

**Secrets:**
- Not needed for this project (leave empty)

**Environment variables:**
- Not needed for this project (leave empty)

**✅ Checkpoint:** All deployment settings configured

---

## 🚀 Step 4: Deploy!

### 4.1 Click "Deploy!" Button

The big blue button at the bottom

### 4.2 Watch Build Logs

You'll see:
```
🔄 Installing dependencies...
📦 Installing streamlit
📦 Installing torch
📦 Installing gymnasium
... (more packages)

✅ Dependencies installed!

🚀 Starting app...
```

**This takes 2-5 minutes** ⏱️

### 4.3 Wait for Success

You'll see:
```
✅ App is live!
```

**✅ Checkpoint:** App deployed successfully

---

## 🎉 Step 5: Test Your App

### 5.1 Get Your Public URL

Your app will be at:
```
https://YOUR_USERNAME-ml-learning-lab-week12-app.streamlit.app
```

### 5.2 Test Basic Functionality

- [ ] App loads without errors
- [ ] All 5 tabs are accessible
- [ ] Sidebar shows correctly
- [ ] Model status shows "CartPole Model Ready"

### 5.3 Run an Episode

- [ ] Click "Run Episode(s)"
- [ ] Episode runs successfully
- [ ] Environment renders
- [ ] Stats update correctly
- [ ] No errors in console

### 5.4 Check Other Tabs

- [ ] Performance tab shows metrics
- [ ] Analysis tab displays info
- [ ] About tab loads correctly

**✅ Checkpoint:** App fully functional

---

## 📝 Step 6: Update Documentation

### 6.1 Copy Your Public URL

Example: `https://audrey-ml-learning-lab-week12.streamlit.app`

### 6.2 Update Main README

Add to your `ml-learning-lab/README.md`:
```markdown
## 🌟 Featured Project: Week 12 - Autonomous RL Agent

### 🤖 [Live Demo →](https://YOUR_URL.streamlit.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_URL.streamlit.app)
```

### 6.3 Update Streamlit App README

Update `streamlit_app/README.md`:
```markdown
# 🤖 Autonomous RL Agent - Interactive Demo

## 🌐 Live Demo

[Try it now!](https://YOUR_URL.streamlit.app)
```

### 6.4 Commit & Push
```bash
git add README.md streamlit_app/README.md
git commit -m "docs: add live Streamlit app URL"
git push origin main
```

**✅ Checkpoint:** Documentation updated

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"

**Cause:** Missing package in requirements.txt

**Solution:**
1. Check the error log for missing package
2. Add to `requirements.txt`
3. Commit and push
4. Streamlit will auto-redeploy

### Issue: "File not found: cartpole_best_model.pt"

**Cause:** Model file not committed to GitHub

**Solution:**
```bash
git add results/day79/cartpole_best_model.pt
git commit -m "add trained model"
git push origin main
```

### Issue: App Crashes on Startup

**Cause:** Error in code

**Solution:**
1. Check Streamlit Cloud logs
2. Look for Python errors
3. Fix the error locally
4. Test locally first
5. Push fix to GitHub

### Issue: "App is taking too long"

**Cause:** Large dependencies or model file

**Solution:**
- Normal for first deployment (3-5 minutes)
- Subsequent deployments are faster
- If >10 minutes, check logs for errors

### Issue: Changes Not Showing Up

**Cause:** Not pushed to GitHub

**Solution:**
```bash
git status  # Check what's uncommitted
git add .
git commit -m "update: fix bug"
git push origin main
```

Streamlit auto-redeploys within 2 minutes

---

## 🔄 Updating Your App

Any time you push to GitHub:
1. Make changes locally
2. Test locally
3. Commit and push
4. Streamlit auto-redeploys
```bash
# Example update workflow
git add streamlit_app/app.py
git commit -m "update: improve UI"
git push origin main

# Wait ~2 minutes for auto-redeploy
```

---

## 📊 Monitoring

### View Logs

In Streamlit Cloud dashboard:
- Click on your app
- View "Logs" tab
- See all console output

### Usage Analytics

Streamlit Cloud shows:
- Number of visitors
- App uptime
- Resource usage

---

## 💰 Costs

**Streamlit Cloud Free Tier:**
- ✅ Unlimited public apps
- ✅ 1GB RAM per app
- ✅ Automatic SSL/HTTPS
- ✅ Auto-deploy from GitHub
- ❌ Limited to 3 apps max

**Perfect for portfolio projects!** 🎉

---

## ✅ Success Criteria

Your deployment is successful when:
- ✅ App loads at public URL
- ✅ No errors on startup
- ✅ All features work
- ✅ URL is shareable
- ✅ Documentation updated

---

## 🎊 Congratulations!

Your PPO Agent Demo is now live! 🚀

**Next Steps:**
1. Share URL on LinkedIn
2. Add to resume/portfolio
3. Include in internship applications
4. Create demo video
5. Celebrate! 🎉

---

**Deployment Date:** _____________

**Public URL:** _____________

**Status:** ✅ Live
