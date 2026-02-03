# Autonomous RL Agent with PPO

*An end-to-end reinforcement learning project: from algorithm implementation to production deployment.*

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)

[**🚀 Live Demo**](https://your-streamlit-url.streamlit.app) | [**📄 Research Paper**](docs/RESEARCH_PAPER.md) | [**📊 Results**](#results)

---

## 🎯 Project Overview

This project implements **Proximal Policy Optimization (PPO)** from scratch in PyTorch and trains autonomous agents on classic reinforcement learning environments. The trained models are deployed as an interactive web application using Streamlit.

**Key Features:**
- ✅ PPO algorithm implementation (~400 lines)
- ✅ Training on CartPole-v1 and LunarLander-v3
- ✅ Interactive Streamlit demo with live rendering
- ✅ Complete research documentation
- ✅ Production-ready deployment

![Demo GIF](docs/images/demo.gif)
*Watch the agent learn to balance and land in real-time*

---

## 📊 Results

### CartPole-v1

| Metric | Value | Target |
|--------|-------|--------|
| **Best Reward** | 500.0 ✅ | 475+ |
| **Success Rate** | 95% | 90% |
| **Training Time** | 7 minutes | - |
| **Episodes** | 500 | - |

![CartPole Training Curve](docs/images/cartpole_training_curve.png)

### LunarLander-v3

| Metric | Value | Target |
|--------|-------|--------|
| **Best Reward** | 94.9 | 200+ |
| **Success Rate** | 65% | 90% |
| **Training Time** | 20 minutes | - |
| **Episodes** | 1000 | - |

![LunarLander Training Curve](docs/images/lunarlander_training_curve.png)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
pip
```

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/autonomous-rl-agent.git
cd autonomous-rl-agent

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo Locally
```bash
cd streamlit_app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Train Your Own Agent
```bash
# Train CartPole
python src/train.py --env CartPole-v1 --episodes 500

# Train LunarLander
python src/train.py --env LunarLander-v3 --episodes 1000
```

---

## 🏗️ Architecture

### PPO Network
```
Input (State)
     ↓
Shared FC Layer (128/256 units)
     ↓
Shared FC Layer (128/256 units)
     ↓
     ├─→ Actor Head → Action Probabilities
     └─→ Critic Head → State Value
```

**Key Components:**
- **Actor-Critic Design:** Shared feature extraction with separate policy and value heads
- **Clipped Objective:** Prevents destructive policy updates
- **GAE:** Generalized Advantage Estimation for variance reduction

### Algorithm Overview
```python
# Simplified PPO update
for episode in episodes:
    # Collect trajectory
    states, actions, rewards = collect_episode()

    # Compute advantages
    advantages = compute_gae(rewards, values)

    # PPO update
    ratio = exp(new_log_prob - old_log_prob)
    loss = -min(ratio * advantages, 
                clip(ratio, 1-ε, 1+ε) * advantages)

    # Optimize
    loss.backward()
    optimizer.step()
```

---

## 📁 Project Structure
```
autonomous-rl-agent/
├── src/                    # Source code
│   ├── ppo_agent.py       # PPO implementation
│   ├── network.py         # Neural network
│   └── train.py           # Training script
├── models/                 # Trained models
│   ├── cartpole_best.pt
│   └── lunarlander_best.pt
├── streamlit_app/          # Web demo
│   └── app.py
├── docs/                   # Documentation
│   ├── RESEARCH_PAPER.md
│   └── images/
└── README.md
```

---

## 🎮 Usage Examples

### Load and Test a Model
```python
from src.network import PPONetwork
import torch
import gymnasium as gym

# Load model
model = PPONetwork(state_dim=4, action_dim=2, hidden_dim=128)
model.load_state_dict(torch.load('models/cartpole_best.pt'))
model.eval()

# Test in environment
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()

for _ in range(500):
    action, _, _ = model.get_action(state, deterministic=True)
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Train with Custom Hyperparameters
```python
from src.train import train_ppo

train_ppo(
    env_name='CartPole-v1',
    episodes=1000,
    learning_rate=3e-4,
    clip_epsilon=0.2,
    gamma=0.99
)
```

---

## 🔬 Technical Details

### Hyperparameters

| Parameter | CartPole | LunarLander |
|-----------|----------|-------------|
| Learning Rate | 3e-4 | 3e-4 |
| Clip Epsilon | 0.2 | 0.2 |
| Gamma | 0.99 | 0.99 |
| Hidden Units | 128 | 256 |
| Batch Size | Episode | Episode |

### Dependencies

**Core:**
- PyTorch 2.0+
- Gymnasium 0.29+
- NumPy 1.24+
- Matplotlib 3.7+

**Demo:**
- Streamlit 1.28+
- Pillow 10.0+

See `requirements.txt` for complete list.

---

## 📖 Documentation

- **[Research Paper](docs/RESEARCH_PAPER.md)** - Complete analysis and findings
- **[Deployment Guide](docs/DEPLOYMENT.md)** - How to deploy the Streamlit app
- **[API Documentation](docs/API.md)** - Code reference

---

## 🎯 Key Learnings

### What Worked Well
✅ **Simple architecture** - 2 hidden layers was sufficient  
✅ **Monte Carlo returns** - Stable learning signal  
✅ **Regular checkpointing** - Saved best performing models  
✅ **Progressive difficulty** - CartPole first, then LunarLander

### Challenges & Solutions
⚠️ **Sample inefficiency** → Advantage normalization helped  
⚠️ **Sparse rewards (LunarLander)** → Longer training needed  
⚠️ **Policy collapse** → Clipped objective prevented this

### Future Improvements
- [ ] Train LunarLander longer (reach 200+ target)
- [ ] Implement continuous action spaces
- [ ] Add hyperparameter tuning (Optuna)
- [ ] Compare with DQN and A2C
- [ ] Add more environments

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI Spinning Up** - RL fundamentals and best practices
- **Sutton & Barto** - Reinforcement Learning: An Introduction
- **PyTorch Team** - Excellent framework and documentation
- **Gymnasium** - Standardized RL environments

---

## 📬 Contact

**Your Name** - [LinkedIn](https://linkedin.com/in/yourprofile) - your.email@example.com

**Project Link:** [https://github.com/YOUR_USERNAME/autonomous-rl-agent](https://github.com/YOUR_USERNAME/autonomous-rl-agent)

**Live Demo:** [https://your-app.streamlit.app](https://your-app.streamlit.app)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/autonomous-rl-agent?style=social)
![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/autonomous-rl-agent?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/YOUR_USERNAME/autonomous-rl-agent?style=social)

**Made with ❤️ as part of my 168-day ML learning journey**

---

*This project demonstrates end-to-end ML skills: algorithm implementation, model training, deployment, and documentation. Perfect for showcasing in portfolios and job applications.*
