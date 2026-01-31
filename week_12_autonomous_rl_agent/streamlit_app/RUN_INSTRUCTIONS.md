# How to Run the Streamlit App

## Option 1: Run Locally

### Step 1: Navigate to the app directory
```bash
cd streamlit_app
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
streamlit run app.py
```

The app will open in your browser at: http://localhost:8501

## Option 2: Run from Colab (Development)

Since Streamlit requires a persistent server, running from Colab is limited.
Instead, use localtunnel or ngrok for testing:
```bash
!pip install streamlit pyngrok

# In a separate cell
!streamlit run streamlit_app/app.py &

# Then expose with ngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"Streamlit app available at: {public_url}")
```

## Testing Checklist

- [ ] App loads without errors
- [ ] All 5 tabs are accessible
- [ ] "Run Episode" button works
- [ ] Environment renders correctly
- [ ] Episode stats update in real-time
- [ ] Action probabilities display
- [ ] Performance charts render
- [ ] History tab tracks episodes
- [ ] CSV download works
- [ ] All links and buttons functional

## Troubleshooting

**Model not found error:**
- Ensure `../results/day79/cartpole_best_model.pt` exists
- App will fall back to random model for demo

**Import errors:**
- Check all dependencies are installed
- Verify Python version (3.8+)
- Run: `pip install -r requirements.txt`

**Port already in use:**
- Stop other Streamlit instances
- Or specify different port: `streamlit run app.py --server.port 8502`

## Files Structure
```
streamlit_app/
├── app.py                    # Main application
├── ppo_network.py           # Model architecture
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── .streamlit/             # Configuration
│   └── config.toml
└── RUN_INSTRUCTIONS.md     # This file
```

## Next Steps

1. Test locally first
2. Fix any bugs
3. Deploy to Streamlit Cloud
4. Share the link!
