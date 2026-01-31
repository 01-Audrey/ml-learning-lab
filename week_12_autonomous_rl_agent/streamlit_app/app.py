import streamlit as st
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ppo_network import PPONetwork
import time
from PIL import Image
import io
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PPO Agent Demo - Week 12",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Autonomous RL Agent Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Week 12: PPO Algorithm Showcase</div>', unsafe_allow_html=True)

# Initialize session state
if 'episode_history' not in st.session_state:
    st.session_state.episode_history = []
if 'current_episode' not in st.session_state:
    st.session_state.current_episode = 0

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Environment selection
    env_name = st.selectbox(
        "Select Environment",
        ["CartPole-v1", "LunarLander-v3"],
        help="Choose which environment to run"
    )

    st.markdown("---")

    # Episode controls
    st.subheader("Episode Controls")
    num_episodes = st.slider("Number of Episodes", 1, 10, 1)
    deterministic = st.checkbox("Deterministic Policy", value=True)
    show_probs = st.checkbox("Show Action Probs", value=True)
    speed = st.slider("Animation Speed", 0.01, 0.2, 0.05, 0.01)

    st.markdown("---")

    # Algorithm info
    st.subheader("Algorithm Info")
    st.info("""
    **Algorithm:** PPO (Proximal Policy Optimization)

    **Features:**
    - Multi-epoch updates (5x)
    - Clipped surrogate objective
    - Optimized hyperparameters
    - Expert-level performance

    **Network:**
    - Input: State dimension
    - Hidden: 128-128 neurons
    - Output: Actions + Value
    - Params: ~35K (CartPole)
    """)

    st.markdown("---")

    # Model status
    st.subheader("Model Status")

    if env_name == "CartPole-v1":
        st.success("CartPole Model Ready")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Performance", "475±25")
            st.metric("CV", "5.2%")
        with col2:
            st.metric("Success Rate", "95%")
            st.metric("Episodes Run", st.session_state.current_episode)
    else:
        st.warning("LunarLander Model")
        st.info("Model available but requires Day 78 checkpoint")

    # Clear history button
    if st.button("Clear History"):
        st.session_state.episode_history = []
        st.session_state.current_episode = 0
        st.rerun()

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Demo", 
    "Performance", 
    "Training History", 
    "Analysis", 
    "About"
])

with tab1:
    st.header("Live Agent Demonstration")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Environment Rendering")
        render_placeholder = st.empty()

        # Control buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_episode = st.button("Run Episode(s)", key="run_episode", use_container_width=True)
        with btn_col2:
            save_stats = st.checkbox("Save to History", value=True)

    with col2:
        st.subheader("Episode Stats")
        stats_placeholder = st.empty()

        st.subheader("Action Probabilities")
        prob_placeholder = st.empty()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run episode logic
    if run_episode and env_name == "CartPole-v1":
        try:
            # Load model
            device = torch.device("cpu")
            model = PPONetwork(state_dim=4, action_dim=2, hidden_dim=128).to(device)

            # Try to load trained model
            model_loaded = False
            try:
                model.load_state_dict(torch.load('../results/day79/cartpole_best_model.pt', map_location=device))
                st.success("Loaded trained model from Day 79!")
                model_loaded = True
            except:
                st.warning("Using random model (for demo purposes)")

            model.eval()

            # Run multiple episodes
            for ep in range(num_episodes):
                status_text.text(f"Running episode {ep+1}/{num_episodes}...")

                # Create environment
                env = gym.make(env_name, render_mode="rgb_array")
                state, _ = env.reset()

                episode_reward = 0
                episode_length = 0
                done = False
                actions_taken = []

                # Run episode
                while not done and episode_length < 500:
                    # Get action
                    action, probs, value = model.get_action(state, deterministic=deterministic)
                    actions_taken.append(action)

                    # Step environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    episode_length += 1
                    state = next_state

                    # Update display
                    if episode_length % 5 == 0 or done:
                        # Render
                        frame = env.render()
                        render_placeholder.image(frame, caption=f"Episode {ep+1}/{num_episodes} - Step {episode_length}", use_column_width=True)

                        # Update stats
                        status_class = "success-card" if episode_reward >= 475 else "warning-card" if episode_reward >= 200 else "metric-card"
                        stats_html = f"""
                        <div class="{status_class}">
                            <h3>Episode {ep+1} Progress</h3>
                            <p><strong>Steps:</strong> {episode_length}/500</p>
                            <p><strong>Reward:</strong> {episode_reward:.1f}</p>
                            <p><strong>Status:</strong> {'Completed' if done else 'Running'}</p>
                            <p><strong>Left/Right:</strong> {actions_taken.count(0)}/{actions_taken.count(1)}</p>
                        </div>
                        """
                        stats_placeholder.markdown(stats_html, unsafe_allow_html=True)

                        # Show action probabilities
                        if show_probs:
                            prob_fig, ax = plt.subplots(figsize=(5, 3))
                            actions = ['Left', 'Right']
                            colors = ['#FF6B6B', '#4ECDC4']
                            bars = ax.bar(actions, probs[0], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                            ax.set_ylabel('Probability', fontweight='bold')
                            ax.set_title('Action Distribution', fontweight='bold')
                            ax.set_ylim([0, 1])
                            ax.grid(True, alpha=0.3, axis='y')

                            # Add value labels
                            for bar, prob in zip(bars, probs[0]):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                       f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

                            prob_placeholder.pyplot(prob_fig)
                            plt.close()

                        # Update progress bar
                        progress = (episode_length / 500.0) * (1.0 / num_episodes) + (ep / num_episodes)
                        progress_bar.progress(min(progress, 1.0))

                        time.sleep(speed)

                env.close()

                # Save to history
                if save_stats:
                    st.session_state.episode_history.append({
                        'episode': st.session_state.current_episode + 1,
                        'reward': episode_reward,
                        'length': episode_length,
                        'left_actions': actions_taken.count(0),
                        'right_actions': actions_taken.count(1),
                        'solved': episode_reward >= 475
                    })
                    st.session_state.current_episode += 1

            # Final summary
            if num_episodes == 1:
                final_class = "success-card" if episode_reward >= 475 else "warning-card"
                final_html = f"""
                <div class="{final_class}">
                    <h2>Episode Complete!</h2>
                    <h3>Total Reward: {episode_reward:.1f}</h3>
                    <p><strong>Episode Length:</strong> {episode_length} steps</p>
                    <p><strong>Action Balance:</strong> Left: {actions_taken.count(0)}, Right: {actions_taken.count(1)}</p>
                    <p><strong>Status:</strong> {'SOLVED! (>=475)' if episode_reward >= 475 else 'Good! (>=200)' if episode_reward >= 200 else 'Try again'}</p>
                </div>
                """
            else:
                avg_reward = np.mean([h['reward'] for h in st.session_state.episode_history[-num_episodes:]])
                success_rate = sum([h['solved'] for h in st.session_state.episode_history[-num_episodes:]]) / num_episodes * 100
                final_class = "success-card" if success_rate >= 80 else "warning-card"
                final_html = f"""
                <div class="{final_class}">
                    <h2>{num_episodes} Episodes Complete!</h2>
                    <h3>Average Reward: {avg_reward:.1f}</h3>
                    <p><strong>Success Rate:</strong> {success_rate:.1f}% ({sum([h['solved'] for h in st.session_state.episode_history[-num_episodes:]])}/{num_episodes})</p>
                    <p><strong>Best Reward:</strong> {max([h['reward'] for h in st.session_state.episode_history[-num_episodes:]]):.1f}</p>
                    <p><strong>Worst Reward:</strong> {min([h['reward'] for h in st.session_state.episode_history[-num_episodes:]]):.1f}</p>
                </div>
                """

            stats_placeholder.markdown(final_html, unsafe_allow_html=True)
            progress_bar.progress(1.0)
            status_text.text("All episodes completed!")

        except Exception as e:
            st.error(f"Error running episode: {str(e)}")
            st.info("Make sure the trained model exists at: `../results/day79/cartpole_best_model.pt`")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

with tab2:
    st.header("Performance Metrics")

    st.subheader("Training Results Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Reward", "475±25", delta="Solved")

    with col2:
        st.metric("Success Rate", "95%", delta="+53% vs baseline")

    with col3:
        st.metric("Training Episodes", "500", delta="Day 79")

    with col4:
        st.metric("CV", "5.2%", delta="Excellent")

    st.markdown("---")

    # Performance comparison chart
    st.subheader("Algorithm Comparison")

    comparison_data = {
        'Algorithm': ['REINFORCE', 'A2C', 'PPO (Base)', 'PPO (Opt)'],
        'Mean Reward': [200, 200, 200, 475],
        'Std Dev': [50, 50, 40, 25],
        'Success Rate (%)': [42, 42, 42, 95]
    }

    df = pd.DataFrame(comparison_data)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77']
        bars = ax.bar(df['Algorithm'], df['Mean Reward'], 
                     yerr=df['Std Dev'], capsize=5, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=2)
        ax.axhline(475, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Solved Threshold')
        ax.set_ylabel('Average Reward', fontweight='bold')
        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_title('Performance Comparison', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        st.pyplot(fig)
        plt.close()

with tab3:
    st.header("Your Testing History")

    if len(st.session_state.episode_history) > 0:
        # Convert to dataframe
        history_df = pd.DataFrame(st.session_state.episode_history)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Plot reward over episodes
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history_df['episode'], history_df['reward'], 
                   marker='o', linewidth=2, markersize=8, 
                   color='#1f77b4', label='Episode Reward')
            ax.axhline(475, color='green', linestyle='--', linewidth=2, 
                      alpha=0.7, label='Solved Threshold')
            ax.fill_between(history_df['episode'], 0, history_df['reward'], 
                           where=(history_df['reward'] >= 475), 
                           color='green', alpha=0.2)
            ax.set_xlabel('Episode Number', fontweight='bold')
            ax.set_ylabel('Total Reward', fontweight='bold')
            ax.set_title('Your Episode Rewards Over Time', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Statistics")
            st.metric("Episodes Run", len(history_df))
            st.metric("Average Reward", f"{history_df['reward'].mean():.1f}")
            st.metric("Best Reward", f"{history_df['reward'].max():.1f}")
            st.metric("Success Rate", f"{(history_df['solved'].sum() / len(history_df) * 100):.1f}%")

            st.markdown("---")

            # Action distribution
            total_left = history_df['left_actions'].sum()
            total_right = history_df['right_actions'].sum()

            fig, ax = plt.subplots(figsize=(5, 5))
            colors = ['#FF6B6B', '#4ECDC4']
            wedges, texts, autotexts = ax.pie(
                [total_left, total_right], 
                labels=['Left', 'Right'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontweight': 'bold'}
            )
            ax.set_title('Overall Action Distribution', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        # Show data table
        st.subheader("Episode Details")
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Download button
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name=f"ppo_agent_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Run some episodes in the Live Demo tab to see your history here!")

with tab4:
    st.header("Technical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hyperparameter Schedules")

        st.code("""
Learning Rate Schedule (Linear):
  Start: 3e-4 (exploration)
  End:   0.0 (fine-tuning)
  Type:  Linear decay

Epsilon Schedule (Linear):
  Start: 0.2 (large trust region)
  End:   0.1 (conservative)
  Type:  Linear decay

Entropy Coefficient (Exponential):
  Start: 0.01 (exploration)
  End:   0.001 (exploitation)
  Type:  Exponential decay
        """, language="python")

    with col2:
        st.subheader("Network Architecture")

        st.code("""
CartPole Network:
  Input:    4 (state dim)
  Hidden1:  128 neurons (ReLU)
  Hidden2:  128 neurons (ReLU)
  Actor:    2 neurons (Softmax)
  Critic:   1 neuron (Value)

  Total Parameters: ~35,000

  Training:
    - Optimizer: Adam
    - Gamma: 0.99
    - GAE Lambda: 0.95
    - Epochs: 5 per batch
        """, language="python")

    st.markdown("---")

    st.subheader("Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Performance**
        - Expert-level: 475±25
        - 95% success rate
        - Lowest variance (CV=5.2%)
        - 5x sample efficiency
        """)

    with col2:
        st.info("""
        **Transfer Learning**
        - Optimized on LunarLander
        - Transferred to CartPole
        - +137% improvement
        - Environment-agnostic
        """)

with tab5:
    st.header("About This Project")

    st.markdown("""
    ## Week 12: Autonomous RL Agent

    This interactive demo showcases the results of **Week 12** of my **168-day ML Learning Journey**.

    ### Achievements

    - Implemented 3 algorithms: REINFORCE, A2C, PPO
    - Optimized PPO: Dynamic hyperparameter schedules (+25% performance)
    - Rigorous testing: 100+ episodes with statistical validation
    - Expert performance: 475±25 on CartPole (95% success rate)
    - Transfer learning: Validated across environments (+137% improvement)
    - Research paper: ~6,500 word technical analysis

    ### What I Learned

    **Technical Skills:**
    - Deep Reinforcement Learning (PPO, A2C, REINFORCE)
    - PyTorch implementation from scratch
    - Hyperparameter optimization strategies
    - Statistical testing and validation
    - Transfer learning techniques

    **Professional Skills:**
    - Research methodology
    - Technical writing
    - Data visualization
    - Software engineering
    - Portfolio development

    ### Week 12 Timeline

    - **Day 78**: PPO Optimization (3000 episodes, hyperparameter schedules)
    - **Day 79**: Transfer Learning (CartPole validation)
    - **Day 80**: Extensive Testing (100+ episodes, statistical analysis)
    - **Day 81**: Research Paper (~6,500 words, publication-quality)
    - **Day 82**: Interactive Demo (this app!)
    - **Day 83**: Deployment & Videos (coming soon)
    - **Day 84**: Blog Post & Final Polish (coming soon)

    ### Resources

    - Research Paper: `../results/day81/COMPLETE_RESEARCH_PAPER.txt`
    - Training Code: `../day_78_ppo_optimization.ipynb`
    - Testing Results: `../day_80_extensive_testing.ipynb`

    ### Author

    **Audrey**  
    3rd Year Computer Science Student  
    Lyceum of the Philippines University - Laguna  
    Specialization: Game Development & AI

    **Target**: Aerospace AI/ML Internship (Summer 2026)

    ---

    ### Progress

    - **Day**: 82/168 (48.8%)
    - **Week**: 12/24
    - **Weeks Completed**: 11/24

    *Part of my journey to master Machine Learning and AI for aerospace applications*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.2rem; font-weight: bold;'>Autonomous RL Agent Interactive Demo</p>
    <p>Week 12 | Day 82/168 | ML Learning Journey</p>
    <p>Built with Streamlit | PyTorch | Gymnasium</p>
</div>
""", unsafe_allow_html=True)
