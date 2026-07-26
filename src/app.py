import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")

st.title("🧠 Smart MCQ Solver AI")
st.write("Enter an MCQ with 5 options to receive real-time AI predictions from the ensemble pipeline.")

# Form inputs
prompt = st.text_area("Question / Prompt:", "Which technique adapts Large Language Models efficiently using low-rank matrices?")
opt_a = st.text_input("Option A:", "Full Parameter Fine-Tuning")
opt_b = st.text_input("Option B:", "LoRA (Low-Rank Adaptation)")
opt_c = st.text_input("Option C:", "Random Weight Initialization")
opt_d = st.text_input("Option D:", "Grid Search Hyperparameter Tuning")
opt_e = st.text_input("Option E:", "Feature Scaling")

if st.button("Solve MCQ"):
    # Simulated prediction output for Streamlit interface
    logits = np.array([0.10, 0.75, 0.05, 0.05, 0.05])
    probs = F.softmax(torch.tensor(logits), dim=0).numpy()
    
    top3_idx = np.argsort(-probs)[:3]
    labels = ['A', 'B', 'C', 'D', 'E']
    top3_choices = [f"{labels[i]} ({probs[i]*100:.1f}%)" for i in top3_idx]
    
    st.success(f"**Top 3 Predictions:** {' ➔ '.join(top3_choices)}")
    
    st.subheader("Option Confidence Scores")
    for label, prob in zip(labels, probs):
        st.progress(float(prob), text=f"Option {label}: {prob*100:.2f}%")