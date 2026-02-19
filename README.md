#  Fake News Detection — Human–AI Interaction Simulator (SDT)

This project is an interactive simulation framework for studying how humans and AI systems jointly detect fake news using **Signal Detection Theory (SDT)**.

Instead of treating AI as a black-box classifier, the simulator models **decision-making behavior**: sensitivity, bias, thresholds, and how human and AI signals combine to influence final judgments.

The system is deployed as a **Streamlit web app**, allowing real-time exploration of different Human–AI collaboration strategies.

---

## What This Project Does

- Simulates fake-news detection as a signal detection process  
- Models Combined Human–AI architectures  
- Computes key SDT metrics:
  - d′ (sensitivity)  
  - False alarm rate  
  - Hit rate  
  - Accuracy  
- Visualizes how changing parameters affects outcomes  

---

## Architectures Implemented  
- Human-first → AI assist  
- AI-first → Human review  
- Parallel Human–AI fusion  

Each architecture allows adjusting:
- Sensitivity (d′)  
- Decision thresholds  
- Noise assumptions  
---

## Key Files

| File | Purpose |
|-----|--------|
| `simulation.py` | Core SDT math and probability functions |
| `Ai_acts_fake.py` | AI behavior when stimulus is fake |
| `Ai_acts_real.py` | AI behavior when stimulus is real |
| `2_thresholds.py` | Dual-threshold decision logic |
| `simulationapp.py` | Streamlit web application |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python version specification |


