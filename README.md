---
title: Blood Env
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
---

# 🩸 BloodEnv: AI-Driven Emergency Blood Logistics

**BloodEnv** is a high-stakes simulation environment designed for the Scaler OpenEnv Hackathon. It challenges AI agents to manage a critical blood supply chain across a network of hospitals to save lives and minimize wastage.

---

## 🚀 The Real-World Challenge
In healthcare, blood is a perishable resource. This environment simulates:
- **Perishability:** Every blood unit has an expiry countdown.
- **Urgency:** Random life-threatening emergencies requiring immediate blood units.
- **Logistics:** Routing blood from surplus nodes to deficit nodes under time pressure.

---

## 🛠️ Environment Specification
Built using the **OpenEnv** framework, this environment follows strict Pydantic modeling for reliable AI interaction.

### 1. Observation Space
- `node_inventories`: Current blood units at each hospital with their expiry days.
- `active_emergencies`: Real-time demand for blood at specific nodes.
- `current_step`: Progress within the 30-step mission.

### 2. Action Space
- `source_node`: Sending hospital.
- `target_node`: Receiving hospital.
- `units`: Number of packets to transfer.

### 3. Reward Function
- **+100.0**: For every emergency successfully met (Life Saved!).
- **-50.0**: For failing to meet an emergency demand.
- **-20.0**: For every unit of blood that expires (Wastage Penalty).

---

## 🧠 Baseline Agent
This environment is evaluated using **Meta Llama-3 (8B/70B)**. 

---
**Developed for Scaler OpenEnv Hackathon 2026** 🥁







