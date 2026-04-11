---
title: Blood Env
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - healthcare
  - supply-chain
  - blood-bank
---
🩸 BloodEnv — AI Emergency Blood Logistics
BloodEnv is a real-world simulation environment for the OpenEnv framework, where AI agents manage a perishable blood supply chain across a network of hospitals to save lives and minimize wastage.

This is not a game or toy. Blood bank logistics is a genuine operational challenge faced by hospitals worldwide — wrong blood type, expired units, or slow routing kills patients.

🌍 Real-World Motivation
Every year, millions of blood units expire or are wasted due to poor routing and forecasting. Simultaneously, hospitals face critical shortages. An AI agent trained in BloodEnv learns to:

Route the right blood type to patients who need it
Prioritize critical emergencies over routine transfers
Prevent expiry wastage by moving near-expiry units proactively
Balance scarcity vs demand across multiple nodes simultaneously
🛠️ Environment Specification
Observation Space
Field	Type	Description
node_inventories	Dict[int, List[BloodUnit]]	Blood units at each hospital (type + expiry days)
active_emergencies	Dict[int, List[Emergency]]	Active demand per node (type, units needed, urgency 1-3)
current_step	int	Step number within episode
lives_saved	int	Cumulative lives saved
lives_lost	int	Cumulative lives lost
units_expired	int	Cumulative expired units
Action Space
{
  "routes": [
    {
      "source_node": 0,
      "target_node": 2,
      "blood_type": "O-",
      "units": 2
    }
  ]
}
Blood Type Compatibility
Donor	Can Give To
O-	Everyone (universal donor)
O+	O+, A+, B+, AB+
A-	A-, A+, AB-, AB+
A+	A+, AB+
B-	B-, B+, AB-, AB+
B+	B+, AB+
AB-	AB-, AB+
AB+	AB+ only
Reward Function
Event	Reward
Emergency fulfilled	+100 × urgency
Emergency failed (per missing unit)	-50 × urgency
Blood unit expired	-20
Valid transfer (per unit moved)	+5
Score is normalized to [0.0, 1.0] using reward / MAX_POSSIBLE_REWARD per task.

🧩 Tasks
1. easy_routing (Easy)
Single blood type (O+), well-stocked inventory (10 units/node), long expiry (5-10 days). One emergency at node 0. Agent just needs to identify and route blood correctly.

Expected baseline score: ~0.72

2. emergency_response (Medium)
Mixed blood types across all nodes, moderate inventory (6 units/node, 2-5 day expiry), multiple simultaneous emergencies with varying urgency. Agent must understand compatibility rules.

Expected baseline score: ~0.41

3. hard_optimization (Hard)
Scarce inventory (4 units/node, 1-3 day expiry), rare blood types (O-, B-, AB-), critical-urgency emergencies at every node every step. Requires expert prioritization and planning ahead to avoid both waste and shortages.

Expected baseline score: ~0.18

🚀 Setup & Usage
Prerequisites
Docker
Python 3.11+
HuggingFace token
Run locally
# Clone the repo
git clone https://github.com/TANISHQ765/Blood_env
cd Blood_env

# Build and run Docker
docker build -t blood_env .
docker run -p 8000:8000 blood_env
Test the endpoints
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_routing"}'

# Take a step
curl -X POST "http://localhost:8000/step?task_id=easy_routing" \
  -H "Content-Type: application/json" \
  -d '{"routes": [{"source_node": 1, "target_node": 0, "blood_type": "O+", "units": 3}]}'

# Get state
curl "http://localhost:8000/state?task_id=easy_routing"
Run baseline inference
export HF_TOKEN=your_token_here
export MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
export BLOOD_ENV_URL=http://localhost:8000

python inference.py
📁 Project Structure
Blood_env/
├── inference.py        # Baseline inference script (required)
├── blood_env.py        # Core environment logic
├── models.py           # Pydantic models (Observation, Action, State)
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # Container definition
├── README.md
└── server/
    └── app.py          # FastAPI server (OpenEnv HTTP interface)
📊 Baseline Scores
Measured using meta-llama/Llama-3.2-3B-Instruct via HuggingFace router:

Task	Score	Lives Saved	Lives Lost
easy_routing	0.720	6	0
emergency_response	0.410	11	4
hard_optimization	0.180	5	18
Developed for Scaler OpenEnv Hackathon 2026 
