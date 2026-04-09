import os
import sys
import json
import re
from openai import OpenAI
from blood_env import BloodEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_evaluation(task_id):
    env = BloodEnv(task_id=task_id)
    obs = env.reset()

    
    print(f"[START] task={task_id} env=blood_env model={MODEL_NAME}", flush=True)

    total_rewards = []
    done = False
    step_count = 0

    try:
        while not done and step_count < 30:
            step_count += 1
            
            prompt = f"State: {obs.json()}. Output ONLY a valid JSON Action with routes. Example: {{\"routes\": [{{\"source_node\": 0, \"target_node\": 1, \"units\": 2}}]}}"
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an emergency blood logistics AI. Output pure JSON without markdown wrappers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1 
            )
            
            raw_content = response.choices[0].message.content
            cleaned_content = re.sub(r'```json\n?|```\n?', '', raw_content).strip()
            
            try:
                action_dict = json.loads(cleaned_content)
                action = Action(**action_dict)
            except (json.JSONDecodeError, ValueError):
                action = Action(routes=[]) 
            
            obs, reward, done, info = env.step(action)
            total_rewards.append(reward)
            
    
            print(f"[STEP] step={step_count} action={action.json()} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

    except Exception as e:
        
        print(f"[END] success=false steps={step_count} score=0.01 rewards=0.00 error={str(e)}", flush=True)
        return

    
    total_sum = sum(total_rewards)
    raw_score = 0.5 + (total_sum / 2000.0)
    final_score = max(0.01, min(0.99, raw_score))
    

    rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
    print(f"[END] success=true steps={step_count} score={final_score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":

    task_env = os.getenv("TASK_ID") or os.getenv("TASK")
    
    if task_env:
        run_evaluation(task_env)
    else:

        tasks = ["easy_routing", "emergency_response", "hard_optimization"]
        for t in tasks:
            run_evaluation(t)
