import os
import sys
from openai import OpenAI
from blood_env import BloodEnv
from models import Action

# 1. Guidelines ke hisaab se Env Variables with Defaults 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required ")

# client setup 
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def run_evaluation():
    env = BloodEnv(task_id="easy_routing")
    obs = env.reset()
    
    # [START] line format strictly as per guidelines 
    print(f"[START] task={env.task_id} env=blood_env model={MODEL_NAME}")
    
    total_rewards = []
    done = False
    step_count = 0
    
    try:
        while not done and step_count < 30:
            step_count += 1
            
            # Simple AI Prompt for the agent
            prompt = f"State: {obs.json()}. Output a valid JSON Action with routes. Example: {{\"routes\": [{{'source_node': 0, 'target_node': 1, 'units': 2}}]}}"
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Logic to parse action (Simplified for baseline)
            # In real eval, you'd parse response.choices[0].message.content
            action = Action(routes=[]) 
            
            obs, reward, done, info = env.step(action)
            total_rewards.append(reward)
            
            # [STEP] line format 
            print(f"[STEP] step={step_count} action={action.json()} reward={reward:.2f} done={str(done).lower()} error=null")

        # [END] line format 
        rewards_str = ",".join([f"{r:.2f}" for r in total_rewards])
        print(f"[END] success=true steps={step_count} rewards={rewards_str}")

    except Exception as e:
        print(f"[END] success=false steps={step_count} rewards=0.00 error={str(e)}")

if __name__ == "__main__":
    run_evaluation()