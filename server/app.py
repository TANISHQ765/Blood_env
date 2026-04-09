import uvicorn
from fastapi import Request, FastAPI
from blood_env import BloodEnv  
from models import Action

app = FastAPI(title="BloodEnv API")

env = BloodEnv()

@app.post("/reset")
async def reset_env(request: Request):
    
    try:
        payload = await request.json()
        task_id = payload.get("task_id", "easy_routing")
    except:
        task_id = "easy_routing"
        
    env.task_id = task_id
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs, 
        "reward": reward, 
        "done": done, 
        "info": info 
    }

@app.get("/")
def health_check():
    return {"status": "BloodEnv Server is Running"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
