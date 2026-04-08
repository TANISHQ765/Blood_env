import uvicorn
from fastapi import FastAPI
from openenv.server import create_app
from blood_env import BloodEnv

# Create the OpenEnv FastAPI app
app = create_app(BloodEnv)

def main():
    """Main entry point for multi-mode deployment."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()