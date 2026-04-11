
"""
BloodEnv Inference Script
=========================
Calls the deployed HF Space server via HTTP (OpenEnv standard),
uses an LLM to decide blood routing actions each step,
and emits structured [START] / [STEP] / [END] logs.
Environment variables required:
  API_BASE_URL   - LLM endpoint (default: HuggingFace router)
  MODEL_NAME     - Model identifier
  HF_TOKEN       - HuggingFace / API key
  BLOOD_ENV_URL  - Base URL of the deployed BloodEnv HF Space
  TASK_ID        - (optional) run a single task; otherwise runs all 3
"""

import os
import json
import re
import httpx
from openai import OpenAI

# ── LLM config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Environment server config ─────────────────────────────────────────────────
BLOOD_ENV_URL = os.getenv("BLOOD_ENV_URL", "http://localhost:8000")
MAX_STEPS     = 30
BENCHMARK     = "blood_env"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Logging helpers ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # action_str must be a compact single-line string — no JSON blobs
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Server calls ───────────────────────────────────────────────────────────────

def server_reset(task_id: str) -> dict:
    resp = httpx.post(f"{BLOOD_ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def server_step(action: dict) -> dict:
    resp = httpx.post(f"{BLOOD_ENV_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── LLM action generation ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI managing emergency blood logistics across 5 hospitals (nodes 0-4).
Each turn you receive the current hospital inventories and active emergencies.
You must output a JSON action deciding which blood units to transfer between hospitals.
Blood type compatibility rules (donor -> recipient):
- O- is universal donor (can give to anyone)
- O+ can give to: O+, A+, B+, AB+
- A- can give to: A-, A+, AB-, AB+
- A+ can give to: A+, AB+
- B- can give to: B-, B+, AB-, AB+
- B+ can give to: B+, AB+
- AB- can give to: AB-, AB+
- AB+ can give to: AB+ only
Strategy:
1. First check which nodes have emergencies and what blood type they need
2. Find a nearby node with compatible blood (prefer exact type, O- as last resort)
3. Transfer blood to meet emergencies — prioritize CRITICAL (urgency=3) first
4. Avoid letting blood expire — move blood with low expiry_days to nodes with demand
Output ONLY valid JSON, no markdown, no explanation:
{"routes": [{"source_node": 0, "target_node": 1, "blood_type": "O+", "units": 2}]}
If no transfer is needed, output: {"routes": []}
"""


def build_prompt(obs: dict, step: int) -> str:
    lines = [f"Step {step}/{obs.get('max_steps', 30)}"]
    lines.append(f"Lives saved so far: {obs.get('lives_saved', 0)} | Lost: {obs.get('lives_lost', 0)}")
    lines.append("")
    lines.append("=== ACTIVE EMERGENCIES ===")
    emergencies = obs.get("active_emergencies", {})
    has_emergency = False
    for node_id, emgs in emergencies.items():
        if emgs:
            has_emergency = True
            for e in emgs:
                urgency_label = {1: "NORMAL", 2: "URGENT", 3: "CRITICAL"}.get(e.get("urgency", 1), "NORMAL")
                lines.append(
                    f"  Node {node_id}: needs {e['units_needed']}x {e['blood_type']} [{urgency_label}]"
                )
    if not has_emergency:
        lines.append("  None currently")

    lines.append("")
    lines.append("=== HOSPITAL INVENTORIES ===")
    inventories = obs.get("node_inventories", {})
    for node_id, units in inventories.items():
        summary: dict[str, int] = {}
        for u in units:
            bt = u["blood_type"]
            summary[bt] = summary.get(bt, 0) + 1
        inv_str = ", ".join(f"{cnt}x{bt}" for bt, cnt in sorted(summary.items())) or "EMPTY"
        lines.append(f"  Node {node_id}: {inv_str}")

    lines.append("")
    lines.append("Decide your transfers. Output JSON only:")
    return "\n".join(lines)


def get_action(obs: dict, step: int) -> tuple[dict, str]:
    """Ask LLM for action. Returns (action_dict, compact_action_str)."""
    prompt = build_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        cleaned = re.sub(r"```json\n?|```\n?", "", raw).strip()
        action_dict = json.loads(cleaned)
        # Validate structure
        if "routes" not in action_dict:
            action_dict = {"routes": []}
    except Exception as exc:
        print(f"[DEBUG] LLM/parse error: {exc}", flush=True)
        action_dict = {"routes": []}

    # Build compact single-line action string for the log
    routes = action_dict.get("routes", [])
    if routes:
        parts = [f"{r['source_node']}->{r['target_node']}:{r.get('units',1)}x{r.get('blood_type','?')}"
                 for r in routes]
        action_str = "routes=[" + ",".join(parts) + "]"
    else:
        action_str = "routes=[]"

    return action_dict, action_str

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = server_reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict, action_str = get_action(obs, step)

            try:
                result = server_step(action_dict)
                obs       = result.get("observation", obs)
                reward    = float(result.get("reward", 0.0))
                done      = bool(result.get("done", False))
                info      = result.get("info", {})
                error_msg = None
            except Exception as exc:
                reward    = 0.0
                done      = True
                info      = {}
                error_msg = str(exc)[:80]

            rewards.append(reward)
            steps_taken = step
            score = float(info.get("score", 0.0))

            log_step(step=step, action_str=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        success = score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task_env = os.getenv("TASK_ID")
    if task_env:
        run_episode(task_env)
    else:
        for task in ["easy_routing", "emergency_response", "hard_optimization"]:
            run_episode(task)
