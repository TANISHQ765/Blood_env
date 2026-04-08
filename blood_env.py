import random
from typing import Dict, List, Tuple, Any
from models import Observation, Action, State

class BloodEnv:
    def __init__(self, task_id: str = "easy_routing"):
        self.task_id = task_id
        self.num_nodes = 5
        self.max_steps = 30
        self.reset()

    def reset(self) -> Observation:
        self.current_step = 0
        # Inventory: Node_ID -> [list of expiry days for each unit]
        if self.task_id == "easy_routing":
            self.inventory = {i: [5] * 10 for i in range(self.num_nodes)}
            self.emergencies = {0: 2} # Small emergency
        elif self.task_id == "emergency_response":
            self.inventory = {i: [3] * 5 for i in range(self.num_nodes)}
            self.emergencies = {i: random.randint(0, 5) for i in range(self.num_nodes)}
        else: # hard_optimization
            self.inventory = {i: [random.randint(1, 3) for _ in range(5)] for i in range(self.num_nodes)}
            self.emergencies = {i: random.randint(5, 10) for i in range(self.num_nodes)}
            
        return self._get_obs()

    def _get_obs(self) -> Observation:
        return Observation(
            node_inventories=self.inventory,
            active_emergencies=self.emergencies,
            current_step=self.current_step,
            task_id=self.task_id
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        reward = 0.0
        # 1. Process Transfers
        for route in action.routes:
            if route.source_node in self.inventory and len(self.inventory[route.source_node]) >= route.units:
                units_to_move = self.inventory[route.source_node][:route.units]
                self.inventory[route.source_node] = self.inventory[route.source_node][route.units:]
                self.inventory[route.target_node].extend(units_to_move)
                reward += 10.0 # Small reward for routing

        # 2. Match Emergencies
        for node, demand in self.emergencies.items():
            if len(self.inventory[node]) >= demand:
                self.inventory[node] = self.inventory[node][demand:]
                reward += 100.0 # Big reward for saving lives
                self.emergencies[node] = 0
            else:
                reward -= 50.0 # Penalty for failing emergency

        # 3. Age blood and check expiry
        for node in self.inventory:
            self.inventory[node] = [d - 1 for d in self.inventory[node] if d > 1]
            expired = sum(1 for d in self.inventory[node] if d <= 1)
            reward -= (expired * 20.0) # Penalty for waste

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # New emergencies for next step
        if not done:
            self.emergencies = {random.randint(0, 4): random.randint(1, 5)}

        return self._get_obs(), reward, done, {}

    def state(self) -> State:
        return State(
            node_inventories=self.inventory,
            active_emergencies=self.emergencies,
            current_step=self.current_step
        )