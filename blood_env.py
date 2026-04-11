import random
from typing import Dict, List, Tuple, Any

from models import (
    BloodUnit, Emergency, Action, Observation, State,
    BLOOD_TYPES, COMPATIBILITY
)

# Max possible reward per task — used for honest score normalization
MAX_POSSIBLE_REWARD = {
    "easy_routing":        1500.0,
    "emergency_response":  3000.0,
    "hard_optimization":   5000.0,
}


def can_transfuse(donor_type: str, recipient_type: str) -> bool:
    return recipient_type in COMPATIBILITY.get(donor_type, [])


def find_compatible_units(inventory: List[BloodUnit], needed_type: str, units_needed: int) -> List[BloodUnit]:
    exact      = [u for u in inventory if u.blood_type == needed_type]
    compatible = [u for u in inventory if u.blood_type != needed_type and can_transfuse(u.blood_type, needed_type)]
    exact.sort(key=lambda u: u.expiry_days)
    compatible.sort(key=lambda u: u.expiry_days)
    pool = exact + compatible
    return pool[:units_needed]


class BloodEnv:

    def __init__(self, task_id: str = "easy_routing"):
        assert task_id in MAX_POSSIBLE_REWARD, f"Unknown task_id: {task_id}"
        self.task_id       = task_id
        self.num_nodes     = 5
        self.max_steps     = 30
        self.total_reward  = 0.0
        self.lives_saved   = 0
        self.lives_lost    = 0
        self.units_expired = 0
        self._inventory:   Dict[int, List[BloodUnit]]  = {}
        self._emergencies: Dict[int, List[Emergency]]  = {}
        self.current_step  = 0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self.current_step  = 0
        self.total_reward  = 0.0
        self.lives_saved   = 0
        self.lives_lost    = 0
        self.units_expired = 0
        self._inventory    = {}
        self._emergencies  = {}

        if self.task_id == "easy_routing":
            self._setup_easy()
        elif self.task_id == "emergency_response":
            self._setup_medium()
        else:
            self._setup_hard()

        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        reward = 0.0

        # 1. Process transfer routes
        for route in action.routes:
            src   = route.source_node
            tgt   = route.target_node
            bt    = route.blood_type
            units = route.units

            if src not in self._inventory or tgt not in self._inventory:
                continue
            if units <= 0:
                continue

            available = [u for u in self._inventory[src] if u.blood_type == bt]
            available.sort(key=lambda u: u.expiry_days)
            to_move = available[:units]

            if not to_move:
                continue

            for unit in to_move:
                self._inventory[src].remove(unit)
                self._inventory[tgt].append(unit)

            reward += 5.0 * len(to_move)

        # 2. Resolve emergencies
        for node, emergencies in self._emergencies.items():
            resolved = []
            for emg in emergencies:
                usable = find_compatible_units(
                    self._inventory[node], emg.blood_type, emg.units_needed
                )
                if len(usable) >= emg.units_needed:
                    for unit in usable:
                        self._inventory[node].remove(unit)
                    reward += 100.0 * emg.urgency
                    self.lives_saved += 1
                    resolved.append(emg)
                else:
                    partial = len(usable)
                    reward -= 50.0 * (emg.units_needed - partial) * emg.urgency
                    self.lives_lost += 1

            for emg in resolved:
                emergencies.remove(emg)

        # 3. Age blood — count expiry BEFORE removing (this was the bug)
        for node in self._inventory:
            before        = self._inventory[node]
            expired_count = sum(1 for u in before if u.expiry_days <= 1)
            self.units_expired += expired_count
            reward -= expired_count * 20.0
            self._inventory[node] = [
                BloodUnit(blood_type=u.blood_type, expiry_days=u.expiry_days - 1)
                for u in before if u.expiry_days > 1
            ]

        self.current_step += 1
        self.total_reward += reward
        done = self.current_step >= self.max_steps

        if not done:
            self._spawn_emergencies()

        # Score MUST be strictly between 0 and 1 (not 0.0, not 1.0)
        max_r = MAX_POSSIBLE_REWARD[self.task_id]
        raw   = self.total_reward / max_r
        # Clamp to open interval (0.01, 0.99)
        score = max(0.01, min(0.99, 0.5 + raw * 0.49))

        info = {
            "score":         float(score),
            "lives_saved":   self.lives_saved,
            "lives_lost":    self.lives_lost,
            "units_expired": self.units_expired,
        }

        return self._get_obs(), reward, done, info

    def state(self) -> State:
        return State(
            node_inventories  = self._inventory,
            active_emergencies= self._emergencies,
            current_step      = self.current_step,
            task_id           = self.task_id,
            lives_saved       = self.lives_saved,
            lives_lost        = self.lives_lost,
            units_expired     = self.units_expired,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Observation:
        return Observation(
            node_inventories  = self._inventory,
            active_emergencies= self._emergencies,
            current_step      = self.current_step,
            max_steps         = self.max_steps,
            task_id           = self.task_id,
            lives_saved       = self.lives_saved,
            lives_lost        = self.lives_lost,
            units_expired     = self.units_expired,
        )

    def _make_unit(self, blood_type: str = None, min_expiry: int = 2, max_expiry: int = 7) -> BloodUnit:
        bt     = blood_type or random.choice(BLOOD_TYPES)
        expiry = random.randint(min_expiry, max_expiry)
        return BloodUnit(blood_type=bt, expiry_days=expiry)

    def _make_emergency(self, blood_type: str = None, urgency: int = 1) -> Emergency:
        bt    = blood_type or random.choice(BLOOD_TYPES)
        units = random.randint(1, 3)
        return Emergency(blood_type=bt, units_needed=units, urgency=urgency)

    # ------------------------------------------------------------------
    # Task setups
    # ------------------------------------------------------------------

    def _setup_easy(self):
        common_type = "O+"
        for i in range(self.num_nodes):
            self._inventory[i]   = [self._make_unit(blood_type=common_type, min_expiry=5, max_expiry=10)
                                     for _ in range(10)]
            self._emergencies[i] = []
        self._emergencies[0] = [Emergency(blood_type=common_type, units_needed=3, urgency=1)]

    def _setup_medium(self):
        for i in range(self.num_nodes):
            self._inventory[i]   = [self._make_unit(min_expiry=2, max_expiry=5) for _ in range(6)]
            self._emergencies[i] = []
        for node in random.sample(range(self.num_nodes), 3):
            self._emergencies[node] = [self._make_emergency(urgency=random.randint(1, 2))]

    def _setup_hard(self):
        rare_types = ["AB-", "B-", "O-"]
        for i in range(self.num_nodes):
            self._inventory[i]   = [self._make_unit(min_expiry=1, max_expiry=3) for _ in range(4)]
            self._emergencies[i] = []
        for node in range(self.num_nodes):
            urgency = random.choice([2, 3])
            bt      = random.choice(rare_types + BLOOD_TYPES)
            self._emergencies[node] = [Emergency(blood_type=bt, units_needed=random.randint(2, 4), urgency=urgency)]

    def _spawn_emergencies(self):
        if self.task_id == "easy_routing":
            if self.current_step % 5 == 0:
                node = random.randint(0, self.num_nodes - 1)
                self._emergencies[node].append(self._make_emergency(blood_type="O+", urgency=1))

        elif self.task_id == "emergency_response":
            if self.current_step % random.choice([2, 3]) == 0:
                node = random.randint(0, self.num_nodes - 1)
                self._emergencies[node].append(self._make_emergency(urgency=random.randint(1, 2)))

        else:
            node = random.randint(0, self.num_nodes - 1)
            self._emergencies[node].append(self._make_emergency(urgency=random.choice([2, 3])))
            if random.random() < 0.4:
                node2 = random.randint(0, self.num_nodes - 1)
                self._emergencies[node2].append(self._make_emergency(urgency=3))
