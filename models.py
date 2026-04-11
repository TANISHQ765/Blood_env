
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Compatibility matrix: donor -> list of compatible recipients
COMPATIBILITY = {
    "O-":  ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],  # universal donor
    "O+":  ["O+", "A+", "B+", "AB+"],
    "A-":  ["A-", "A+", "AB-", "AB+"],
    "A+":  ["A+", "AB+"],
    "B-":  ["B-", "B+", "AB-", "AB+"],
    "B+":  ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}


class BloodUnit(BaseModel):
    blood_type: str = Field(..., description="Blood type e.g. O-, A+")
    expiry_days: int = Field(..., description="Days remaining before expiry")


class Emergency(BaseModel):
    blood_type: str = Field(..., description="Required blood type")
    units_needed: int = Field(..., description="Number of units needed")
    urgency: int = Field(default=1, description="1=normal, 2=urgent, 3=critical")


class RouteDirective(BaseModel):
    source_node: int = Field(..., description="Hospital ID sending blood")
    target_node: int = Field(..., description="Hospital ID receiving blood")
    blood_type: str = Field(..., description="Blood type being transferred")
    units: int = Field(..., description="Number of units to transfer")


class Action(BaseModel):
    routes: List[RouteDirective] = Field(default_factory=list)


class NodeInventory(BaseModel):
    node_id: int
    units: List[BloodUnit] = Field(default_factory=list)


class Observation(BaseModel):
    node_inventories: Dict[int, List[BloodUnit]] = Field(
        ..., description="Blood units at each node with type and expiry"
    )
    active_emergencies: Dict[int, List[Emergency]] = Field(
        ..., description="Active emergencies at each node"
    )
    current_step: int
    max_steps: int
    task_id: str
    lives_saved: int = 0
    lives_lost: int = 0
    units_expired: int = 0


class State(BaseModel):
    node_inventories: Dict[int, List[BloodUnit]]
    active_emergencies: Dict[int, List[Emergency]]
    current_step: int
    task_id: str
    lives_saved: int
    lives_lost: int
    units_expired: int
