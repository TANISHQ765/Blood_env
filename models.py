from pydantic import BaseModel, Field
from typing import List, Dict

class RouteDirective(BaseModel):
    source_node: int = Field(..., description="ID of the hospital sending blood")
    target_node: int = Field(..., description="ID of the hospital receiving blood")
    units: int = Field(..., description="Number of blood units to transfer")

class Action(BaseModel):
    routes: List[RouteDirective]

class Observation(BaseModel):
    node_inventories: Dict[int, List[int]] = Field(..., description="Blood units at each node with expiry days")
    active_emergencies: Dict[int, int] = Field(..., description="Units needed at each node immediately")
    current_step: int
    task_id: str

class State(BaseModel):
    node_inventories: Dict[int, List[int]]
    active_emergencies: Dict[int, int]
    current_step: int