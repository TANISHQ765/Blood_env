"""Blood Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BloodAction, BloodObservation


class BloodEnv(
    EnvClient[BloodAction, BloodObservation, State]
):
    """
    Client for the Blood Env Environment.
    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    Example:
        >>> # Connect to a running server
        >>> with BloodEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(BloodAction(message="Hello!"))
        ...     print(result.observation.echoed_message)
    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = BloodEnv.from_docker_image("blood_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(BloodAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: BloodAction) -> Dict:
        """
        Convert BloodAction to JSON payload for step message.
        Args:
            action: BloodAction instance
        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[BloodObservation]:
        """
        Parse server response into StepResult[BloodObservation].
        Args:
            payload: JSON response data from server
        Returns:
            StepResult with BloodObservation
        """
        obs_data = payload.get("observation", {})
        observation = BloodObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        Args:
            payload: JSON response from state request
        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
