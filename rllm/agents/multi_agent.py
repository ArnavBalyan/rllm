from typing import Any, Dict, List, Optional

from rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from rllm.engine.multi_agent_execution_engine import AgentRole


class MultiAgentBase(BaseAgent):
    """Base class for Multi-Agent workflows"""
    
    def __init__(
        self, 
        agent_id: str, 
        role: AgentRole = AgentRole.SPECIALIST,
        system_prompt: str = "",
        **kwargs
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self._trajectory = Trajectory()
        self.multi_agent_context: Dict[str, Any] = {}
        
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Must be implemented by concrete agent classes"""
        raise NotImplementedError("update_from_env must be implemented by concrete agent classes")

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Must be implemented by concrete agent classes"""
        raise NotImplementedError("update_from_model must be implemented by concrete agent classes")

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        """Must be implemented by concrete agent classes"""
        raise NotImplementedError("_parse_model_response must be implemented by concrete agent classes")

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Must be implemented by concrete agent classes"""
        raise NotImplementedError("chat_completions must be implemented by concrete agent classes")

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self) -> None:
        """Must be implemented by concrete agent classes"""
        raise NotImplementedError("reset must be implemented by concrete agent classes")
