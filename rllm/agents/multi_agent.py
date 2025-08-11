from typing import Any, Dict, List, Optional

from rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from rllm.engine.multi_agent_execution_engine import AgentRole


class MultiAgentBase(BaseAgent):
    """Base class for agents that work in Chain of Experts workflows"""
    
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
        
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Convert internal state to chat completions format"""
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add Chain of Experts context if available
        if self.multi_agent_context:
            context_content = self._format_multi_agent_context()
            if context_content:
                messages.append({"role": "system", "content": context_content})

        # Add conversation history
        for step in self._trajectory.steps:
            if step.observation:
                messages.append({"role": "user", "content": str(step.observation)})
            if step.model_response:
                messages.append({"role": "assistant", "content": step.model_response})
                
        return messages
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
    
    def reset(self):
        """Reset agent state"""
        self._trajectory = Trajectory()
        self.multi_agent_context = {}
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update agent state from environment"""
        # Extract Chain of Experts context if present
        if isinstance(observation, dict):
            if "collaboration_prompt" in observation:
                self.multi_agent_context["collaboration"] = observation["collaboration_prompt"]
            if "previous_agents" in observation:
                self.multi_agent_context["previous_agents"] = observation["previous_agents"]
        
        # Create or update current step
        if not self._trajectory.steps or self._trajectory.steps[-1].done:
            # Start new step
            step = Step(observation=observation, reward=reward, done=done, info=info)
            self._trajectory.steps.append(step)
        else:
            # Update current step
            current_step = self._trajectory.steps[-1]
            current_step.observation = observation
            current_step.reward = reward
            current_step.done = done
            current_step.info.update(info)
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        """Update agent state from model response"""
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.model_response = response
            current_step.action = self._parse_action(response)
        
        return Action(action=response)
    
    def get_current_state(self) -> Optional[Step]:
        """Get current step"""
        return self._trajectory.steps[-1] if self._trajectory.steps else None
    
    def _format_multi_agent_context(self) -> str:
        """Format Chain of Experts context into a readable prompt"""
        context_parts = []
        
        if "collaboration" in self.multi_agent_context:
            context_parts.append(f"CHAIN OF EXPERTS CONTEXT:\n{self.multi_agent_context['collaboration']}")
        
        if "previous_agents" in self.multi_agent_context:
            agents_info = self.multi_agent_context["previous_agents"]
            context_parts.append(f"PREVIOUS AGENTS IN CHAIN: {len(agents_info)} agents have provided input")
        
        return "\n\n".join(context_parts)
    
    def _parse_action(self, response: str) -> Any:
        """Parse action from model response - override in subclasses"""
        return response
