import copy
import logging
import re
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.multi_agent import MultiAgentBase
from rllm.agents.frozenlake_agent import FrozenLakeAgent

logger = logging.getLogger(__name__)


class FrozenLakeProposerAgent(MultiAgentBase):
    """
    Proposer agent for FrozenLake Chain of Experts.
    
    Role: Analyzes the initial state and proposes a high-level strategy.
    Focus: Safe path planning and obstacle identification.
    """
    
    SYSTEM_PROMPT = """You are the PROPOSER in a Chain of Experts for FrozenLake navigation.
Your role: Analyze the frozen lake and propose a high-level strategic approach.

FrozenLake Quick Guide:
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Actions: Up | Down | Left | Right

As the PROPOSER, you should:
1. Analyze the current board layout
2. Identify the safest general path direction
3. Note any immediate dangers (holes near player)
4. Suggest a strategic approach (e.g., "move right first to avoid holes", "take indirect path for safety")

Your analysis will help the next expert make the specific move decision.

You should show your strategic thinking and then propose the NEXT ACTION in ``` ```.
The final action MUST be one of: Up, Down, Left, Right.
Focus on SAFETY and STRATEGY rather than just the shortest path.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        super().__init__(agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update proposer agent with environment observation and multi-agent context"""
        # Call parent to handle multi-agent context and update trajectory
        super().update_from_env(observation, reward, done, info, **kwargs)
        
        # The observation is now stored in the trajectory by the parent class
        # No need to manually manage messages - they're handled by chat_completions property
        self.step += 1


class FrozenLakeExpertAgent(MultiAgentBase):
    """
    Expert agent for FrozenLake Chain of Experts.
    
    Role: Takes the Proposer's strategy and makes the tactical move decision.
    Focus: Precise movement execution with risk assessment.
    """
    
    SYSTEM_PROMPT = """You are the EXPERT in a Chain of Experts for FrozenLake navigation.
Your role: Take the Proposer's strategic advice and make the specific tactical move.

FrozenLake Quick Guide:
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Actions: Up | Down | Left | Right

As the EXPERT, you should:
1. Review the Proposer's strategic analysis
2. Assess immediate movement options and risks
3. Consider slip probability and backup plans
4. Make the precise tactical decision

You will receive context from the PROPOSER about their strategic recommendations.
Use this guidance to make the optimal specific move.

You should show your tactical analysis and then output the NEXT ACTION in ``` ```.
The final action MUST be one of: Up, Down, Left, Right.
Focus on PRECISE EXECUTION of the proposed strategy.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        super().__init__(agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update expert agent with environment observation and multi-agent context"""
        # Call parent to handle multi-agent context and update trajectory
        super().update_from_env(observation, reward, done, info, **kwargs)
        
        # The observation is now stored in the trajectory by the parent class
        # No need to manually manage messages - they're handled by chat_completions property
        self.step += 1


class FrozenLakeJudgeAgent(MultiAgentBase):
    """
    Judge agent for FrozenLake Chain of Experts.
    
    Role: Reviews both Proposer and Expert recommendations to make the final decision.
    Focus: Safety validation and optimal choice selection.
    """
    
    SYSTEM_PROMPT = """You are the JUDGE in a Chain of Experts for FrozenLake navigation.
Your role: Review both the Proposer's strategy and Expert's tactical decision to make the final move.

FrozenLake Quick Guide:
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Actions: Up | Down | Left | Right

As the JUDGE, you should:
1. Review the Proposer's strategic analysis
2. Evaluate the Expert's tactical recommendation
3. Validate safety and optimality of the proposed move
4. Make the final authoritative decision

You will receive context from both the PROPOSER and EXPERT.
Your job is to synthesize their input and make the best final decision.

You should show your judgment process and then output the FINAL ACTION in ``` ```.
The final action MUST be one of: Up, Down, Left, Right.
Focus on SAFETY VALIDATION and OPTIMAL CHOICE SELECTION.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        super().__init__(agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update judge agent with environment observation and multi-agent context"""
        # Call parent to handle multi-agent context and update trajectory
        super().update_from_env(observation, reward, done, info, **kwargs)
        
        # The observation is now stored in the trajectory by the parent class
        # No need to manually manage messages - they're handled by chat_completions property
        self.step += 1


class FrozenLakeMultiAgentEnv:
    """
    Enhanced FrozenLake environment for Chain of Experts.
    
    Handles context passing between agents in the chain.
    """
    
    def __init__(self, base_env):
        self.base_env = base_env
        self.multi_agent_context = {}
        self.agent_history = []
    
    def reset_with_input(self, agent_input: Dict[str, Any]):
        """Reset environment with input from previous agents in the chain"""
        self.multi_agent_context = agent_input
        self.agent_history = []
        
        # Extract agent responses for context
        for agent_id, agent_data in agent_input.items():
            if "chat_completions" in agent_data:
                chat_completions = agent_data["chat_completions"]
                if chat_completions:
                    # Get the last assistant message (the agent's response)
                    for completion in reversed(chat_completions):
                        if completion.get("role") == "assistant":
                            self.agent_history.append({
                                "agent": agent_id,
                                "response": completion.get("content", "")
                            })
                            break
        
        return self.base_env.reset()
    
    def reset(self):
        """Standard reset without multi-agent context"""
        self.multi_agent_context = {}
        self.agent_history = []
        return self.base_env.reset()
    
    def step(self, action):
        """Forward step to base environment"""
        return self.base_env.step(action)
    
    def render(self, *args, **kwargs):
        """Forward render to base environment"""
        return self.base_env.render(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward all other attributes to base environment"""
        return getattr(self.base_env, name)
    
    def get_collaboration_prompt(self) -> str:
        """Generate collaboration prompt from agent history"""
        if not self.agent_history:
            return ""
        
        prompt_parts = []
        for entry in self.agent_history:
            agent_name = entry["agent"].replace("_", " ").title()
            response = entry["response"]
            prompt_parts.append(f"{agent_name}:\n{response}\n")
        
        return "\n".join(prompt_parts) 