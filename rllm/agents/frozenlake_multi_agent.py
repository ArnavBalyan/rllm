import copy
import logging
import re
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.multi_agent import MultiAgentBase
from rllm.agents.frozenlake_agent import FrozenLakeAgent

logger = logging.getLogger(__name__)


class FrozenLakeProposerAgent(FrozenLakeAgent, MultiAgentBase):
    """
    Proposer agent for FrozenLake Chain of Experts.
    
    Role: Analyzes the initial state and proposes a high-level strategy.
    Focus: Safe path planning and obstacle identification.
    """
    
    SYSTEM_PROMPT = """You are the PROPOSER in a Chain of Experts for FrozenLake navigation.
Your role: Analyze the frozen lake and propose a high-level strategic approach. Ensure your response is in 10 words or less.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.
No MATTER WHAT HAPPENS, ENSURE YOUR THINGKING AND RESPONSE IS IN 10 WORDS OR LESS ALWAYS!

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        FrozenLakeAgent.__init__(self, max_steps=max_steps, use_accumulate_history=True, **kwargs)
        MultiAgentBase.__init__(self, agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    # All methods inherited from MultiAgentBase and FrozenLakeAgent


class FrozenLakeExpertAgent(FrozenLakeAgent, MultiAgentBase):
    """
    Expert agent for FrozenLake Chain of Experts.
    
    Role: Takes the Proposer's strategy and makes the tactical move decision.
    Focus: Precise movement execution with risk assessment.
    """
    
    SYSTEM_PROMPT = """You are the EXPERT in a Chain of Experts for FrozenLake navigation.

As the EXPERT, you should:
1. Assess immediate movement options and risks
2. Consider slip probability and backup plans
3. Make the precise tactical decision

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.

"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        FrozenLakeAgent.__init__(self, max_steps=max_steps, use_accumulate_history=True, **kwargs)
        MultiAgentBase.__init__(self, agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    # All methods inherited from MultiAgentBase and FrozenLakeAgent


class FrozenLakeJudgeAgent(FrozenLakeAgent, MultiAgentBase):
    """
    Judge agent for FrozenLake Chain of Experts.
    
    Role: Reviews both Proposer and Expert recommendations to make the final decision.
    Focus: Safety validation and optimal choice selection.
    """
    
    SYSTEM_PROMPT = """You are the JUDGE in a Chain of Experts for FrozenLake navigation.
Your role: Review both the Proposer's strategy to make the final move.

As the JUDGE, you should:
1. Review the Proposer's strategic analysis
2. Validate safety and optimality of the proposed move
3. Make the final authoritative decision

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        FrozenLakeAgent.__init__(self, max_steps=max_steps, use_accumulate_history=True, **kwargs)
        MultiAgentBase.__init__(self, agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    # All methods inherited from MultiAgentBase and FrozenLakeAgent
