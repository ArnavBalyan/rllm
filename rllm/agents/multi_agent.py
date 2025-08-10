from typing import Any, Dict, List, Optional

from rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from rllm.engine.multi_agent_execution_engine import AgentRole


class MultiAgentBase(BaseAgent):
    """Base class for agents that work in multi-agent workflows"""
    
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
        
        # Add multi-agent context if available
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
        # Extract multi-agent context if present
        if isinstance(observation, dict):
            if "collaboration_prompt" in observation:
                self.multi_agent_context["collaboration"] = observation["collaboration_prompt"]
            if "debate_prompt" in observation:
                self.multi_agent_context["debate"] = observation["debate_prompt"]
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
        """Format multi-agent context into a readable prompt"""
        context_parts = []
        
        if "collaboration" in self.multi_agent_context:
            context_parts.append(f"COLLABORATION CONTEXT:\n{self.multi_agent_context['collaboration']}")
        
        if "debate" in self.multi_agent_context:
            context_parts.append(f"DEBATE CONTEXT:\n{self.multi_agent_context['debate']}")
        
        if "previous_agents" in self.multi_agent_context:
            agents_info = self.multi_agent_context["previous_agents"]
            context_parts.append(f"PREVIOUS AGENTS: {len(agents_info)} agents have provided input")
        
        return "\n\n".join(context_parts)
    
    def _parse_action(self, response: str) -> Any:
        """Parse action from model response - override in subclasses"""
        return response


class ProposerAgent(MultiAgentBase):
    """Agent that proposes initial solutions"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a Proposer Agent. Your role is to:
1. Analyze the given problem thoroughly
2. Propose initial solutions or approaches
3. Provide clear reasoning for your proposals
4. Be creative and consider multiple angles

Provide well-structured proposals that other agents can build upon."""
        
        super().__init__(role=AgentRole.PROPOSER, system_prompt=system_prompt, **kwargs)


class CriticAgent(MultiAgentBase):
    """Agent that critiques and refines solutions"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a Critic Agent. Your role is to:
1. Carefully analyze proposals from other agents
2. Identify potential flaws, weaknesses, or improvements
3. Provide constructive criticism and suggestions
4. Ensure logical consistency and completeness

Be thorough but constructive in your analysis."""
        
        super().__init__(role=AgentRole.CRITIC, system_prompt=system_prompt, **kwargs)


class JudgeAgent(MultiAgentBase):
    """Agent that makes final decisions"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a Judge Agent. Your role is to:
1. Review all previous agent responses
2. Weigh the merits of different proposals and critiques
3. Make final decisions or synthesize the best elements
4. Provide clear justification for your final answer

Be decisive while acknowledging the contributions of other agents."""
        
        super().__init__(role=AgentRole.JUDGE, system_prompt=system_prompt, **kwargs)


class SpecialistAgent(MultiAgentBase):
    """Agent with domain-specific expertise"""
    
    def __init__(self, specialty: str = "", **kwargs):
        self.specialty = specialty
        system_prompt = f"""You are a Specialist Agent with expertise in: {specialty}.
Your role is to:
1. Apply your specialized knowledge to the problem
2. Provide domain-specific insights and solutions
3. Explain technical concepts clearly for other agents
4. Highlight important domain-specific considerations

Leverage your expertise while remaining collaborative."""
        
        super().__init__(role=AgentRole.SPECIALIST, system_prompt=system_prompt, **kwargs)


class AggregatorAgent(MultiAgentBase):
    """Agent that combines and synthesizes multiple inputs"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are an Aggregator Agent. Your role is to:
1. Combine insights from multiple expert agents
2. Identify common themes and reconcile differences
3. Synthesize a comprehensive solution
4. Present a unified, coherent final answer

Focus on integration and synthesis of diverse perspectives."""
        
        super().__init__(role=AgentRole.AGGREGATOR, system_prompt=system_prompt, **kwargs)
    
    def _format_multi_agent_context(self) -> str:
        """Enhanced context formatting for aggregation"""
        base_context = super()._format_multi_agent_context()
        
        # Add specific formatting for aggregation
        if "previous_agents" in self.multi_agent_context:
            agents_info = self.multi_agent_context["previous_agents"]
            
            aggregation_prompt = "\n\nEXPERT RESPONSES TO SYNTHESIZE:\n"
            for i, agent_data in enumerate(agents_info):
                aggregation_prompt += f"\nExpert {i+1} ({agent_data.get('agent_id', 'Unknown')}):\n"
                aggregation_prompt += f"{agent_data.get('response', '')}\n"
                aggregation_prompt += "-" * 50 + "\n"
            
            aggregation_prompt += "\nPlease synthesize these expert opinions into a comprehensive solution."
            
            return base_context + aggregation_prompt
        
        return base_context


class DebaterAgent(MultiAgentBase):
    """Agent designed for debate scenarios"""
    
    def __init__(self, position: str = "", **kwargs):
        self.position = position
        system_prompt = f"""You are a Debater Agent taking the position: {position}.
Your role is to:
1. Argue persuasively for your assigned position
2. Address counterarguments from other debaters
3. Use evidence and logical reasoning
4. Remain respectful while being forceful in your arguments

Defend your position while engaging constructively with opponents."""
        
        super().__init__(role=AgentRole.SPECIALIST, system_prompt=system_prompt, **kwargs)
    
    def _format_multi_agent_context(self) -> str:
        """Enhanced context formatting for debates"""
        base_context = super()._format_multi_agent_context()
        
        if "debate" in self.multi_agent_context:
            debate_context = "\n\nDEBATE CONTEXT:\n"
            debate_context += f"Your position: {self.position}\n"
            debate_context += self.multi_agent_context["debate"]
            return base_context + debate_context
        
        return base_context


class MathExpertAgent(SpecialistAgent):
    """Specialized agent for mathematical problems"""
    
    def __init__(self, **kwargs):
        super().__init__(specialty="Mathematics", **kwargs)
        self.system_prompt += """

As a mathematics expert, you should:
- Show step-by-step solutions
- Use proper mathematical notation
- Double-check calculations
- Explain mathematical concepts clearly
- Consider multiple solution approaches"""


class CodeExpertAgent(SpecialistAgent):
    """Specialized agent for coding problems"""
    
    def __init__(self, **kwargs):
        super().__init__(specialty="Software Engineering", **kwargs)
        self.system_prompt += """

As a coding expert, you should:
- Write clean, efficient code
- Include proper error handling
- Add meaningful comments
- Consider edge cases
- Follow best practices and design patterns
- Test your solutions"""
    
    def _parse_action(self, response: str) -> Any:
        """Extract code from response"""
        # Simple code extraction - could be more sophisticated
        if "```" in response:
            code_blocks = response.split("```")
            for i, block in enumerate(code_blocks):
                if i % 2 == 1:  # Odd indices are code blocks
                    # Remove language identifier if present
                    lines = block.strip().split('\n')
                    if lines and lines[0].strip() in ['python', 'py', 'javascript', 'js', 'java', 'cpp', 'c++']:
                        return '\n'.join(lines[1:])
                    return block.strip()
        
        return response


class ReasoningExpertAgent(SpecialistAgent):
    """Specialized agent for logical reasoning and analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(specialty="Logical Reasoning", **kwargs)
        self.system_prompt += """

As a reasoning expert, you should:
- Break down complex problems into steps
- Identify assumptions and premises
- Apply logical principles systematically
- Check for logical fallacies
- Consider alternative perspectives
- Provide clear justification for conclusions""" 