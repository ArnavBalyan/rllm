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


class ProposerAgent(MultiAgentBase):
    """Agent that proposes initial solutions in Chain of Experts"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a Proposer Agent in a Chain of Experts workflow. Your role is to:
1. Analyze the given problem thoroughly and comprehensively
2. Propose initial solutions, approaches, or strategies
3. Provide clear reasoning for your proposals
4. Be creative and consider multiple angles
5. Set up a strong foundation for the next agent in the chain

Your output will be passed to the next agent in the chain, so provide well-structured proposals that other agents can build upon."""
        
        super().__init__(role=AgentRole.PROPOSER, system_prompt=system_prompt, **kwargs)


class CriticAgent(MultiAgentBase):
    """Agent that critiques and refines solutions in Chain of Experts"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a Critic Agent in a Chain of Experts workflow. Your role is to:
1. Carefully analyze proposals and solutions from the previous agent in the chain
2. Identify potential flaws, weaknesses, or areas for improvement
3. Provide constructive criticism and specific suggestions
4. Ensure logical consistency and completeness
5. Refine and improve upon the previous agent's work

Review the previous agent's output critically but constructively, building upon their work to create a better solution."""
        
        super().__init__(role=AgentRole.CRITIC, system_prompt=system_prompt, **kwargs)


class JudgeAgent(MultiAgentBase):
    """Agent that makes final decisions in Chain of Experts"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are a Judge Agent in a Chain of Experts workflow. Your role is to:
1. Review all previous agent responses in the chain
2. Weigh the merits of different proposals, critiques, and refinements
3. Make final decisions or synthesize the best elements from the chain
4. Provide clear justification for your final answer
5. Deliver a definitive, well-reasoned conclusion

Be decisive while acknowledging the valuable contributions of previous agents in the chain. Your decision is final."""
        
        super().__init__(role=AgentRole.JUDGE, system_prompt=system_prompt, **kwargs)


class SpecialistAgent(MultiAgentBase):
    """Agent with domain-specific expertise for Chain of Experts"""
    
    def __init__(self, specialty: str = "", **kwargs):
        self.specialty = specialty
        system_prompt = f"""You are a Specialist Agent with expertise in: {specialty}.
Your role in the Chain of Experts workflow is to:
1. Apply your specialized knowledge to the problem
2. Provide domain-specific insights and solutions
3. Explain technical concepts clearly for other agents in the chain
4. Highlight important domain-specific considerations
5. Build upon previous agents' work with your specialized expertise

Leverage your expertise while remaining collaborative and preparing well-structured output for the next agent in the chain."""
        
        super().__init__(role=AgentRole.SPECIALIST, system_prompt=system_prompt, **kwargs)


class AggregatorAgent(MultiAgentBase):
    """Agent that combines and synthesizes multiple inputs in Chain of Experts"""
    
    def __init__(self, **kwargs):
        system_prompt = """You are an Aggregator Agent in a Chain of Experts workflow. Your role is to:
1. Combine insights from all previous agents in the chain
2. Identify common themes and reconcile differences
3. Synthesize a comprehensive solution from the chain of expert inputs
4. Present a unified, coherent final answer
5. Ensure all valuable contributions from the chain are represented

Focus on integration and synthesis of the diverse perspectives provided by previous agents in the chain."""
        
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


class MathExpertAgent(SpecialistAgent):
    """Specialized agent for mathematical problems in Chain of Experts"""
    
    def __init__(self, **kwargs):
        super().__init__(specialty="Mathematics", **kwargs)
        self.system_prompt += """

As a mathematics expert in the Chain of Experts, you should:
- Show step-by-step solutions building on previous agents' work
- Use proper mathematical notation and reasoning
- Double-check calculations and verify logical consistency
- Explain mathematical concepts clearly for subsequent agents
- Consider multiple solution approaches and identify the most robust one"""


class CodeExpertAgent(SpecialistAgent):
    """Specialized agent for coding problems in Chain of Experts"""
    
    def __init__(self, **kwargs):
        super().__init__(specialty="Software Engineering", **kwargs)
        self.system_prompt += """

As a coding expert in the Chain of Experts, you should:
- Write clean, efficient code building on previous agents' analysis
- Include proper error handling and edge case considerations
- Add meaningful comments and documentation
- Follow best practices and design patterns
- Test and validate your solutions
- Explain your implementation choices for subsequent agents"""
    
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
    """Specialized agent for logical reasoning and analysis in Chain of Experts"""
    
    def __init__(self, **kwargs):
        super().__init__(specialty="Logical Reasoning", **kwargs)
        self.system_prompt += """

As a reasoning expert in the Chain of Experts, you should:
- Break down complex problems into logical steps
- Identify assumptions and premises from previous agents' work
- Apply logical principles systematically
- Check for logical fallacies and inconsistencies
- Consider alternative perspectives and approaches
- Provide clear justification for conclusions
- Structure your reasoning for the next agent in the chain""" 