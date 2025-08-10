from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from rllm.environments.base.base_env import BaseEnv


class MultiAgentEnv(BaseEnv):
    """Base environment class for multi-agent workflows"""
    
    def __init__(self, task_data: Dict[str, Any] = None, **kwargs):
        super().__init__()
        self.task_data = task_data or {}
        self.multi_agent_context: Dict[str, Any] = {}
        self.agent_history: List[Dict[str, Any]] = []
        self.current_agent_id: Optional[str] = None
        
    def reset(self) -> Tuple[Dict, Dict]:
        """Standard reset without multi-agent input"""
        self.multi_agent_context = {}
        self.agent_history = []
        self.current_agent_id = None
        return self._create_observation(), self._create_info()
    
    def reset_with_input(self, agent_input: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Reset with input from previous agents in the workflow"""
        self.multi_agent_context = agent_input
        
        # Process input from previous agents
        self._process_multi_agent_input(agent_input)
        
        observation = self._create_observation_with_context()
        info = self._create_info_with_context()
        
        return observation, info
    
    def set_current_agent(self, agent_id: str):
        """Set the current agent ID for context tracking"""
        self.current_agent_id = agent_id
    
    def _process_multi_agent_input(self, agent_input: Dict[str, Any]):
        """Process input from previous agents"""
        # Store previous agent outputs
        for source_agent, agent_output in agent_input.items():
            if isinstance(agent_output, dict) and "response" in agent_output:
                self.agent_history.append({
                    "agent_id": source_agent,
                    "response": agent_output["response"],
                    "timestamp": agent_output.get("timestamp", None)
                })
        
        # Handle debate context if present
        if "debate_context" in agent_input:
            self.multi_agent_context["debate"] = agent_input["debate_context"]
    
    @abstractmethod
    def _create_observation(self) -> Dict[str, Any]:
        """Create observation for single-agent mode"""
        pass
    
    @abstractmethod  
    def _create_info(self) -> Dict[str, Any]:
        """Create info for single-agent mode"""
        pass
    
    def _create_observation_with_context(self) -> Dict[str, Any]:
        """Create observation with multi-agent context"""
        base_obs = self._create_observation()
        
        # Add multi-agent context to observation
        if self.agent_history:
            base_obs["previous_agents"] = self.agent_history
            base_obs["collaboration_prompt"] = self._format_collaboration_prompt()
        
        if "debate" in self.multi_agent_context:
            base_obs["debate_context"] = self.multi_agent_context["debate"]
            base_obs["debate_prompt"] = self._format_debate_prompt()
        
        return base_obs
    
    def _create_info_with_context(self) -> Dict[str, Any]:
        """Create info with multi-agent context"""
        base_info = self._create_info()
        base_info["multi_agent_mode"] = True
        base_info["current_agent"] = self.current_agent_id
        base_info["agent_count"] = len(self.agent_history) + 1
        return base_info
    
    def _format_collaboration_prompt(self) -> str:
        """Format a prompt including previous agent responses"""
        if not self.agent_history:
            return ""
        
        prompt = "Previous agent responses:\n\n"
        for i, agent_data in enumerate(self.agent_history):
            prompt += f"Agent {agent_data['agent_id']}:\n{agent_data['response']}\n\n"
        
        prompt += "Please consider the above responses and provide your analysis or solution:"
        return prompt
    
    def _format_debate_prompt(self) -> str:
        """Format a prompt for debate scenarios"""
        debate_ctx = self.multi_agent_context.get("debate", {})
        round_num = debate_ctx.get("round", 0)
        
        prompt = f"Debate Round {round_num + 1}:\n\n"
        
        # Add previous rounds if any
        previous_rounds = debate_ctx.get("previous_rounds", [])
        for round_data in previous_rounds:
            prompt += f"Round {round_data['round'] + 1} responses:\n"
            for agent_id, output in round_data["outputs"].items():
                prompt += f"  {agent_id}: {output.get('response', '')}\n"
            prompt += "\n"
        
        prompt += "Please provide your response for this debate round, considering the previous arguments:"
        return prompt


class MathMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment for math problems"""
    
    def __init__(self, problem: str = "", solution: str = "", **kwargs):
        super().__init__(**kwargs)
        self.problem = problem
        self.solution = solution
        self.attempts = 0
        self.max_attempts = 3
    
    def _create_observation(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "type": "math_problem"
        }
    
    def _create_info(self) -> Dict[str, Any]:
        return {
            "solution": self.solution,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts
        }
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the math environment"""
        self.attempts += 1
        
        # Extract answer from action (assuming action is the agent's response)
        response = str(action)
        
        # Simple reward calculation (would be more sophisticated in practice)
        reward = self._calculate_math_reward(response)
        
        # Check if done
        done = (reward >= 1.0) or (self.attempts >= self.max_attempts)
        
        # Create next observation
        next_obs = {
            "problem": self.problem,
            "previous_response": response,
            "attempt": self.attempts
        }
        
        info = {
            "solution": self.solution,
            "attempts": self.attempts,
            "correct": reward >= 1.0
        }
        
        return next_obs, reward, done, info
    
    def _calculate_math_reward(self, response: str) -> float:
        """Calculate reward for math response"""
        # This is a simplified example - in practice you'd want more sophisticated evaluation
        if self.solution.lower().strip() in response.lower():
            return 1.0
        return 0.0
    
    @staticmethod
    def from_dict(info: Dict) -> "MathMultiAgentEnv":
        return MathMultiAgentEnv(
            problem=info.get("problem", ""),
            solution=info.get("solution", ""),
            task_data=info
        )


class CodeMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment for coding problems"""
    
    def __init__(self, problem_description: str = "", test_cases: List[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.problem_description = problem_description
        self.test_cases = test_cases or []
        self.submitted_code = ""
        self.execution_results = []
    
    def _create_observation(self) -> Dict[str, Any]:
        return {
            "problem": self.problem_description,
            "test_cases": self.test_cases,
            "type": "coding_problem"
        }
    
    def _create_info(self) -> Dict[str, Any]:
        return {
            "test_cases": self.test_cases,
            "execution_results": self.execution_results
        }
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the coding environment"""
        response = str(action)
        self.submitted_code = response
        
        # Execute code against test cases (simplified)
        reward = self._evaluate_code(response)
        
        # Always done after one step for coding problems
        done = True
        
        next_obs = {
            "problem": self.problem_description,
            "submitted_code": response,
            "execution_results": self.execution_results
        }
        
        info = {
            "test_cases": self.test_cases,
            "execution_results": self.execution_results,
            "passed_tests": sum(1 for r in self.execution_results if r.get("passed", False))
        }
        
        return next_obs, reward, done, info
    
    def _evaluate_code(self, code: str) -> float:
        """Evaluate submitted code (simplified)"""
        # This is a placeholder - in practice you'd want to safely execute code
        # and run test cases
        self.execution_results = []
        
        # Simplified scoring based on code content
        if "def " in code and "return" in code:
            return 0.8
        elif "def " in code:
            return 0.5
        else:
            return 0.2
    
    @staticmethod  
    def from_dict(info: Dict) -> "CodeMultiAgentEnv":
        return CodeMultiAgentEnv(
            problem_description=info.get("problem", ""),
            test_cases=info.get("test_cases", []),
            task_data=info
        )


class DebateEnv(MultiAgentEnv):
    """Environment specifically designed for multi-agent debates"""
    
    def __init__(self, topic: str = "", position_options: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.topic = topic
        self.position_options = position_options or ["For", "Against"]
        self.round_number = 0
        self.agent_positions = {}
    
    def _create_observation(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "position_options": self.position_options,
            "round": self.round_number,
            "type": "debate"
        }
    
    def _create_info(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "round": self.round_number,
            "agent_positions": self.agent_positions
        }
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the debate environment"""
        response = str(action)
        
        # Record agent position if this is a position-taking response
        if self.current_agent_id and any(pos in response for pos in self.position_options):
            for pos in self.position_options:
                if pos.lower() in response.lower():
                    self.agent_positions[self.current_agent_id] = pos
                    break
        
        self.round_number += 1
        
        # Simple reward for engagement (more sophisticated scoring would analyze argument quality)
        reward = 0.5 if len(response) > 50 else 0.2
        
        # Debate continues until explicitly ended
        done = False
        
        next_obs = {
            "topic": self.topic,
            "round": self.round_number,
            "latest_response": response
        }
        
        info = {
            "round": self.round_number,
            "agent_positions": self.agent_positions,
            "response_length": len(response)
        }
        
        return next_obs, reward, done, info
    
    @staticmethod
    def from_dict(info: Dict) -> "DebateEnv":
        return DebateEnv(
            topic=info.get("topic", ""),
            position_options=info.get("position_options", ["For", "Against"]),
            task_data=info
        ) 