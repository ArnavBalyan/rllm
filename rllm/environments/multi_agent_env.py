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
        """Reset with input from previous agents in the Chain of Experts"""
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
        """Process input from previous agents in the chain"""
        # Store previous agent outputs
        for source_agent, agent_output in agent_input.items():
            if isinstance(agent_output, dict) and "response" in agent_output:
                self.agent_history.append({
                    "agent_id": source_agent,
                    "response": agent_output["response"],
                    "timestamp": agent_output.get("timestamp", None)
                })
    
    @abstractmethod
    def _create_observation(self) -> Dict[str, Any]:
        """Create observation for single-agent mode"""
        pass
    
    @abstractmethod  
    def _create_info(self) -> Dict[str, Any]:
        """Create info for single-agent mode"""
        pass
    
    def _create_observation_with_context(self) -> Dict[str, Any]:
        """Create observation with Chain of Experts context"""
        base_obs = self._create_observation()
        
        # Add Chain of Experts context to observation
        if self.agent_history:
            base_obs["previous_agents"] = self.agent_history
            base_obs["collaboration_prompt"] = self._format_collaboration_prompt()
        
        return base_obs
    
    def _create_info_with_context(self) -> Dict[str, Any]:
        """Create info with Chain of Experts context"""
        base_info = self._create_info()
        base_info["multi_agent_mode"] = True
        base_info["current_agent"] = self.current_agent_id
        base_info["agent_count"] = len(self.agent_history) + 1
        return base_info
    
    def _format_collaboration_prompt(self) -> str:
        """Format a prompt including previous agent responses in the chain"""
        if not self.agent_history:
            return ""
        
        prompt = "Previous agent responses in the Chain of Experts:\n\n"
        for i, agent_data in enumerate(self.agent_history):
            prompt += f"Agent {agent_data['agent_id']}:\n{agent_data['response']}\n\n"
        
        prompt += "Please consider the above responses and provide your analysis or solution:"
        return prompt


class MathMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment for math problems in Chain of Experts"""
    
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
    """Multi-agent environment for coding problems in Chain of Experts"""
    
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