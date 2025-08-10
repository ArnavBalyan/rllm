import asyncio
import json
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.base_env import BaseEnv
from rllm.misc import colorful_print

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Define different roles agents can play in multi-agent workflows"""
    PROPOSER = "proposer"  # Proposes initial solutions
    CRITIC = "critic"      # Critiques and refines solutions
    JUDGE = "judge"        # Makes final decisions
    SPECIALIST = "specialist"  # Domain-specific expert
    AGGREGATOR = "aggregator"  # Combines multiple inputs


@dataclass
class AgentConfig:
    """Configuration for a single agent in multi-agent workflow"""
    agent_id: str
    agent_class: type
    agent_args: Dict[str, Any] = field(default_factory=dict)
    role: AgentRole = AgentRole.SPECIALIST
    model_path: Optional[str] = None
    max_response_length: int = 8192
    max_prompt_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class WorkflowConnection:
    """Represents a connection between agents in the workflow"""
    from_agent: str
    to_agent: str
    transform_fn: Optional[callable] = None  # Optional transformation function


@dataclass
class WorkflowPhase:
    """
    Represents a logical execution phase in the multi-agent workflow.
    
    A phase is a collection of agents that execute together in a specific mode:
    - For Chain of Experts: each phase contains exactly one agent
    - The phase defines HOW agents execute (sequential/parallel) and WHICH agents participate
    
    Note: This is different from a "step" which refers to one turn of conversation/action.
    """
    phase_id: str
    agent_ids: List[str]  # Agents that participate in this phase
    execution_mode: str = "sequential"  # "sequential" for Chain of Experts
    description: Optional[str] = None  # Human-readable description of what this phase does


class BaseWorkflow(ABC):
    """Abstract base class for defining multi-agent workflows"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.connections: List[WorkflowConnection] = []
        self.phases: List[WorkflowPhase] = []
    
    @abstractmethod
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowPhase], List[WorkflowConnection]]:
        """
        Define the workflow structure.
        
        Returns:
            Tuple of (agent_configs, workflow_phases, connections)
        """
        pass
    
    @abstractmethod
    def process_phase_output(self, phase_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process outputs from a workflow phase"""
        pass


class ChainOfExpertsWorkflow(BaseWorkflow):
    """
    Chain of Experts workflow: Agent A → Agent B → Agent C
    
    Each agent in the chain receives the output from the previous agent as context.
    This enables sequential refinement and specialization of solutions.
    """
    
    def __init__(self, agent_configs: List[AgentConfig]):
        super().__init__("chain_of_experts")
        self.agent_configs_list = agent_configs
    
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowPhase], List[WorkflowConnection]]:
        connections = []
        phases = []
        
        # Create sequential connections (Agent A → Agent B → Agent C)
        for i in range(len(self.agent_configs_list) - 1):
            connections.append(WorkflowConnection(
                from_agent=self.agent_configs_list[i].agent_id,
                to_agent=self.agent_configs_list[i + 1].agent_id
            ))
        
        # Create sequential phases (each phase contains exactly one agent)
        for i, config in enumerate(self.agent_configs_list):
            phases.append(WorkflowPhase(
                phase_id=f"phase_{i}",
                agent_ids=[config.agent_id],
                execution_mode="sequential",
                description=f"Execute {config.role.value} agent: {config.agent_id}"
            ))
        
        return self.agent_configs_list, phases, connections
    
    def process_phase_output(self, phase_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """For chain of experts, pass output directly to next agent"""
        return phase_outputs


class MultiAgentExecutionEngine:
    """
    Simplified execution engine for Chain of Experts training.
    
    Processes a single batch through sequential phases of the Chain of Experts:
    - Phase 0: Proposer processes batch → outputs
    - Phase 1: Expert processes batch with Proposer context → outputs  
    - Phase 2: Judge processes batch with Expert context → final outputs
    """
    
    def __init__(
        self,
        workflow: BaseWorkflow,
        env_class: type,
        env_args: Dict[str, Any] = None,
        engine_name: str = "verl",
        tokenizer=None,
        rollout_engine=None,
        trajectory_timeout: Optional[int] = None,
        max_workers: int = 64,
        **kwargs
    ):
        self.workflow = workflow
        self.env_class = env_class
        self.env_args = env_args or {}
        self.engine_name = engine_name
        self.tokenizer = tokenizer
        self.rollout_engine = rollout_engine
        self.trajectory_timeout = trajectory_timeout or int(1e9)
        self.max_workers = max_workers
        self.kwargs = kwargs
        
        # Initialize workflow structure
        self.agent_configs, self.workflow_phases, self.connections = workflow.define_workflow()
        
        # Create individual agent execution engines for each agent type
        # Each agent gets its own engine which can connect to separate vLLM instances
        self.agent_engines: Dict[str, AgentExecutionEngine] = {}
        self._initialize_agent_engines()
        
        # Environment instances for the batch
        self.envs: List[BaseEnv] = []
    
    def _initialize_agent_engines(self):
        """
        Initialize individual execution engines for each agent.
        
        Each agent gets its own AgentExecutionEngine which:
        - Can connect to a separate vLLM instance via router
        - Handles its own model path and configuration
        - Uses the same async infrastructure as single-agent rLLM
        """
        for config in self.agent_configs:
            # Create engine args specific to this agent
            agent_engine_args = self.kwargs.copy()
            
            # Add agent-specific model configuration if provided
            if config.model_path:
                agent_engine_args["model_path"] = config.model_path
            
            # Add sampling parameters
            agent_engine_args["sampling_params"] = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                **agent_engine_args.get("sampling_params", {})
            }
            
            self.agent_engines[config.agent_id] = AgentExecutionEngine(
                engine_name=self.engine_name,
                tokenizer=self.tokenizer,
                rollout_engine=self.rollout_engine,  # Shared rollout engine with router
                agent_class=config.agent_class,
                agent_args=config.agent_args,
                env_class=self.env_class,
                env_args=self.env_args,
                n_parallel_agents=1,  # Each agent engine handles one agent
                max_response_length=config.max_response_length,
                max_prompt_length=config.max_prompt_length,
                trajectory_timeout=self.trajectory_timeout,
                max_workers=self.max_workers,
                **agent_engine_args
            )
    
    def update_envs_and_agents(self, envs: List[BaseEnv]):
        """Update environment instances for the training batch"""
        self.envs = envs
        
        # Distribute environments to agent engines
        for agent_id, engine in self.agent_engines.items():
            # Each agent engine gets a copy of environments for the batch
            agent_envs = [env for env in envs]  # Could be shared or copied based on thread safety
            agent_agents = [engine.agent_class(**engine.agent_args) for _ in envs]
            engine.update_envs_and_agents(agent_envs, agent_agents)
    
    def execute_chain_of_experts_batch(
        self, 
        timing_raw: Dict[str, Any] = None, 
        meta_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Chain of Experts on a training batch.
        
        - Step 1: Proposer → Expert → Judge → Environment.step() → reward
        - Step 2: Proposer → Expert → Judge → Environment.step() → reward  
        - Step 3: Proposer → Expert → Judge → Environment.step() → reward
        - ... until episode done
        
        Each environment step goes through the complete chain sequentially.
        
        Returns:
            List of workflow results for each item in the batch
        """
        if timing_raw is None:
            timing_raw = {}
        if meta_info is None:
            meta_info = {}
        
        batch_size = len(self.envs)
        colorful_print(f"Executing Chain of Experts on batch of size {batch_size}", "cyan")
        
        # Execute trajectories step-by-step through the chain
        return self._execute_chain_trajectories_step_by_step(timing_raw, meta_info)
    
    def _execute_chain_trajectories_step_by_step(
        self, 
        timing_raw: Dict[str, Any], 
        meta_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute Chain of Experts trajectories step-by-step.        
        This ensures proper sequential chain execution per RL step.
        """
        batch_size = len(self.envs)
        workflow_results = []
        
        # Initialize trajectories for each batch item
        batch_trajectories = []
        for batch_idx in range(batch_size):
            # Create agent trajectories dynamically based on workflow
            agent_trajectories = {}
            for agent_config in self.agent_configs:
                agent_trajectories[agent_config.agent_id] = {
                    "steps": [], 
                    "total_reward": 0.0
                }
            
            trajectory = {
                "workflow_type": self.workflow.workflow_id,
                "batch_idx": batch_idx,
                "steps": [],  # List of chain steps
                "agent_trajectories": agent_trajectories,
                "episode_reward": 0.0,
                "episode_length": 0,
            }
            batch_trajectories.append(trajectory)
        
        # Reset all environments
        observations = []
        for batch_idx, env in enumerate(self.envs):
            obs, info = env.reset()
            observations.append(obs)
        
        # Reset all agents
        for engine in self.agent_engines.values():
            for agent in engine.agents:
                agent.reset()
        
        # Execute episodes step by step
        episodes_done = [False] * batch_size
        max_steps = self.agent_configs[0].agent_args.get("max_steps", 10)  # Get from agent config
        
        for step_num in range(max_steps):
            if all(episodes_done):
                break
                
            colorful_print(f"Executing chain step {step_num + 1}/{max_steps}", "yellow")
            
            # Process each active batch item through the chain
            for batch_idx in range(batch_size):
                if episodes_done[batch_idx]:
                    continue
                    
                env = self.envs[batch_idx]
                obs = observations[batch_idx]
                
                # Execute the chain for this step
                step_result = self._execute_chain_step(
                    batch_idx=batch_idx,
                    observation=obs,
                    step_num=step_num,
                    timing_raw=timing_raw
                )
                
                # Apply final action to environment
                final_action = step_result["final_action"]
                try:
                    action_value = self._parse_action_from_response(final_action)
                    next_obs, reward, done, info = env.step(action_value)
                    
                    # Update trajectory
                    step_result["reward"] = reward
                    step_result["done"] = done
                    step_result["next_observation"] = next_obs
                    
                    batch_trajectories[batch_idx]["steps"].append(step_result)
                    batch_trajectories[batch_idx]["episode_reward"] += reward
                    batch_trajectories[batch_idx]["episode_length"] += 1
                    
                    for agent_config in self.agent_configs:
                        agent_id = agent_config.agent_id
                        if agent_id in step_result["chain_responses"]:

                            final_agent_id = self.workflow_phases[-1].agent_ids[0]
                            reward_for_agent = reward if agent_id == final_agent_id else 0.0
                            
                            batch_trajectories[batch_idx]["agent_trajectories"][agent_id]["steps"].append({
                                "step": step_num,
                                "response": step_result["chain_responses"][agent_id],
                                "reward": reward_for_agent,
                            })
                            
                            if agent_id == final_agent_id:
                                batch_trajectories[batch_idx]["agent_trajectories"][agent_id]["total_reward"] += reward
                    
                    if done or env.finished():
                        episodes_done[batch_idx] = True
                        colorful_print(f"Episode {batch_idx} completed with reward {batch_trajectories[batch_idx]['episode_reward']}", "green")
                    else:
                        observations[batch_idx] = next_obs
                        
                except Exception as e:
                    logger.error(f"Error in chain step for batch {batch_idx}: {str(e)}")
                    episodes_done[batch_idx] = True
        
        for trajectory in batch_trajectories:
            workflow_results.append(self._format_trajectory_for_training(trajectory))
        
        return workflow_results
    
    def _execute_chain_step(
        self, 
        batch_idx: int, 
        observation: Any, 
        step_num: int,
        timing_raw: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step of the workflow following the DAG connections.
        
        Uses the workflow phases and connections to dynamically execute agents
        in the correct order based on the workflow definition.
        """
        chain_responses = {}
        
        # Execute each phase in the workflow sequentially
        for phase_idx, phase in enumerate(self.workflow_phases):
            phase_id = phase.phase_id
            
            with timing_raw.get(f"{phase_id}_phase", {}):
                # For Chain of Experts, each phase has exactly one agent
                agent_id = phase.agent_ids[0]
                engine = self.agent_engines[agent_id]
                agent = engine.agents[batch_idx] if batch_idx < len(engine.agents) else engine.agents[0]
                
                agent.update_from_env(observation, 0.0, False, {})
                
                agent_context = self._prepare_agent_context(agent_id, chain_responses)
                if hasattr(agent, 'collaboration_prompt'):
                    agent.collaboration_prompt = agent_context
                
                agent_response = self._get_agent_response(agent, engine)
                chain_responses[agent_id] = agent_response
        
        final_agent_id = self.workflow_phases[-1].agent_ids[0]
        final_action = chain_responses[final_agent_id]
        
        return {
            "step": step_num,
            "observation": observation,
            "chain_responses": chain_responses,
            "final_action": final_action,
        }
    
    def _prepare_agent_context(self, agent_id: str, previous_responses: Dict[str, str]) -> str:
        """
        Prepare context for an agent based on workflow connections.
        
        Finds all incoming connections to this agent and formats the context
        from previous agents' responses.
        """
        context_parts = []
        
        # Find incoming connections to this agent
        for connection in self.connections:
            if connection.to_agent == agent_id:
                source_agent = connection.from_agent
                
                if source_agent in previous_responses:
                    source_response = previous_responses[source_agent]
                    
                    # Apply transformation if provided
                    if connection.transform_fn:
                        source_response = connection.transform_fn(source_response)
                    
                    # Format the context nicely
                    agent_name = source_agent.replace("_", " ").title()
                    context_parts.append(f"{agent_name} Analysis:\n{source_response}")
        
        return "\n\n".join(context_parts)
    
    def _get_agent_response(self, agent: BaseAgent, engine: AgentExecutionEngine) -> str:
        """
        Get a single response from an agent using the engine's rollout system.
        
        This is a simplified version that gets one response per agent per step.
        """
        # For now, use a placeholder - in real implementation, this would:
        # 1. Convert agent.chat_completions to tokens
        # 2. Send to rollout_engine for generation  
        # 3. Parse response and update agent
        # 4. Return the agent's action response
        
        # Placeholder implementation
        import random
        actions = ["Up", "Down", "Left", "Right"]
        agent_name = agent.__class__.__name__.replace("FrozenLake", "").replace("Agent", "")
        
        # Simulate agent thinking and action selection
        action = random.choice(actions)
        response = f"[{agent_name} thinking] I need to analyze the board and choose the best move. Action: ```{action}```"
        
        # Update agent with this response
        try:
            agent.update_from_model(response)
        except:
            pass  # Handle any update errors gracefully
            
        return response
    
    def _parse_action_from_response(self, response: str) -> int:
        """
        Parse the final action from Judge's response.
        
        Expects action in the format: ```Up``` or ```Down``` etc.
        Returns integer action value for FrozenLake environment.
        """
        import re
        
        # Extract action from ```action``` format
        action_match = re.search(r'```(\w+)```', response)
        if action_match:
            action_str = action_match.group(1).strip().lower()
            
            # Map to FrozenLake action values
            action_mapping = {
                "left": 1,
                "down": 2, 
                "right": 3,
                "up": 4,
            }
            
            return action_mapping.get(action_str, 1)  # Default to left if parsing fails
        
        # Fallback if no action found
        logger.warning(f"Could not parse action from response: {response}")
        return 1  # Default action
    
    def _format_trajectory_for_training(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the chain trajectory for PPO training.
        
        The training expects specific format with tokens, rewards, etc.
        """
        final_agent_id = self.workflow_phases[-1].agent_ids[0]
        final_agent_trajectory = trajectory["agent_trajectories"][final_agent_id]
        
        formatted = {
            "workflow_type": trajectory["workflow_type"],
            "batch_idx": trajectory["batch_idx"],
            "agent_trajectories": {
                final_agent_id: {  
                    "prompt_tokens": torch.tensor([1, 2, 3]),  # Placeholder - should be real tokens
                    "response_tokens": torch.tensor([4, 5, 6]),  # Placeholder - should be real tokens
                    "response_masks": torch.tensor([1, 1, 1]),  # Placeholder
                    "trajectory_reward": final_agent_trajectory["total_reward"],
                    "chat_completions": [],  # Should contain actual chat history
                    "metrics": {
                        "episode_length": trajectory["episode_length"],
                        "episode_reward": trajectory["episode_reward"],
                    },
                }
            },
            "phase_outputs": {},  # Legacy format compatibility
        }
        
        return formatted
    
    def _prepare_batch_item_context(
        self, 
        batch_idx: int, 
        agent_id: str, 
        previous_phase_outputs: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Prepare context for a specific batch item and agent based on previous phases.
        
        Args:
            batch_idx: Index of the item in the batch
            agent_id: Current agent ID
            previous_phase_outputs: Outputs from previous phases
            
        Returns:
            Context dict for this batch item and agent
        """
        context = {}
        
        # Find incoming connections to this agent
        for connection in self.connections:
            if connection.to_agent == agent_id:
                source_agent = connection.from_agent
                
                # Find the output from source agent in previous phases
                for phase_id, phase_batch_outputs in previous_phase_outputs.items():
                    if batch_idx < len(phase_batch_outputs):
                        phase_output = phase_batch_outputs[batch_idx]
                        if source_agent in phase_output:
                            data = phase_output[source_agent]
                            
                            # Apply transformation if provided
                            if connection.transform_fn:
                                data = connection.transform_fn(data)
                            
                            context[source_agent] = data
        
        return context 