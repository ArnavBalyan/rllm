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
class WorkflowStep:
    """
    Represents a logical execution phase in the multi-agent workflow.
    
    A step is a collection of agents that execute together in a specific mode:
    - For Chain of Experts: each step contains exactly one agent
    - The step defines HOW agents execute (sequential/parallel) and WHICH agents participate
    """
    step_id: str
    agent_ids: List[str]  # Agents that participate in this step
    execution_mode: str = "sequential"  # "sequential" for Chain of Experts
    description: Optional[str] = None  # Human-readable description of what this step does


class BaseWorkflow(ABC):
    """Abstract base class for defining multi-agent workflows"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.connections: List[WorkflowConnection] = []
        self.steps: List[WorkflowStep] = []
    
    @abstractmethod
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowStep], List[WorkflowConnection]]:
        """
        Define the workflow structure.
        
        Returns:
            Tuple of (agent_configs, workflow_steps, connections)
        """
        pass
    
    @abstractmethod
    def process_step_output(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process outputs from a workflow step"""
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
    
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowStep], List[WorkflowConnection]]:
        connections = []
        steps = []
        
        # Create sequential connections (Agent A → Agent B → Agent C)
        for i in range(len(self.agent_configs_list) - 1):
            connections.append(WorkflowConnection(
                from_agent=self.agent_configs_list[i].agent_id,
                to_agent=self.agent_configs_list[i + 1].agent_id
            ))
        
        # Create sequential steps (each step contains exactly one agent)
        for i, config in enumerate(self.agent_configs_list):
            steps.append(WorkflowStep(
                step_id=f"step_{i}",
                agent_ids=[config.agent_id],
                execution_mode="sequential",
                description=f"Execute {config.role.value} agent: {config.agent_id}"
            ))
        
        return self.agent_configs_list, steps, connections
    
    def process_step_output(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """For chain of experts, pass output directly to next agent"""
        return step_outputs


class MultiAgentExecutionEngine:
    """
    Enhanced execution engine for multi-agent workflows.
    
    Architecture:
    - Creates individual AgentExecutionEngine instances for each agent
    - Each agent can have its own vLLM instance/shard via the router
    - Orchestrates sequential/parallel execution according to workflow definition
    - Integrates with existing rLLM async infrastructure
    """
    
    def __init__(
        self,
        workflow: BaseWorkflow,
        env_class: type,
        env_args: Dict[str, Any] = None,
        engine_name: str = "openai",
        tokenizer=None,
        rollout_engine=None,
        n_parallel_workflows: int = 1,
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
        self.n_parallel_workflows = n_parallel_workflows
        self.trajectory_timeout = trajectory_timeout or int(1e9)
        self.max_workers = max_workers
        self.kwargs = kwargs
        
        # Initialize workflow structure
        self.agent_configs, self.workflow_steps, self.connections = workflow.define_workflow()
        
        # Create individual agent execution engines for each agent type
        # Each agent gets its own engine which can connect to separate vLLM instances
        self.agent_engines: Dict[str, AgentExecutionEngine] = {}
        self._initialize_agent_engines()
        
        # Environment instances for each workflow
        self.envs: List[BaseEnv] = []
        
        # Workflow state tracking
        self.workflow_states: Dict[str, Dict] = {}
    
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
        """Update environment instances for workflows"""
        self.envs = envs
        
        # Distribute environments to agent engines
        for agent_id, engine in self.agent_engines.items():
            # Each agent engine gets a copy of environments
            agent_envs = [env for env in envs]  # Could be shared or copied based on thread safety
            agent_agents = [engine.agent_class(**engine.agent_args) for _ in envs]
            engine.update_envs_and_agents(agent_envs, agent_agents)
    
    async def execute_multi_agent_trajectory(
        self, 
        workflow_idx: int, 
        application_id: str, 
        seed: int = 0, 
        mode: str = "Text",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a complete multi-agent workflow trajectory.
        
        For Chain of Experts:
        1. Execute Step 0: Proposer agent
        2. Execute Step 1: Expert agent (receives Proposer output as context)
        3. Execute Step 2: Critic agent (receives Expert output as context)
        4. Execute Step 3: Judge agent (receives Critic output as context)
        """
        
        workflow_id = f"{application_id}_workflow_{workflow_idx}"
        self.workflow_states[workflow_id] = {
            "step_outputs": {},
            "agent_trajectories": {},
            "current_step": 0,
            "start_time": time.time()
        }
        
        try:
            # Execute each step in the workflow sequentially
            for step_idx, step in enumerate(self.workflow_steps):
                colorful_print(f"Executing workflow step {step_idx}: {step.step_id} - {step.description}", "cyan")
                
                step_output = await self._execute_workflow_step(
                    workflow_id, step, workflow_idx, application_id, seed, mode, **kwargs
                )
                
                self.workflow_states[workflow_id]["step_outputs"][step.step_id] = step_output
                self.workflow_states[workflow_id]["current_step"] = step_idx + 1
            
            # Process final workflow output
            final_output = self._process_workflow_completion(workflow_id)
            
            return final_output
            
        except Exception as e:
            logger.error(f"Error in workflow {workflow_id}: {str(e)}")
            traceback.print_exc()
            raise e
        finally:
            # Cleanup workflow state
            if workflow_id in self.workflow_states:
                del self.workflow_states[workflow_id]
    
    async def _execute_workflow_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        workflow_idx: int,
        application_id: str,
        seed: int,
        mode: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        For Chain of Experts, this executes exactly one agent sequentially.
        """
        
        if step.execution_mode == "sequential":
            return await self._execute_sequential_step(
                workflow_id, step, workflow_idx, application_id, seed, mode, **kwargs
            )
        else:
            raise ValueError(f"Unknown execution mode: {step.execution_mode}")
    
    async def _execute_sequential_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        workflow_idx: int,
        application_id: str,
        seed: int,
        mode: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agents sequentially within a step.
        
        For Chain of Experts, each step has exactly one agent.
        """
        step_outputs = {}
        
        for agent_id in step.agent_ids:
            # Prepare input for this agent based on previous steps
            agent_input = self._prepare_agent_input(workflow_id, agent_id)
            
            # Execute agent using its dedicated AgentExecutionEngine
            agent_output = await self._execute_single_agent(
                agent_id, workflow_idx, application_id, seed, mode, agent_input, **kwargs
            )
            
            step_outputs[agent_id] = agent_output
            
            # Store agent trajectory
            self.workflow_states[workflow_id]["agent_trajectories"][agent_id] = agent_output
        
        return step_outputs
    
    async def _execute_single_agent(
        self,
        agent_id: str,
        workflow_idx: int,
        application_id: str,
        seed: int,
        mode: str,
        agent_input: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a single agent with given input.
        
        This uses the agent's dedicated AgentExecutionEngine which:
        - Connects to the appropriate vLLM instance via router
        - Handles async execution
        - Manages the agent's environment interaction
        """
        
        engine = self.agent_engines[agent_id]
        
        # Update agent's environment with the input from previous agents
        env = engine.envs[workflow_idx]
        agent = engine.agents[workflow_idx]
        
        # Reset environment and agent with the input from previous agents
        if agent_input:
            # Custom reset with input from previous agents in chain
            observation, info = await asyncio.get_event_loop().run_in_executor(
                engine.executor, lambda: env.reset() if not hasattr(env, 'reset_with_input') 
                else env.reset_with_input(agent_input)
            )
        else:
            observation, info = await asyncio.get_event_loop().run_in_executor(
                engine.executor, env.reset
            )
        
        agent.reset()
        
        # Execute the agent trajectory using existing async infrastructure
        trajectory_result = await engine.run_agent_trajectory_async(
            workflow_idx, application_id, seed, mode, **kwargs
        )
        
        return trajectory_result
    
    def _prepare_agent_input(self, workflow_id: str, agent_id: str) -> Dict[str, Any]:
        """
        Prepare input for an agent based on previous workflow steps.
        
        For Chain of Experts:
        - First agent gets no input (starts fresh)
        - Subsequent agents get the output from the previous agent as context
        """
        
        # Find incoming connections to this agent
        incoming_data = {}
        
        for connection in self.connections:
            if connection.to_agent == agent_id:
                source_agent = connection.from_agent
                
                # Find the output from source agent in previous steps
                for step_id, step_output in self.workflow_states[workflow_id]["step_outputs"].items():
                    if source_agent in step_output:
                        data = step_output[source_agent]
                        
                        # Apply transformation if provided
                        if connection.transform_fn:
                            data = connection.transform_fn(data)
                        
                        incoming_data[source_agent] = data
        
        return incoming_data
    
    def _process_workflow_completion(self, workflow_id: str) -> Dict[str, Any]:
        """Process the completion of a workflow"""
        
        state = self.workflow_states[workflow_id]
        total_time = time.time() - state["start_time"]
        
        # Get final outputs (from the last step)
        final_step_outputs = list(state["step_outputs"].values())[-1] if state["step_outputs"] else {}
        
        # Compute workflow-level metrics
        workflow_result = {
            "workflow_id": workflow_id,
            "workflow_type": self.workflow.workflow_id,
            "total_time": total_time,
            "steps_completed": state["current_step"],
            "final_outputs": final_step_outputs,
            "agent_trajectories": state["agent_trajectories"],
            "step_outputs": state["step_outputs"]
        }
        
        # Apply workflow-specific processing
        processed_result = self.workflow.process_step_output(workflow_result)
        
        return processed_result
    
    async def execute_multi_agent_workflows(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple multi-agent workflows in parallel"""
        
        max_concurrent = self.n_parallel_workflows
        all_results = {}
        
        # Create task queue
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Track completed workflows
        completed = 0
        total = len(tasks)
        
        async def workflow_wrapper(task_id: int, task: Dict[str, Any]):
            nonlocal completed
            async with semaphore:
                try:
                    # Initialize environment for this workflow
                    env = self.env_class.from_dict({**self.env_args, **task})
                    
                    # Execute workflow
                    application_id = str(uuid.uuid4())
                    result = await self.execute_multi_agent_trajectory(
                        workflow_idx=task_id,
                        application_id=application_id,
                        seed=task.get("seed", 0)
                    )
                    
                    result["task_id"] = task_id
                    result["task"] = task
                    
                    completed += 1
                    colorful_print(f"Multi-agent workflows {completed}/{total} completed", "cyan")
                    
                    return task_id, result
                    
                except Exception as e:
                    logger.error(f"Error in workflow {task_id}: {str(e)}")
                    traceback.print_exc()
                    raise e
        
        # Create all workflow tasks
        workflow_tasks = [workflow_wrapper(task_id, task) for task_id, task in task_queue]
        
        # Execute all workflows
        for coro in asyncio.as_completed(workflow_tasks):
            try:
                task_id, result = await coro
                all_results[task_id] = result
            except Exception as e:
                logger.error(f"Workflow execution failed: {str(e)}")
                raise e
        
        # Return results in original order
        return [all_results[i] for i in range(len(tasks))] 