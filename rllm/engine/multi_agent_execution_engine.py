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
    """Represents a step in the multi-agent workflow"""
    step_id: str
    agent_ids: List[str]  # Agents that participate in this step
    execution_mode: str = "sequential"  # "sequential", "parallel", "debate"
    max_rounds: int = 1  # For debate mode
    aggregation_fn: Optional[callable] = None  # For parallel mode


class BaseWorkflow(ABC):
    """Abstract base class for defining multi-agent workflows"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.connections: List[WorkflowConnection] = []
        self.steps: List[WorkflowStep] = []
    
    @abstractmethod
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowStep], List[WorkflowConnection]]:
        """Define the workflow structure"""
        pass
    
    @abstractmethod
    def process_step_output(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process outputs from a workflow step"""
        pass


class ChainOfExpertsWorkflow(BaseWorkflow):
    """Chain of Experts workflow: Agent A -> Agent B -> Agent C"""
    
    def __init__(self, agent_configs: List[AgentConfig]):
        super().__init__("chain_of_experts")
        self.agent_configs_list = agent_configs
    
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowStep], List[WorkflowConnection]]:
        connections = []
        steps = []
        
        # Create sequential connections
        for i in range(len(self.agent_configs_list) - 1):
            connections.append(WorkflowConnection(
                from_agent=self.agent_configs_list[i].agent_id,
                to_agent=self.agent_configs_list[i + 1].agent_id
            ))
        
        # Create sequential steps
        for i, config in enumerate(self.agent_configs_list):
            steps.append(WorkflowStep(
                step_id=f"step_{i}",
                agent_ids=[config.agent_id],
                execution_mode="sequential"
            ))
        
        return self.agent_configs_list, steps, connections
    
    def process_step_output(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        # For chain of experts, pass output directly to next agent
        return step_outputs


class MixtureOfExpertsWorkflow(BaseWorkflow):
    """Mixture of Experts workflow: Multiple agents work in parallel then aggregate"""
    
    def __init__(self, expert_configs: List[AgentConfig], aggregator_config: AgentConfig):
        super().__init__("mixture_of_experts")
        self.expert_configs = expert_configs
        self.aggregator_config = aggregator_config
    
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowStep], List[WorkflowConnection]]:
        all_configs = self.expert_configs + [self.aggregator_config]
        
        # Parallel step for experts
        expert_step = WorkflowStep(
            step_id="expert_parallel",
            agent_ids=[config.agent_id for config in self.expert_configs],
            execution_mode="parallel",
            aggregation_fn=self._aggregate_expert_outputs
        )
        
        # Sequential step for aggregator
        aggregator_step = WorkflowStep(
            step_id="aggregator",
            agent_ids=[self.aggregator_config.agent_id],
            execution_mode="sequential"
        )
        
        # Connections from experts to aggregator
        connections = []
        for expert_config in self.expert_configs:
            connections.append(WorkflowConnection(
                from_agent=expert_config.agent_id,
                to_agent=self.aggregator_config.agent_id
            ))
        
        return all_configs, [expert_step, aggregator_step], connections
    
    def _aggregate_expert_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate outputs from multiple experts"""
        aggregated = {
            "expert_responses": [],
            "aggregated_prompt": "Here are the responses from multiple experts:\n\n"
        }
        
        for agent_id, output in outputs.items():
            aggregated["expert_responses"].append({
                "agent_id": agent_id,
                "response": output.get("response", "")
            })
            aggregated["aggregated_prompt"] += f"Expert {agent_id}: {output.get('response', '')}\n\n"
        
        aggregated["aggregated_prompt"] += "Please provide a final answer by considering all expert opinions:"
        return aggregated
    
    def process_step_output(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return step_outputs


class DebateWorkflow(BaseWorkflow):
    """Multi-agent debate workflow where agents discuss and refine solutions"""
    
    def __init__(self, debater_configs: List[AgentConfig], judge_config: AgentConfig, max_rounds: int = 3):
        super().__init__("debate")
        self.debater_configs = debater_configs
        self.judge_config = judge_config
        self.max_rounds = max_rounds
    
    def define_workflow(self) -> Tuple[List[AgentConfig], List[WorkflowStep], List[WorkflowConnection]]:
        all_configs = self.debater_configs + [self.judge_config]
        
        # Debate step
        debate_step = WorkflowStep(
            step_id="debate_round",
            agent_ids=[config.agent_id for config in self.debater_configs],
            execution_mode="debate",
            max_rounds=self.max_rounds
        )
        
        # Judge step
        judge_step = WorkflowStep(
            step_id="judge",
            agent_ids=[self.judge_config.agent_id],
            execution_mode="sequential"
        )
        
        connections = []
        for debater_config in self.debater_configs:
            connections.append(WorkflowConnection(
                from_agent=debater_config.agent_id,
                to_agent=self.judge_config.agent_id
            ))
        
        return all_configs, [debate_step, judge_step], connections
    
    def process_step_output(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return step_outputs


class MultiAgentExecutionEngine:
    """Enhanced execution engine for multi-agent workflows"""
    
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
        self.agent_engines: Dict[str, AgentExecutionEngine] = {}
        self._initialize_agent_engines()
        
        # Environment instances for each workflow
        self.envs: List[BaseEnv] = []
        
        # Workflow state tracking
        self.workflow_states: Dict[str, Dict] = {}
    
    def _initialize_agent_engines(self):
        """Initialize individual execution engines for each agent"""
        for config in self.agent_configs:
            self.agent_engines[config.agent_id] = AgentExecutionEngine(
                engine_name=self.engine_name,
                tokenizer=self.tokenizer,
                rollout_engine=self.rollout_engine,
                agent_class=config.agent_class,
                agent_args=config.agent_args,
                env_class=self.env_class,
                env_args=self.env_args,
                n_parallel_agents=1,  # Each agent engine handles one agent
                max_response_length=config.max_response_length,
                max_prompt_length=config.max_prompt_length,
                trajectory_timeout=self.trajectory_timeout,
                max_workers=self.max_workers,
                **self.kwargs
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
        """Execute a complete multi-agent workflow trajectory"""
        
        workflow_id = f"{application_id}_workflow_{workflow_idx}"
        self.workflow_states[workflow_id] = {
            "step_outputs": {},
            "agent_trajectories": {},
            "current_step": 0,
            "start_time": time.time()
        }
        
        try:
            # Execute each step in the workflow
            for step_idx, step in enumerate(self.workflow_steps):
                colorful_print(f"Executing workflow step {step_idx}: {step.step_id}", "cyan")
                
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
        """Execute a single workflow step"""
        
        if step.execution_mode == "sequential":
            return await self._execute_sequential_step(
                workflow_id, step, workflow_idx, application_id, seed, mode, **kwargs
            )
        elif step.execution_mode == "parallel":
            return await self._execute_parallel_step(
                workflow_id, step, workflow_idx, application_id, seed, mode, **kwargs
            )
        elif step.execution_mode == "debate":
            return await self._execute_debate_step(
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
        """Execute agents sequentially"""
        step_outputs = {}
        
        for agent_id in step.agent_ids:
            # Prepare input for this agent based on previous steps
            agent_input = self._prepare_agent_input(workflow_id, agent_id)
            
            # Execute agent
            agent_output = await self._execute_single_agent(
                agent_id, workflow_idx, application_id, seed, mode, agent_input, **kwargs
            )
            
            step_outputs[agent_id] = agent_output
            
            # Store agent trajectory
            self.workflow_states[workflow_id]["agent_trajectories"][agent_id] = agent_output
        
        return step_outputs
    
    async def _execute_parallel_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        workflow_idx: int,
        application_id: str,
        seed: int,
        mode: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute agents in parallel"""
        
        # Prepare inputs for all agents
        agent_inputs = {}
        for agent_id in step.agent_ids:
            agent_inputs[agent_id] = self._prepare_agent_input(workflow_id, agent_id)
        
        # Execute all agents in parallel
        tasks = []
        for agent_id in step.agent_ids:
            task = self._execute_single_agent(
                agent_id, workflow_idx, f"{application_id}_{agent_id}", 
                seed, mode, agent_inputs[agent_id], **kwargs
            )
            tasks.append((agent_id, task))
        
        # Wait for all to complete
        step_outputs = {}
        for agent_id, task in tasks:
            agent_output = await task
            step_outputs[agent_id] = agent_output
            self.workflow_states[workflow_id]["agent_trajectories"][agent_id] = agent_output
        
        # Apply aggregation function if provided
        if step.aggregation_fn:
            step_outputs = step.aggregation_fn(step_outputs)
        
        return step_outputs
    
    async def _execute_debate_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        workflow_idx: int,
        application_id: str,
        seed: int,
        mode: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute multi-round debate between agents"""
        
        debate_history = []
        agent_contexts = {agent_id: {} for agent_id in step.agent_ids}
        
        for round_idx in range(step.max_rounds):
            colorful_print(f"Debate round {round_idx + 1}/{step.max_rounds}", "yellow")
            
            round_outputs = {}
            
            for agent_id in step.agent_ids:
                # Prepare input including debate history
                agent_input = self._prepare_debate_input(
                    workflow_id, agent_id, debate_history, round_idx
                )
                
                # Execute agent
                agent_output = await self._execute_single_agent(
                    agent_id, workflow_idx, f"{application_id}_{agent_id}_r{round_idx}",
                    seed, mode, agent_input, **kwargs
                )
                
                round_outputs[agent_id] = agent_output
                agent_contexts[agent_id][f"round_{round_idx}"] = agent_output
            
            # Add round to debate history
            debate_history.append({
                "round": round_idx,
                "outputs": round_outputs
            })
        
        # Return final debate state
        return {
            "debate_history": debate_history,
            "final_positions": {agent_id: contexts[f"round_{step.max_rounds-1}"] 
                             for agent_id, contexts in agent_contexts.items()},
            "agent_contexts": agent_contexts
        }
    
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
        """Execute a single agent with given input"""
        
        engine = self.agent_engines[agent_id]
        
        # Update agent's environment with the input
        env = engine.envs[workflow_idx]
        agent = engine.agents[workflow_idx]
        
        # Reset environment and agent with the input
        if agent_input:
            # Custom reset with input
            observation, info = await asyncio.get_event_loop().run_in_executor(
                engine.executor, lambda: env.reset() if not hasattr(env, 'reset_with_input') 
                else env.reset_with_input(agent_input)
            )
        else:
            observation, info = await asyncio.get_event_loop().run_in_executor(
                engine.executor, env.reset
            )
        
        agent.reset()
        
        # Execute the agent trajectory
        trajectory_result = await engine.run_agent_trajectory_async(
            workflow_idx, application_id, seed, mode, **kwargs
        )
        
        return trajectory_result
    
    def _prepare_agent_input(self, workflow_id: str, agent_id: str) -> Dict[str, Any]:
        """Prepare input for an agent based on previous workflow steps"""
        
        # Find incoming connections to this agent
        incoming_data = {}
        
        for connection in self.connections:
            if connection.to_agent == agent_id:
                source_agent = connection.from_agent
                
                # Find the output from source agent
                for step_id, step_output in self.workflow_states[workflow_id]["step_outputs"].items():
                    if source_agent in step_output:
                        data = step_output[source_agent]
                        
                        # Apply transformation if provided
                        if connection.transform_fn:
                            data = connection.transform_fn(data)
                        
                        incoming_data[source_agent] = data
        
        return incoming_data
    
    def _prepare_debate_input(
        self, 
        workflow_id: str, 
        agent_id: str, 
        debate_history: List[Dict], 
        round_idx: int
    ) -> Dict[str, Any]:
        """Prepare input for debate round"""
        
        base_input = self._prepare_agent_input(workflow_id, agent_id)
        
        # Add debate context
        debate_context = {
            "round": round_idx,
            "previous_rounds": debate_history,
            "other_agents": [aid for aid in self.workflow.agent_configs if aid != agent_id]
        }
        
        base_input["debate_context"] = debate_context
        return base_input
    
    def _process_workflow_completion(self, workflow_id: str) -> Dict[str, Any]:
        """Process the completion of a workflow"""
        
        state = self.workflow_states[workflow_id]
        total_time = time.time() - state["start_time"]
        
        # Get final outputs
        final_step_outputs = list(state["step_outputs"].values())[-1] if state["step_outputs"] else {}
        
        # Compute workflow-level metrics
        workflow_result = {
            "workflow_id": workflow_id,
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


# Integration with existing trainer
class MultiAgentPPOTrainer:
    """Integration class for multi-agent workflows with existing PPO training"""
    
    def __init__(
        self,
        base_trainer,  # AgentPPOTrainer instance
        workflow: BaseWorkflow,
        **kwargs
    ):
        self.base_trainer = base_trainer
        self.workflow = workflow
        
        # Replace the single-agent execution engine with multi-agent version
        self.multi_agent_engine = MultiAgentExecutionEngine(
            workflow=workflow,
            env_class=base_trainer.env_class,
            env_args=base_trainer.env_args,
            engine_name="verl",  # Use verl for training
            tokenizer=base_trainer.tokenizer,
            rollout_engine=base_trainer.rollout_wg if hasattr(base_trainer, 'rollout_wg') else None,
            n_parallel_workflows=base_trainer.config.actor_rollout_ref.rollout.n,
            **kwargs
        )
    
    def init_envs_and_agents(self, batch):
        """Initialize environments and agents for multi-agent training"""
        
        # Use base trainer's environment initialization
        envs = self.base_trainer.init_envs_and_agents(batch)
        
        # Update multi-agent engine with environments
        self.multi_agent_engine.update_envs_and_agents(envs)
        
        return envs
    
    async def generate_multi_agent_trajectories(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate multi-agent trajectories for training"""
        
        return await self.multi_agent_engine.execute_multi_agent_workflows(tasks)
    
    def transform_multi_agent_trajectories(self, trajectories: List[Dict[str, Any]]):
        """Transform multi-agent trajectories into format expected by verl"""
        
        # This would need to aggregate multiple agent trajectories into single training examples
        # The exact implementation depends on how you want to train the multi-agent system
        
        # Option 1: Train each agent separately
        # Option 2: Train a unified model with multi-agent context
        # Option 3: Use different loss functions for different agents
        
        # For now, we'll use a simple approach where we concatenate agent responses
        transformed_trajectories = []
        
        for workflow_result in trajectories:
            agent_trajectories = workflow_result.get("agent_trajectories", {})
            
            # Create a unified trajectory by combining agent outputs
            if agent_trajectories:
                # Take the final agent's trajectory as primary
                final_agent_id = list(agent_trajectories.keys())[-1]
                primary_trajectory = agent_trajectories[final_agent_id]
                
                # Add multi-agent context to the trajectory
                primary_trajectory["multi_agent_context"] = {
                    "workflow_type": self.workflow.workflow_id,
                    "agent_count": len(agent_trajectories),
                    "collaboration_history": workflow_result.get("step_outputs", {})
                }
                
                transformed_trajectories.append(primary_trajectory)
        
        return transformed_trajectories 