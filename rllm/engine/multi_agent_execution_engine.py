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

from rllm.agents.agent import Action, BaseAgent, Trajectory, Step
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.base_env import BaseEnv
from rllm.misc import colorful_print
from verl.trainer.ppo.ray_trainer import _timer

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
    Multi-agent execution engine that directly orchestrates workflows without wrapper agents.
    
    This engine manages multiple individual AgentExecutionEngines and coordinates their
    execution according to the defined workflow phases and connections.
    """
    
    def __init__(self, workflow: BaseWorkflow, env_class, *, env_args=None,
                 engine_name="verl", tokenizer=None, rollout_engine=None,
                 config=None, trajectory_timeout=None, max_workers=64, max_steps=10, **kwargs):
        self.workflow = workflow
        self.agent_cfgs, self.phases, self.connections = workflow.define_workflow()
        self.max_steps = max_steps
        self.trajectory_timeout = trajectory_timeout
        self.env_class = env_class
        self.env_args = env_args or {}
        self.envs = []
        
        # Create individual execution engines for each agent
        self.role_engines = self._init_role_engines(
            env_class, env_args, tokenizer, rollout_engine,
            config, trajectory_timeout, max_workers, engine_name, **kwargs
        )
        
        # Create individual agents for each role
        self.agents: Dict[str, BaseAgent] = {}
        for cfg in self.agent_cfgs:
            engine = self.role_engines[cfg.agent_id]
            agent_init_args = engine.agent_args.copy()
            self.agents[cfg.agent_id] = engine.agent_class(agent_id=cfg.agent_id, **agent_init_args)

    def _init_role_engines(self, env_class, env_args, tokenizer, rollout_engine, config, trajectory_timeout, max_workers, engine_name, **kwargs):
        role_engines = {}
        global_config = config  
        for agent_cfg in self.agent_cfgs:
            agent_engine_args = kwargs.copy()
            
            agent_engine_args.pop('max_response_length', None)
            agent_engine_args.pop('max_prompt_length', None)
            
            if agent_cfg.model_path:
                agent_engine_args["model_path"] = agent_cfg.model_path
            
            agent_engine_args["sampling_params"] = {
                "temperature": agent_cfg.temperature,
                "top_p": agent_cfg.top_p,
                **agent_engine_args.get("sampling_params", {})
            }
            
            role_engines[agent_cfg.agent_id] = AgentExecutionEngine(
                engine_name=engine_name,
                tokenizer=tokenizer,
                rollout_engine=rollout_engine,
                config=global_config,
                agent_class=agent_cfg.agent_class,
                agent_args=agent_cfg.agent_args,
                env_class=env_class,
                env_args=env_args,
                n_parallel_agents=1,
                max_response_length=agent_cfg.max_response_length,
                max_prompt_length=agent_cfg.max_prompt_length,
                trajectory_timeout=trajectory_timeout,
                max_workers=max_workers,
                **agent_engine_args
            )
        return role_engines
    
    def update_envs_and_agents(self, envs: List[BaseEnv]):
        """Update environments for the multi-agent workflow"""
        self.envs = envs
        colorful_print(f"🌍 Updated {len(envs)} environments for multi-agent workflow", "green")
    
    async def run_workflow_trajectory_async(self, env_idx: int, application_id: str, seed: int = 0, mode: str = "Token", **kwargs) -> Dict[str, Any]:
        """
        Execute a complete workflow trajectory for a single environment.
        
        This replaces the single-agent trajectory execution with multi-agent workflow orchestration.
        """
        env = self.envs[env_idx]
        trajectory = Trajectory()
        
        colorful_print(f"\n{'='*100}", "cyan")
        colorful_print(f"🚀 Starting Workflow Trajectory {env_idx} - {self.workflow.workflow_id}", "cyan")
        colorful_print(f"{'='*100}", "cyan")
        
        # Reset environment
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(None, env.reset)
        info["max_steps"] = self.max_steps
        
        # Reset all agents
        for agent_id, agent in self.agents.items():
            agent.reset()
            colorful_print(f"🔄 Reset {agent_id}", "yellow")
        
        total_reward = 0.0
        final_tokens = []
        final_masks = []
        chat_completions = []
        
        # Execute workflow for each environment step
        for step_idx in range(self.max_steps):
            colorful_print(f"\n🔢 Environment Step {step_idx + 1}/{self.max_steps}", "blue")
            
            # Update all agents with current environment state
            for agent_id, agent in self.agents.items():
                agent.update_from_env(observation, 0.0, False, info)
            
            # Execute workflow phases sequentially
            phase_responses = {}
            final_action = None
            
            for phase_idx, phase in enumerate(self.phases):
                agent_id = phase.agent_ids[0]  # Chain of experts: one agent per phase
                agent = self.agents[agent_id]
                engine = self.role_engines[agent_id]
                
                colorful_print(f"🎯 Phase {phase_idx + 1}: {agent_id.upper()}", "yellow")
                
                # Inject context from previous phases
                self._inject_workflow_context(agent_id, phase_responses)
                
                # Get LLM response
                prompt_msgs = agent.chat_completions
                response = await engine.get_model_response(
                    prompt_msgs, 
                    application_id, 
                    max_tokens=engine.max_response_length,
                    **engine.sampling_params
                )
                
                # Update agent with response and get action
                action = agent.update_from_model(response)
                phase_responses[agent_id] = response
                final_action = action
                
                colorful_print(f"✅ {agent_id} → Action: {action.action}", "green")
            
            # Execute final action in environment
            if final_action:
                observation, reward, done, info = await loop.run_in_executor(
                    None, env.step, final_action.action
                )
                total_reward += reward
                
                # Store step in trajectory
                step = Step(
                    observation=observation,
                    model_response=f"Workflow: {' → '.join(phase_responses.keys())}",
                    action=final_action.action,
                    reward=reward,
                    done=done,
                    info=info.copy()
                )
                trajectory.steps.append(step)
                
                # Collect final agent's tokens for training
                final_agent_id = self.phases[-1].agent_ids[0]
                final_agent = self.agents[final_agent_id]
                if hasattr(final_agent, 'chat_completions'):
                    chat_completions = final_agent.chat_completions
                
                if done:
                    colorful_print(f"🏁 Environment episode complete at step {step_idx + 1}", "green")
                    break
        
        colorful_print(f"🎉 Workflow trajectory complete! Total reward: {total_reward}", "green")
        
        # Return result in expected format for training
        if mode == "Token":
            # Convert final agent's messages to tokens for training
            final_agent_id = self.phases[-1].agent_ids[0]
            final_agent = self.agents[final_agent_id]
            
            from rllm.agents.utils import convert_messages_to_tokens_and_masks
            
            if hasattr(final_agent, 'chat_completions') and final_agent.chat_completions:
                try:
                    engine = self.role_engines[final_agent_id]
                    prompt_tokens, response_tokens = convert_messages_to_tokens_and_masks(
                        final_agent.chat_completions,
                        tokenizer=engine.tokenizer,
                        parser=engine.chat_parser,
                        contains_first_msg=True,
                        contains_generation_msg=True
                    )
                    response_masks = torch.ones_like(response_tokens)
                except Exception as e:
                    colorful_print(f"⚠️ Token conversion failed: {e}", "yellow")
                    prompt_tokens = torch.empty(0, dtype=torch.long)
                    response_tokens = torch.empty(0, dtype=torch.long)
                    response_masks = torch.empty(0, dtype=torch.long)
            else:
                prompt_tokens = torch.empty(0, dtype=torch.long)
                response_tokens = torch.empty(0, dtype=torch.long)
                response_masks = torch.empty(0, dtype=torch.long)
            
            return {
                "idx": env_idx,
                "trajectory_reward": total_reward,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_masks": response_masks,
                "chat_completions": chat_completions,
                "metrics": {
                    "workflow_steps": len(trajectory.steps),
                    "phases_executed": len(self.phases),
                    "total_reward": total_reward
                }
            }
        else:
            return {
                "idx": env_idx,
                "trajectory": trajectory,
                "total_reward": total_reward,
                "chat_completions": chat_completions
            }
    
    def _inject_workflow_context(self, current_agent_id: str, phase_responses: Dict[str, str]):
        """Inject context from previous phases into current agent"""
        context_parts = []
        
        # Find connections to current agent and add context from previous responses
        for conn in self.connections:
            if conn.to_agent == current_agent_id and conn.from_agent in phase_responses:
                context_parts.append(f"Input from {conn.from_agent.upper()}: {phase_responses[conn.from_agent]}")
        
        if context_parts:
            chain_context = "\n\n".join(context_parts)
            current_agent = self.agents[current_agent_id]
            if hasattr(current_agent, 'multi_agent_context'):
                current_agent.multi_agent_context["chain_context"] = chain_context
                colorful_print(f"📥 Injected context into {current_agent_id}: {len(chain_context)} chars", "cyan")
    
    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Token", **kwargs):
        """Generate trajectories for all environments using workflow execution"""
        if timing_raw is None:
            timing_raw = {}
        
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"
        
        max_concurrency = len(self.envs)
        
        async def launch_workflow_trajectory(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_workflow_trajectory_async(
                    env_idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs
                )
                return result
            except Exception as e:
                colorful_print(f"❌ Workflow trajectory {env_idx} failed: {e}", "red")
                traceback.print_exc()
                raise e
        
        # Execute all workflow trajectories
        tasks = [launch_workflow_trajectory(i) for i in range(len(self.envs))]
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                yield result
            except Exception as e:
                colorful_print(f"❌ Workflow execution failed: {e}", "red")
                raise e
    
    def execute_chain_of_experts_batch(
        self, 
        timing_raw: Dict[str, Any] = None, 
        meta_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute Chain of Experts workflow on a training batch"""
        batch_size = len(self.envs)
        colorful_print(f"🚀 Starting Chain of Experts batch execution with {batch_size} environments", "cyan")
        
        async def _collect_batch():
            batch = []
            async for traj in self.trajectory_generator(timing_raw=timing_raw, mode="Token", **meta_info or {}):
                batch.append(traj)
            return batch
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with ThreadPoolExecutor() as ex:
                    fut = ex.submit(asyncio.run, _collect_batch())
                    batch_timeout = meta_info.get('batch_execution_timeout', self.trajectory_timeout) if meta_info else self.trajectory_timeout
                    results = fut.result(timeout=batch_timeout)
            else:
                results = loop.run_until_complete(_collect_batch())
        except Exception as e:
            raise RuntimeError(f"Chain of Experts batch execution failed: {str(e)}") from e
        
        colorful_print(f"✅ Chain of Experts batch complete with {len(results)} results", "green")
        return self._format_results_for_training(results)
    
    def _format_results_for_training(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format workflow results for PPO training"""
        if not results:
            raise RuntimeError("Chain of Experts execution produced no results")
        
        formatted_results = []
        for i, token_result in enumerate(results):
            if not token_result:
                raise RuntimeError(f"Chain of Experts result {i} is empty")
            
            # Validate required fields
            required_fields = ["trajectory_reward", "chat_completions"]
            for field in required_fields:
                if field not in token_result:
                    colorful_print(f"❌ Missing required field '{field}' in result {i}", "red")
                    raise ValueError(f"Missing required field '{field}' in trajectory result {i}")
            
            formatted_results.append({
                "workflow_type": self.workflow.workflow_id,
                "batch_idx": token_result.get("idx", 0),
                "agent_trajectories": {
                    # Attribute to final agent for reward accounting
                    self.phases[-1].agent_ids[0]: {
                        "prompt_tokens": token_result.get("prompt_tokens", torch.empty(0, dtype=torch.long)),
                        "response_tokens": token_result.get("response_tokens", torch.empty(0, dtype=torch.long)),
                        "response_masks": token_result.get("response_masks", torch.empty(0, dtype=torch.long)),
                        "trajectory_reward": token_result.get("trajectory_reward", 0.0),
                        "chat_completions": token_result.get("chat_completions", []),
                        "metrics": token_result.get("metrics", {}),
                    }
                },
                "phase_outputs": {},
            })
        
        colorful_print(f"✅ Successfully formatted {len(formatted_results)} Chain of Experts results", "green")
        return formatted_results
        