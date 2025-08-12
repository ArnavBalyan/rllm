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


class MultiAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, workflow: BaseWorkflow, env_class, *, env_args=None,
                 engine_name="verl", tokenizer=None, rollout_engine=None,
                 config=None, trajectory_timeout=None, max_workers=64, **kwargs):
        self.workflow = workflow
        self.agent_cfgs, self.phases, self.connections = workflow.define_workflow()

        # Create individual execution engines for each agent (no coordinator)
        self.role_engines = self._init_role_engines(
            env_class, env_args, tokenizer, rollout_engine,
            config, trajectory_timeout, max_workers, engine_name, **kwargs
        )

        # Initialize as regular execution engine but with workflow agent class
        super().__init__(
            engine_name=engine_name,
            tokenizer=tokenizer,
            rollout_engine=rollout_engine,
            config=config,
            agent_class=WorkflowAgent,  # New clean workflow agent
            agent_args=dict(
                phases=self.phases,
                connections=self.connections,
                agent_cfgs=self.agent_cfgs,
                role_engines=self.role_engines,
            ),
            env_class=env_class,
            env_args=env_args or {},
            max_steps=self.agent_cfgs[0].agent_args.get("max_steps", 10),
            trajectory_timeout=trajectory_timeout,
            max_workers=max_workers,
            n_parallel_agents=1,
        )

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
        self.envs = envs
        
        # Create WorkflowAgent instances for each environment
        workflow_agents = []
        for i, env in enumerate(envs):
            workflow_agent = WorkflowAgent(
                agent_id=f"workflow_agent_{i}",
                phases=self.phases,
                connections=self.connections,
                agent_cfgs=self.agent_cfgs,
                role_engines=self.role_engines,
            )
            workflow_agents.append(workflow_agent)
        
        super().update_envs_and_agents(envs, workflow_agents)
    
    def execute_chain_of_experts_batch(
        self, 
        timing_raw: Dict[str, Any] = None, 
        meta_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Chain of Experts on a training batch.        
        Each environment step goes through the complete chain sequentially.
        
        Returns:
            List of workflow results for each item in the batch
        """
        batch_size = len(self.envs)
        
        is_validation = meta_info.get("validate", False)
        print("Reached multi agent execution engine with batch size", batch_size)
        results = []
        async def _collect():
            batch = []
            async for traj in self.trajectory_generator(timing_raw=timing_raw, mode="Token", **meta_info):
                batch.append(traj)
            return batch
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as ex:
                    fut = ex.submit(asyncio.run, _collect())
                    batch_timeout = meta_info.get('batch_execution_timeout', self.trajectory_timeout)
                    results = fut.result(timeout=batch_timeout)
            else:
                results = loop.run_until_complete(_collect())
        except Exception as e:
            raise RuntimeError(f"Chain of Experts batch execution failed: {str(e)}") from e
        print("Multi agent execution batch complete, all trajectories across all steps done, will exit now")
        return self._format_results_for_training(results)
    
    def _format_results_for_training(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format the chain trajectory for PPO training.
        
        The training expects specific format with tokens, rewards, etc.
        """
        if not results:
            raise RuntimeError("Chain of Experts execution produced no results - this indicates a critical system failure")
        
        formatted_results = []
        for i, token_result in enumerate(results):
            if not token_result:
                raise RuntimeError(f"Chain of Experts result {i} is empty - this indicates incomplete trajectory execution")
            
            # Validate required fields
            required_fields = ["trajectory_reward", "chat_completions"]
            for field in required_fields:
                if field not in token_result:
                    colorful_print(f"❌ Missing required field '{field}' in result {i}", "red")
                    raise ValueError(f"Chain of Experts result validation failed: missing required field '{field}' in trajectory result {i}")
            
            # token_result keys documented in AgentExecutionEngine.run_agent_trajectory_async (mode="Token")
            formatted_results.append({
                "workflow_type": self.workflow.workflow_id,
                "batch_idx": token_result.get("idx", 0),
                "agent_trajectories": {
                    # attribute to final agent id for reward accounting
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


class WorkflowAgent(BaseAgent):
    """
    Clean workflow agent that manages multiple internal agents following single agent paradigm.
    Executes agents sequentially according to workflow phases and passes context between them.
    """
    
    def __init__(self, agent_id: str, phases: List[WorkflowPhase], connections: List[WorkflowConnection], 
                 agent_cfgs: List[AgentConfig], role_engines: Dict[str, AgentExecutionEngine]):
        super().__init__()
        self.agent_id = agent_id
        self.phases = phases
        self.connections = connections
        self.agent_cfgs = {cfg.agent_id: cfg for cfg in agent_cfgs}
        self.role_engines = role_engines
        
        # Create internal agents
        self.internal_agents: Dict[str, BaseAgent] = {}
        for cfg in agent_cfgs:
            engine = role_engines[cfg.agent_id]
            agent_init_args = engine.agent_args.copy()
            self.internal_agents[cfg.agent_id] = engine.agent_class(agent_id=cfg.agent_id, **agent_init_args)

        from rllm.agents.agent import Trajectory
        self._trajectory = Trajectory()
        self._current_observation = None
        self._current_reward = 0.0
        self._current_done = False
        self._current_info = {}
        
    def reset(self):
        """Reset all internal agents and workflow state"""
        for agent_id, agent in self.internal_agents.items():
            agent.reset()
            colorful_print(f"🔄 Reset {agent_id} for new episode", "yellow")
        
        from rllm.agents.agent import Trajectory
        self._trajectory = Trajectory()
        self._current_observation = None
        self._current_reward = 0.0
        self._current_done = False
        self._current_info = {}
        
        colorful_print(f"✅ WorkflowAgent reset complete", "green")

    @property
    def chat_completions(self) -> list[dict]:
        """Return the first agent's chat completions to bootstrap the workflow"""
        # The workflow starts with the first phase's agent
        if self.phases:
            first_agent_id = self.phases[0].agent_ids[0]
            first_agent = self.internal_agents[first_agent_id]
            return first_agent.chat_completions
        else:
            # Fallback if no phases defined
            return [{"role": "system", "content": "Empty workflow - no phases defined"}]

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update all internal agents with environment state for chain execution"""
        colorful_print(f"🌍 WorkflowAgent updating all agents from environment", "cyan")
        
        # Store current environment state
        self._current_observation = observation
        self._current_reward = reward
        self._current_done = done
        self._current_info = info
        
        # Update all internal agents with the same environment state
        # They will all need access to the current observation for the chain execution
        for agent_id, agent in self.internal_agents.items():
            colorful_print(f"📤 Updating {agent_id} with environment state", "yellow")
            agent.update_from_env(observation, reward, done, info, **kwargs)
        
    def update_from_model(self, response: str, **kwargs) -> Action:
        """Execute the complete workflow chain within this single step"""
        colorful_print(f"\n{'='*100}", "cyan")
        colorful_print(f"🔗 WORKFLOW EXECUTION - Sequential Agent Chain", "cyan")
        colorful_print(f"{'='*100}", "cyan")
        
        # Execute all agents in the workflow sequentially
        previous_responses: Dict[str, str] = {}
        final_action = None
        
        for phase_idx, phase in enumerate(self.phases):
            agent_id = phase.agent_ids[0]
            agent = self.internal_agents[agent_id]
            engine = self.role_engines[agent_id]
            
            colorful_print(f"\n🎯 Phase {phase_idx + 1}/{len(self.phases)}: {agent_id.upper()}", "yellow")
            colorful_print("-" * 50, "white")
            
            # Inject context from previous agents in this chain
            self._inject_chain_context(agent_id, previous_responses)
            
            # Make LLM call for this agent
            prompt_msgs = agent.chat_completions
            colorful_print(f"🤖 Making LLM call for {agent_id}", "blue")
            
            # Get real LLM response using the agent's dedicated engine
            agent_response = self._get_agent_response(engine, agent)
            
            # Update agent with response
            action = agent.update_from_model(agent_response, **kwargs)
            previous_responses[agent_id] = agent_response
            final_action = action
            
            colorful_print(f"✅ {agent_id} complete → action: {action.action}", "green")
        
        # Store the complete workflow step
        from rllm.agents.agent import Step
        step = Step(
            observation=self._current_observation,
            model_response=f"Chain execution: {' → '.join(previous_responses.keys())}",
            action=final_action.action if final_action else "0",
            reward=self._current_reward,
            done=self._current_done,
            info=self._current_info.copy()
        )
        self._trajectory.steps.append(step)
        
        colorful_print(f"\n🎉 Complete workflow chain executed! Final action: {final_action.action if final_action else '0'}", "green")
        colorful_print(f"{'='*100}", "cyan")
        
        return final_action if final_action else Action(action="0")
    
    def _inject_chain_context(self, current_agent_id: str, previous_responses: Dict[str, str]):
        """Inject context from previous agents in the current chain execution"""
        context_parts = []
        
        # Find connections to current agent and add context from previous responses
        for conn in self.connections:
            if conn.to_agent == current_agent_id and conn.from_agent in previous_responses:
                context_parts.append(f"Input from {conn.from_agent.upper()}: {previous_responses[conn.from_agent]}")
                
        if context_parts:
            chain_context = "\n\n".join(context_parts)
            current_agent = self.internal_agents[current_agent_id]
            if hasattr(current_agent, 'multi_agent_context'):
                current_agent.multi_agent_context["chain_context"] = chain_context
                colorful_print(f"📥 Injected context into {current_agent_id}: {len(chain_context)} chars", "cyan")
    
    def _get_agent_response(self, engine: AgentExecutionEngine, agent: BaseAgent) -> str:
        """Get LLM response for a specific agent using its dedicated engine"""
        import asyncio
        import uuid
        
        async def _get_response():
            application_id = str(uuid.uuid4())
            prompt_msgs = agent.chat_completions
            
            colorful_print(f"📤 Sending {len(prompt_msgs)} messages to LLM", "blue")
            response = await engine.get_model_response(
                prompt_msgs, 
                application_id, 
                max_tokens=engine.max_response_length, 
                **engine.sampling_params
            )
            colorful_print(f"📥 Received response: {response[:100]}{'...' if len(response) > 100 else ''}", "green")
            return response
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to handle this differently
                # For now, return a mock response - in production this needs proper async handling
                return f"```Right``` [Agent: {agent.agent_id}]"
            else:
                return loop.run_until_complete(_get_response())
        except Exception as e:
            colorful_print(f"❌ LLM call failed for {agent.agent_id}: {e}", "red")
            return f"```Right``` [Error fallback for {agent.agent_id}]"
        
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
        
    def get_current_state(self):
        return self._trajectory.steps[-1] if self._trajectory.steps else None
    