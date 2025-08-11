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
        # 1.  Build workflow / DAG
        self.workflow          = workflow
        self.agent_cfgs, self.phases, self.connections = workflow.define_workflow()

        # 2.  One per-role engine (real LLM inference)
        self.role_engines = self._init_role_engines(
            env_class, env_args or {}, tokenizer, rollout_engine,
            config, trajectory_timeout, max_workers, engine_name, **kwargs
        )

        # 3.  Parent initialisation with *ChainCoordinatorAgent*
        super().__init__(
            engine_name=engine_name,
            tokenizer=tokenizer,
            rollout_engine=rollout_engine,
            config=config,
            agent_class=ChainCoordinatorAgent,
            agent_args=dict(
                phases      = self.phases,
                connections = self.connections,
                agent_cfgs  = self.agent_cfgs,
                role_engines= self.role_engines,
            ),
            env_class=env_class,
            env_args=env_args or {},
            max_steps= self.agent_cfgs[0].agent_args.get("max_steps", 10),
            trajectory_timeout=trajectory_timeout,
            max_workers=max_workers,
            n_parallel_agents=1,          # we give parent N coordinators later
        )

    def _init_role_engines(self, env_class, env_args, tokenizer, rollout_engine, config, trajectory_timeout, max_workers, engine_name, **kwargs):
        """
        Initialize individual execution engines for each agent.
        
        Each agent gets its own AgentExecutionEngine which:
        - Can connect to a separate vLLM instance via router
        - Handles its own model path and configuration
        - Uses the same async infrastructure as single-agent rLLM
        """
        role_engines = {}
        global_config = config  # keep reference to the full Hydra config for worker engines
        for agent_cfg in self.agent_cfgs:
            # Create engine args specific to this agent
            agent_engine_args = kwargs.copy()
            
            # Remove parameters that will be passed explicitly to avoid conflicts
            agent_engine_args.pop('max_response_length', None)
            agent_engine_args.pop('max_prompt_length', None)
            
            # Add agent-specific model configuration if provided
            if agent_cfg.model_path:
                agent_engine_args["model_path"] = agent_cfg.model_path
            
            # Add sampling parameters
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
        """Update environment instances for the training batch"""
        self.envs = envs
        
        # Create one ChainCoordinatorAgent per environment
        coordinators = []
        for env in envs:
            coordinator = ChainCoordinatorAgent(
                agent_id=self.agent_cfgs[0].agent_id,  # Coordinator ID not used for env update
                phases=self.phases,
                connections=self.connections,
                agent_cfgs=self.agent_cfgs,
                role_engines=self.role_engines,
            )
            coordinators.append(coordinator)
        
        # Update parent with coordinators
        super().update_envs_and_agents(envs, coordinators)

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
        
        # Reduce logging verbosity during validation
        is_validation = meta_info.get("validate", False)
        if not is_validation:
            colorful_print(f"\n{'='*80}", "cyan")
            colorful_print(f"🔗 CHAIN OF EXPERTS EXECUTION - Batch Size: {batch_size}", "cyan")
            colorful_print(f"{'='*80}", "cyan")
        
        # Execute trajectories step-by-step through the chain using parent's async generator
        results = []
        async def _collect():
            batch = []
            async for traj in self.trajectory_generator(timing_raw=timing_raw, mode="Token", **meta_info):
                batch.append(traj)
            return batch
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in an event loop, run in a temporary loop via a thread
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as ex:
                    fut = ex.submit(asyncio.run, _collect())
                    results = fut.result(timeout=600)
            else:
                results = loop.run_until_complete(_collect())
        except Exception as e:
            logger.error(f"Failed to collect trajectories: {e}")
            raise
        
        return self._format_results_for_training(results)
    
    def _format_results_for_training(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format the chain trajectory for PPO training.
        
        The training expects specific format with tokens, rewards, etc.
        """
        formatted_results = []
        for token_result in results:
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
        return formatted_results


class ChainCoordinatorAgent(BaseAgent):
    """Coordinates sequential multi-agent execution per environment step."""
    def __init__(self, agent_id: str, phases: List[WorkflowPhase], connections: List[WorkflowConnection], agent_cfgs: List[AgentConfig], role_engines: Dict[str, AgentExecutionEngine]):
        # BaseAgent has no custom __init__; just call object init
        super().__init__()
        self.phases = phases
        self.connections = connections
        self.agent_cfgs = {cfg.agent_id: cfg for cfg in agent_cfgs}
        self.role_engines = role_engines
        # one logical internal agent state per role id
        self.internal_agents: Dict[str, BaseAgent] = {}
        for cfg in agent_cfgs:
            engine = role_engines[cfg.agent_id]
            self.internal_agents[cfg.agent_id] = engine.agent_class(agent_id=cfg.agent_id, **engine.agent_args)

        from rllm.agents.agent import Trajectory
        self._trajectory = Trajectory()
    
    def reset(self):
        for agent in self.internal_agents.values():
            agent.reset()
        super().reset()

    # ------------------------------------------------------------------
    # BaseAgent interface overrides
    # ------------------------------------------------------------------

    @property
    def chat_completions(self) -> list[dict]:
        """Return a minimal system prompt so the chat-template parser always has input.

        The coordinator never actually talks to the model; the parent execution
        engine still expects a non-empty prompt when it calls
        `engine.get_model_response` for trajectory bookkeeping.  Returning a
        single system message avoids the `IndexError: list index out of range`
        inside the chat-template parser while keeping the content trivial.
        """
        msgs = [{"role": "system", "content": "Coordinator agent placeholder."}]
        
        if self._trajectory.steps and self._trajectory.steps[-1].model_response:
            assistant_content = self._trajectory.steps[-1].model_response
        else:
            assistant_content = "I am ready to coordinate the chain of experts."
        
        msgs.append({"role": "assistant", "content": assistant_content})
        return msgs
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        # Propagate environment feedback to each underlying agent.
        for agent in self.internal_agents.values():
            agent.update_from_env(observation, reward, done, info, **kwargs)
        # Record a minimal step so AgentExecutionEngine bookkeeping doesn't crash.
        from rllm.agents.agent import Step
        step = Step(observation=observation, reward=reward, done=done, info=info)
        self._trajectory.steps.append(step)
        return None

    def get_current_state(self):
        from typing import Optional
        return self._trajectory.steps[-1] if self._trajectory.steps else None
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        # Execute chain sequentially for a single environment step
        previous_responses: Dict[str, str] = {}
        for phase in self.phases:
            agent_id = phase.agent_ids[0]
            engine = self.role_engines[agent_id]
            agent = self.internal_agents[agent_id]
            # Prepare collaboration context from previous responses
            ctx = self._prepare_agent_context(agent_id, previous_responses)
            if hasattr(agent, "collaboration_prompt") and ctx:
                agent.collaboration_prompt = ctx
            # Call real model generation via engine.get_model_response
            text = self._get_real_response(engine, agent)
            agent.update_from_model(text)
            previous_responses[agent_id] = text
        # Parse final action from the last agent's response
        final_agent_id = self.phases[-1].agent_ids[0]
        final_text = previous_responses.get(final_agent_id, "")

        # Record assistant message in our own trajectory so that downstream token
        # accumulation logic sees an assistant message.
        from rllm.agents.agent import Step
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = final_text
        else:
            self._trajectory.steps.append(Step(model_response=final_text))

        return Action(action=self._parse_action_from_response(final_text))
    
    def _get_real_response(self, engine: AgentExecutionEngine, agent: BaseAgent) -> str:
        async def _gen():
            application_id = str(uuid.uuid4())
            # Safety: ensure at least one system prompt exists, otherwise chat parser crashes.
            prompt_msgs = agent.chat_completions
            if not prompt_msgs:
                prompt_msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            return await engine.get_model_response(prompt_msgs, application_id, max_tokens=engine.max_response_length, **engine.sampling_params)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as ex:
                    fut = ex.submit(asyncio.run, _gen())
                    return fut.result(timeout=60)
            else:
                return loop.run_until_complete(_gen())
        except Exception as e:
            logger.warning(f"Coordinator fallback due to generation error: {e}")
            return self._fallback_response(agent)
    
    def _prepare_agent_context(self, agent_id: str, previous: Dict[str, str]) -> str:
        parts = []
        for conn in self.connections:
            if conn.to_agent == agent_id and conn.from_agent in previous:
                txt = previous[conn.from_agent]
                if conn.transform_fn:
                    txt = conn.transform_fn(txt)
                parts.append(f"{conn.from_agent} -> {agent_id}:\n{txt}")
        return "\n\n".join(parts)
    
    def _fallback_response(self, agent: BaseAgent) -> str:
        # last-resort: deterministic placeholder
        return "[fallback] ```Left```"
    
    def _parse_action_from_response(self, response: str) -> int:
        import re
        m = re.search(r"```(\w+)```", response)
        if not m:
            return 1
        s = m.group(1).lower()
        return {"left":1, "down":2, "right":3, "up":4}.get(s, 1)
    