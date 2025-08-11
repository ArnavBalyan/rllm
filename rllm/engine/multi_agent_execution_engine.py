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
                    # Production note: batch execution timeout should be configurable
                    batch_timeout = meta_info.get('batch_execution_timeout', 600)
                    results = fut.result(timeout=batch_timeout)
            else:
                results = loop.run_until_complete(_collect())
        except Exception as e:
            colorful_print(f"❌ CRITICAL ERROR: Chain of Experts batch execution failed", "red")
            colorful_print(f"🔥 Error details: {str(e)}", "red")
            logger.error(f"Failed to collect trajectories: {e}")
            raise RuntimeError(f"Chain of Experts batch execution failed: {str(e)}") from e
        
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


class ChainCoordinatorAgent(BaseAgent):
    """Coordinates sequential multi-agent execution per environment step."""
    def __init__(self, agent_id: str, phases: List[WorkflowPhase], connections: List[WorkflowConnection], agent_cfgs: List[AgentConfig], role_engines: Dict[str, AgentExecutionEngine]):
        # Production audit log: Log chain configuration
        colorful_print(f"\n🏗️  INITIALIZING CHAIN OF EXPERTS COORDINATOR", "cyan")
        colorful_print(f"{'─'*60}", "cyan")
        colorful_print(f"🆔 Coordinator ID: {agent_id}", "white")
        colorful_print(f"📊 Number of Phases: {len(phases)}", "white")
        colorful_print(f"🔗 Number of Connections: {len(connections)}", "white")
        colorful_print(f"🤖 Number of Agents: {len(agent_cfgs)}", "white")
        
        # Log phase sequence
        phase_sequence = " → ".join([f"{phase.phase_id}[{','.join(phase.agent_ids)}]" for phase in phases])
        colorful_print(f"🎭 Phase Sequence: {phase_sequence}", "white")
        
        # Validate configuration
        if not phases:
            colorful_print(f"❌ ERROR: No phases configured!", "red")
            raise ValueError("Chain of Experts requires at least one phase")
        
        if not agent_cfgs:
            colorful_print(f"❌ ERROR: No agents configured!", "red")
            raise ValueError("Chain of Experts requires at least one agent")
        
        # Validate that all phases have corresponding agents
        for phase in phases:
            for agent_id_in_phase in phase.agent_ids:
                if agent_id_in_phase not in [cfg.agent_id for cfg in agent_cfgs]:
                    colorful_print(f"❌ ERROR: Phase {phase.phase_id} references unknown agent {agent_id_in_phase}!", "red")
                    raise ValueError(f"Unknown agent {agent_id_in_phase} in phase {phase.phase_id}")
        
        colorful_print(f"✅ Configuration validation passed", "green")
        
        # BaseAgent has no custom __init__; just call object init
        super().__init__()
        self.phases = phases
        self.connections = connections
        self.agent_cfgs = {cfg.agent_id: cfg for cfg in agent_cfgs}
        self.role_engines = role_engines
        
        # one logical internal agent state per role id
        self.internal_agents: Dict[str, BaseAgent] = {}
        for cfg in agent_cfgs:
            colorful_print(f"🔧 Initializing agent: {cfg.agent_id} ({cfg.agent_class.__name__})", "blue")
            engine = role_engines[cfg.agent_id]
            self.internal_agents[cfg.agent_id] = engine.agent_class(agent_id=cfg.agent_id, **engine.agent_args)
            colorful_print(f"   ✅ Agent {cfg.agent_id} initialized successfully", "green")

        # Minimal trajectory for bookkeeping used by AgentExecutionEngine
        from rllm.agents.agent import Trajectory
        self._trajectory = Trajectory()
        
        colorful_print(f"🎉 Chain of Experts Coordinator initialization complete!", "green")
        colorful_print(f"{'─'*60}\n", "cyan")
        
        # Mark initialization as complete for state validation
        self._initialization_complete = True
        # Track chain execution state
        self._chain_executed = False
    
    def reset(self):
        colorful_print(f"🔄 Resetting Chain of Experts Coordinator", "cyan")
        for agent in self.internal_agents.values():
            agent.reset()
        
        # Reset our own trajectory and execution state
        from rllm.agents.agent import Trajectory
        self._trajectory = Trajectory()
        self._chain_executed = False
        colorful_print(f"✅ Chain coordinator reset complete", "green")
        
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
        
        # Determine appropriate assistant content based on execution state
        if self._trajectory.steps and self._trajectory.steps[-1].model_response:
            assistant_content = self._trajectory.steps[-1].model_response
            # Ensure it's not empty or whitespace
            if not assistant_content.strip():
                raise RuntimeError("Chain coordinator has empty model response - this indicates a serious execution flow issue")
            colorful_print(f"📋 Using real chain execution result as assistant message", "blue")
        else:
            # During trajectory setup phase, provide a valid placeholder
            # The actual chain execution happens in update_from_model()
            assistant_content = "Chain of Experts coordinator ready for execution."
            colorful_print(f"📋 Using placeholder assistant message (chain hasn't executed yet)", "blue")
        
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
        # Log start of chain execution
        colorful_print(f"\n{'='*100}", "cyan")
        colorful_print(f"🔗 CHAIN OF EXPERTS COORDINATOR - Starting Chain Execution", "cyan")
        colorful_print(f"{'='*100}", "cyan")
        
        # Execute chain sequentially for a single environment step
        previous_responses: Dict[str, str] = {}
        expected_agents = [phase.agent_ids[0] for phase in self.phases]
        
        for phase in self.phases:
            agent_id = phase.agent_ids[0]
            
            # Validate agent exists
            if agent_id not in self.internal_agents:
                colorful_print(f"❌ CRITICAL ERROR: Agent {agent_id} not found in internal agents", "red")
                raise RuntimeError(f"Chain execution failed: Agent {agent_id} not properly initialized")
            
            if agent_id not in self.role_engines:
                colorful_print(f"❌ CRITICAL ERROR: Engine for agent {agent_id} not found", "red")
                raise RuntimeError(f"Chain execution failed: Engine for agent {agent_id} not found")
            
            engine = self.role_engines[agent_id]
            agent = self.internal_agents[agent_id]
            
            # Log agent start
            colorful_print(f"\n🤖 EXECUTING AGENT: {agent_id.upper()}", "yellow")
            colorful_print(f"{'─'*80}", "yellow")
            
            # Prepare collaboration context from previous responses
            ctx = self._prepare_agent_context(agent_id, previous_responses)
            # Also prepare a structured map of previous agent outputs
            prev_map = {conn.from_agent: previous_responses[conn.from_agent]
                        for conn in self.connections
                        if conn.to_agent == agent_id and conn.from_agent in previous_responses}
            
            # Update agent with collaboration context via proper MultiAgentBase mechanism
            if ctx:
                # Pass context through observation dict as expected by MultiAgentBase
                current_observation = agent.get_current_state().observation if agent.get_current_state() else None
                context_observation = {
                    "collaboration_prompt": ctx,
                    "base_observation": current_observation,
                    "previous_agents": prev_map
                }
                agent.update_from_env(context_observation, 0.0, False, {})
                colorful_print(f"📥 Input Context from Previous Agents:", "white")
                colorful_print(f"{ctx}", "white")
            else:
                colorful_print(f"📥 Input Context: [First agent - no previous context]", "white")
            
            # Log agent's current state/prompt
            if hasattr(agent, 'chat_completions') and agent.chat_completions:
                colorful_print(f"💭 Agent's Current Prompt:", "blue")
                for i, msg in enumerate(agent.chat_completions):
                    role_color = "green" if msg["role"] == "system" else "cyan" if msg["role"] == "user" else "yellow"
                    content_preview = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                    colorful_print(f"   [{i+1}] {msg['role'].upper()}: {content_preview}", role_color)
            else:
                colorful_print(f"❌ CRITICAL ERROR: Agent {agent_id} has no chat completions", "red")
                raise RuntimeError(f"Chain execution failed: Agent {agent_id} has no chat completions")
            
            # Call real model generation via engine.get_model_response
            text = self._get_real_response(engine, agent)
            
            # Validate response is not empty
            if not text or not text.strip():
                colorful_print(f"❌ CRITICAL ERROR: Agent {agent_id} returned empty response", "red")
                raise RuntimeError(f"Chain execution failed: Agent {agent_id} returned empty or whitespace-only response")
            
            # Log agent response
            colorful_print(f"📤 Agent Response:", "green")
            response_preview = text[:800] + "..." if len(text) > 800 else text
            colorful_print(f"{response_preview}", "green")
            
            agent.update_from_model(text)
            previous_responses[agent_id] = text
            
            colorful_print(f"✅ Agent {agent_id.upper()} completed successfully", "green")
        
        # Validate all expected agents executed
        if len(previous_responses) != len(expected_agents):
            colorful_print(f"❌ CRITICAL ERROR: Expected {len(expected_agents)} agents, but only {len(previous_responses)} executed", "red")
            raise RuntimeError(f"Chain execution incomplete: Expected {len(expected_agents)} agents, got {len(previous_responses)}")
        
        for expected_agent in expected_agents:
            if expected_agent not in previous_responses:
                colorful_print(f"❌ CRITICAL ERROR: Agent {expected_agent} did not execute", "red")
                raise RuntimeError(f"Chain execution failed: Agent {expected_agent} missing from execution results")
        
        # Parse final action from the last agent's response
        final_agent_id = self.phases[-1].agent_ids[0]
        final_agent = self.internal_agents[final_agent_id]
        
        # Get the action that was already parsed by the final agent during chain execution
        final_step = final_agent.get_current_state()
        if not final_step or not hasattr(final_step, 'action'):
            colorful_print(f"❌ CRITICAL ERROR: Final agent {final_agent_id} has no parsed action", "red")
            raise RuntimeError(f"Chain execution failed: Final agent {final_agent_id} has no parsed action")
        
        final_action_value = final_step.action
        
        # Log final decision
        colorful_print(f"\n🎯 FINAL CHAIN DECISION", "magenta")
        colorful_print(f"{'─'*80}", "magenta")
        colorful_print(f"🏛️  Final Agent: {final_agent_id.upper()}", "white")
        colorful_print(f"📜 Final Response: {previous_responses[final_agent_id]}", "white")
        
        colorful_print(f"⚡ Parsed Action: {final_action_value}", "yellow")
        
        # Log chain summary
        colorful_print(f"\n📊 CHAIN EXECUTION SUMMARY", "cyan")
        colorful_print(f"{'─'*80}", "cyan")
        colorful_print(f"🔢 Total Agents in Chain: {len(self.phases)}", "white")
        colorful_print(f"🎭 Agent Sequence: {' → '.join([phase.agent_ids[0].upper() for phase in self.phases])}", "white")
        colorful_print(f"💫 Final Action Value: {final_action_value}", "white")
        colorful_print(f"{'='*100}\n", "cyan")

        # Record assistant message in our own trajectory so that downstream token
        # accumulation logic sees an assistant message.
        from rllm.agents.agent import Step
        if self._trajectory.steps:
            self._trajectory.steps[-1].model_response = previous_responses[final_agent_id]
        else:
            self._trajectory.steps.append(Step(model_response=previous_responses[final_agent_id]))

        # Mark chain as executed
        self._chain_executed = True
        colorful_print(f"✅ Chain execution state updated - execution complete", "green")

        return Action(action=final_action_value)
    
    def _get_real_response(self, engine: AgentExecutionEngine, agent: BaseAgent) -> str:
        colorful_print(f"🚀 Calling LLM for response generation...", "blue")
        async def _gen():
            application_id = str(uuid.uuid4())
            colorful_print(f"🔑 Application ID: {application_id}", "blue")
            # Validate that agent has proper chat completions
            prompt_msgs = agent.chat_completions
            if not prompt_msgs:
                colorful_print(f"❌ CRITICAL ERROR: Agent has no chat completions", "red")
                raise RuntimeError(f"Agent {getattr(agent, 'agent_id', 'unknown')} has no chat completions - this indicates improper agent initialization or state management")
            return await engine.get_model_response(prompt_msgs, application_id, max_tokens=engine.max_response_length, **engine.sampling_params)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as ex:
                    colorful_print(f"⏳ Executing LLM call in thread pool...", "blue")
                    fut = ex.submit(asyncio.run, _gen())
                    # Production note: timeout should be configurable via agent config
                    generation_timeout = getattr(engine, 'generation_timeout', 60)
                    return fut.result(timeout=generation_timeout)
            else:
                colorful_print(f"⏳ Executing LLM call directly...", "blue")
                return loop.run_until_complete(_gen())
        except Exception as e:
            colorful_print(f"❌ CRITICAL ERROR: LLM generation failed for agent", "red")
            colorful_print(f"🔥 Error details: {str(e)}", "red")
            colorful_print(f"💥 This indicates a serious issue with model inference", "red")
            logger.error(f"Chain of Experts execution failed due to LLM generation error: {e}")
            raise RuntimeError(f"Chain of Experts failed: LLM generation error for agent - {str(e)}") from e
    
    def _prepare_agent_context(self, agent_id: str, previous: Dict[str, str]) -> str:
        parts = []
        for conn in self.connections:
            if conn.to_agent == agent_id and conn.from_agent in previous:
                txt = previous[conn.from_agent]
                if conn.transform_fn:
                    txt = conn.transform_fn(txt)
                parts.append(f"{conn.from_agent} -> {agent_id}:\n{txt}")
        return "\n\n".join(parts)
    