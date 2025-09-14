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
    PROPOSER = "proposer"  
    CRITIC = "critic"      
    JUDGE = "judge"        
    SPECIALIST = "specialist"
    AGGREGATOR = "aggregator"


@dataclass
class AgentConfig:
    agent_id: str
    agent_class: type
    agent_args: Dict[str, Any] = field(default_factory=dict)
    role: AgentRole = AgentRole.SPECIALIST
    model_path: Optional[str] = None
    max_response_length: int = 4096
    max_prompt_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class WorkflowConnection:
    from_agent: str
    to_agent: str
    transform_fn: Optional[callable] = None


@dataclass
class WorkflowPhase:
    """
    Represents a logical execution phase in the multi-agent workflow.
    The workflow is a Directed Acyclic Graph (DAG) of agents connected by connections.
    Please make sure to not introduce cyclic dependency between agents.
    
    A phase contains exactly one agent that executes in this phase.
    For Chain of Experts, each phase contains one agent in a linear chain.
    
    """
    phase_id: str
    agent_id: str  # Single agent ID for this phase
    execution_mode: str = "sequential"
    description: Optional[str] = None 


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
        
        for i in range(len(self.agent_configs_list) - 1):
            connections.append(WorkflowConnection(
                from_agent=self.agent_configs_list[i].agent_id,
                to_agent=self.agent_configs_list[i + 1].agent_id
            ))
        
        for i, config in enumerate(self.agent_configs_list):
            phases.append(WorkflowPhase(
                phase_id=f"phase_{i}",
                agent_id=config.agent_id,
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
        # Per-environment agents: list indexed by env_idx, each a dict of role_id -> agent
        self.env_agents: List[Dict[str, BaseAgent]] = []
        
        self.reward_mode = config.multi_agent.reward_mode
        
        self.role_engines = self._init_role_engines(
            env_class, env_args, tokenizer, rollout_engine,
            config, trajectory_timeout, max_workers, engine_name, **kwargs
        )
        
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
            
            agent_rollout_engine = rollout_engine[agent_cfg.agent_id]
            
            role_engines[agent_cfg.agent_id] = AgentExecutionEngine(
                engine_name=engine_name,
                tokenizer=tokenizer,
                rollout_engine=agent_rollout_engine,
                config=global_config,
                agent_class=agent_cfg.agent_class,
                agent_args=agent_cfg.agent_args,
                env_class=env_class,
                env_args=env_args,
                n_parallel_agents=1,
                max_response_length=config.data.max_response_length,
                max_prompt_length=config.data.max_prompt_length,
                trajectory_timeout=trajectory_timeout,
                max_workers=max_workers,
                **agent_engine_args
            )
        return role_engines
    
    def update_envs_and_agents(self, envs: List[BaseEnv]):
        self.envs = envs
        # Initialize isolated agents per environment
        self.env_agents = []
        for _ in range(len(envs)):
            per_env_agents: Dict[str, BaseAgent] = {}
            for cfg in self.agent_cfgs:
                role_engine = self.role_engines[cfg.agent_id]
                init_args = role_engine.agent_args.copy()
                per_env_agents[cfg.agent_id] = role_engine.agent_class(agent_id=cfg.agent_id, **init_args)
            self.env_agents.append(per_env_agents)
    
    async def run_workflow_trajectory_async(self, env_idx: int, application_id: str, seed: int = 0, mode: str = "Token", **kwargs) -> Dict[str, Any]:
        env = self.envs[env_idx]
        
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(None, env.reset)
        info["max_steps"] = self.max_steps
        # Use per-environment agent instances to avoid state mixing across envs
        agents = self.env_agents[env_idx]
        
        for agent_id, agent in agents.items():
            agent.reset()
            colorful_print(f"🔄 Reset {agent_id}", "yellow")
        
        total_reward = 0.0
        final_tokens = []
        final_masks = []
        completed_turns = 0
        
        # Agent-level response tokens and masks tracking
        agent_response_tokens = {}
        agent_response_masks = {}
        agent_chat_completions = {}
        agent_matches = {}  # Track matches with final agent per step
        for phase in self.phases:
            agent_response_tokens[phase.agent_id] = []
            agent_response_masks[phase.agent_id] = []
            agent_chat_completions[phase.agent_id] = []
            agent_matches[phase.agent_id] = 0
        
        from rllm.agents.utils import convert_messages_to_tokens_and_masks
        final_agent_id = self.phases[-1].agent_id
        engine = self.role_engines[final_agent_id]
        
        prompt_tokens = []
        termination_reason = None
        response_token_len = 0

        for step_idx in range(self.max_steps):
            
            for agent_id, agent in agents.items():
                agent.update_from_env(observation, 0.0, False, info)
            
            if step_idx == 0:
                final_agent = agents[final_agent_id]
                initial_messages = final_agent.chat_completions
                prompt_tokens, _ = convert_messages_to_tokens_and_masks(
                    initial_messages,
                    tokenizer=engine.tokenizer,
                    parser=engine.chat_parser,
                    contains_first_msg=True,
                    contains_generation_msg=True
                )
            
            phase_responses = {}
            phase_actions = {}  # Track extracted actions from each agent
            phase_upstream_contexts = {}  # Track upstream context sent to each agent for audit
            final_action = None
            step_start_response_len = response_token_len  # Track tokens added in this step
            
            for phase_idx, phase in enumerate(self.phases):
                agent_id = phase.agent_id
                agent = agents[agent_id]
                engine = self.role_engines[agent_id]
                
                upstream_response = self._get_upstream_phase_response(agent_id, phase_responses)
                phase_upstream_contexts[agent_id] = upstream_response  # Store for audit
                
                prompt_msgs = agent.chat_completions.copy()
                
                # UPSTREAM CONTEXT INJECTION: Add to model call only, don't modify agent state
                upstream_msg = {"role": "user", "content": upstream_response}
                prompt_msgs.append(upstream_msg)
                
                # Calculate remaining tokens for this phase
                max_tokens = engine.max_response_length - response_token_len
                
                response = await engine.get_model_response(
                    prompt_msgs, 
                    application_id, 
                    max_tokens=max_tokens,
                    **engine.sampling_params
                )
                
                action = agent.update_from_model(response)
                action_str = action.action 
                
                phase_responses[agent_id] = response
                phase_actions[agent_id] = action_str  # Store extracted action
                # Diagnostics: log generated token length per phase
                try:
                    gen_len = len(engine.tokenizer.encode(response, add_special_tokens=False))
                    print(f"MA_PHASE_TOKENS step={step_idx} phase={phase_idx} agent={agent_id} gen_tokens={gen_len} action={action_str}", flush=True)
                except Exception:
                    pass
                final_action = action_str
            
            # Track agent action matches with final agent (for contribution reward mode)
            final_action_str = phase_actions[final_agent_id]
                
            for agent_id in phase_actions:
                if agent_id != final_agent_id:
                    agent_action_str = phase_actions[agent_id]
                    if agent_action_str == final_action_str:
                        agent_matches[agent_id] += 1
                        print(f"MA_ACTION_MATCH step={step_idx} agent={agent_id} action={agent_action_str} matches_final={final_action_str}", flush=True)
            
            if termination_reason == "TRUNCATION":
                break
            
            # Always call env.step with final_action (matching single agent behavior)
            observation, reward, done, info = await loop.run_in_executor(
                None, env.step, final_action
            )
            # print(f"DEBUG: env_idx={env_idx}, step={step_idx}, action={final_action}, reward={reward}, done={done}")
            total_reward = reward
            # print(f"DEBUG: phase_responses={' '.join(phase_responses[final_agent_id].split()[:500])}{'...' if len(phase_responses[final_agent_id].split()) > 500 else ''}")
            
            # Update final agent's trajectory with environment feedback
            final_agent = agents[final_agent_id]
            if observation:
                obs_str = str(observation)
            
            # Process tokenization for each agent in the workflow
            print(f"🔄 TOKENIZATION_LOOP: Processing {len(agents)} agents: {list(agents.keys())}")
            
            any_agent_truncated = False
            for agent_id, agent in agents.items():
                print(f"🔄 PROCESSING_AGENT: {agent_id}")
                agent_tokens, agent_masks, agent_truncated = self._process_agent_tokenization(
                    agent_id, agent, agents, mode
                )
                
                print(f"🔄 AGENT_RESULT: {agent_id} returned {len(agent_tokens)} tokens, {len(agent_masks)} masks, truncated={agent_truncated}")
                
                # Always collect tokens and masks for all agents
                agent_response_tokens[agent_id].extend(agent_tokens)
                agent_response_masks[agent_id].extend(agent_masks)
                audit_chat_completions = agent.chat_completions.copy()
                audit_chat_completions.append({"role": "user", "content": phase_upstream_contexts[agent_id]})
                agent_chat_completions[agent_id] = audit_chat_completions
                
                # Track if any agent was truncated
                if agent_truncated:
                    any_agent_truncated = True
            
            # Apply truncation penalties after processing all agents
            if any_agent_truncated:
                total_reward = 0.0
                termination_reason = "TRUNCATION"
                
            if termination_reason == "TRUNCATION":
                break
                
            if done:
                break
            completed_turns = completed_turns + 1
        
        if mode == "Token":
            # Derive optional metadata for downstream consumers
            data_source = "unknown"
            uid = f"unknown_{env_idx}"
            try:
                if hasattr(env, 'task_data') and env.task_data:
                    data_source = env.task_data["data_source"]
                    uid = env.task_data["uid"]
                elif hasattr(env, 'entry') and env.entry:
                    data_source = env.entry["data_source"]
                    uid = env.entry["uid"]
            except Exception:
                pass

            # Prepare phase-level data for individual agent training
            phase_data = {}
            for phase in self.phases:
                agent_id = phase.agent_id
                if agent_id in agent_response_tokens:
                    if self.reward_mode == "partial" and total_reward == 1.0 and completed_turns > 0:
                        if agent_id == final_agent_id:
                            agent_reward = total_reward
                        else:
                            agent_reward = agent_matches[agent_id] / completed_turns
                    else:
                        agent_reward = total_reward
                    
                    phase_data[agent_id] = {
                        "response_tokens": torch.tensor(agent_response_tokens[agent_id], dtype=torch.long),
                        "response_masks": torch.tensor(agent_response_masks[agent_id], dtype=torch.long),
                        "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),  # Shared initial context
                        "trajectory_reward": agent_reward,
                        "phase_id": phase.phase_id,
                        "agent_role": phase.agent_id,
                        "chat_completions": agent_chat_completions[agent_id]
                    }
            
            all_response_tokens = []
            all_response_masks = []
            for phase in self.phases:
                agent_id = phase.agent_id
                if agent_id in agent_response_tokens:
                    all_response_tokens.extend(agent_response_tokens[agent_id])
                    all_response_masks.extend(agent_response_masks[agent_id])
            
            return {
                "idx": env_idx,
                "trajectory_reward": total_reward,
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),     # Initial context (like single-agent)
                "response_tokens": torch.tensor(all_response_tokens, dtype=torch.long),  # All workflow responses (combined)
                "response_masks": torch.tensor(all_response_masks, dtype=torch.long),
                "phase_data": phase_data,  # Individual phase-level data for agent training
                "data_source": data_source,
                "uid": uid,
                "metrics": {
                    "steps": len(agents[self.phases[-1].agent_id]._trajectory.steps),
                    "phases_executed": len(self.phases),
                    "total_reward": total_reward,
                    "completed_turns": completed_turns,
                    "prompt_tokens_length": len(prompt_tokens),
                    "response_tokens_length": len(all_response_tokens),
                    "response_masks_length": len(all_response_masks),
                    "phase_count": len(phase_data)
                }
            }
        else:
            return {
                "idx": env_idx,
                "trajectory": agents[self.phases[-1].agent_id]._trajectory,
                "total_reward": total_reward,
                "chat_completions": chat_completions
            }
    
    def _get_upstream_phase_response(self, current_agent_id: str, phase_responses: Dict[str, str]) -> str:
        """Get upstream phase response as a string for injection into prompt"""
        context_parts = []
        
        for conn in self.connections:
            if conn.to_agent == current_agent_id and conn.from_agent in phase_responses:
                previous_response = phase_responses[conn.from_agent]
                context_parts.append(f"Input from {conn.from_agent.upper()}: {previous_response}")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _process_agent_tokenization(self, agent_id: str, agent: BaseAgent, agents: Dict[str, BaseAgent], mode: str) -> Tuple[List[int], List[int], bool]:
        """
        Process tokenization for a specific agent and return tokens, masks, and truncation status.
        
        Args:
            agent_id: ID of the agent to process
            agent: The agent instance
            agents: Dictionary of all agents
            mode: Execution mode ("Token" or other)
            
        Returns:
            Tuple of (tokens, masks, is_truncated)
        """
        from rllm.agents.utils import get_recent_assistant_user_messages, convert_messages_to_tokens_and_masks
        
        engine = self.role_engines[agent_id]
        chat_completions_messages = agent.chat_completions
        assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)
        
        assert assistant_message is not None or mode != "Token", f"Assistant messages is none for agent {agent_id} when accumulating token trajectories which should be conversations. This should not happen."
        assert env_messages is not None or mode != "Token", f"Environment messages is none for agent {agent_id} when accumulating token trajectories which should be conversations. This should not happen."
        
        assistant_msg_tokens, assistant_msg_masks = [], []
        env_msg_tokens, env_msg_masks = [], []
        
        if assistant_message:
            assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
                [assistant_message], 
                tokenizer=engine.tokenizer, 
                parser=engine.chat_parser, 
                contains_first_msg=False, 
                contains_generation_msg=False
            )
        
        if env_messages:
            env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                env_messages, 
                tokenizer=engine.tokenizer, 
                parser=engine.chat_parser, 
                contains_first_msg=False, 
                contains_generation_msg=True
            )
        
        # Combine assistant and environment tokens
        combined_tokens = assistant_msg_tokens + env_msg_tokens
        combined_masks = assistant_msg_masks + env_msg_masks
        
        # Check for truncation using agent's max response length
        if len(combined_tokens) >= engine.max_response_length:
            # Truncation length (matching single agent calculation)
            truncation_length = engine.max_response_length - len(combined_tokens)
            
            # Truncate the response and masks
            if truncation_length < 0:
                truncated_response_tokens = combined_tokens[:truncation_length]
                truncated_response_masks = combined_masks[:truncation_length]
            else:
                # Edge case where the response is exactly the max response length
                truncated_response_tokens = combined_tokens
                truncated_response_masks = combined_masks
            
            # Log truncation details
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"MULTI_AGENT_TRUNCATION: Agent {agent_id} output truncated. Original length: {len(combined_tokens)}, Max allowed: {engine.max_response_length}, Truncated to: {len(truncated_response_tokens)}, Assistant msg tokens: {len(assistant_msg_tokens)}, Env msg tokens: {len(env_msg_tokens)}")

            # Apply reward penalty if assistant response was truncated (matching single agent logic)
            if len(combined_tokens) - len(env_msg_tokens) > engine.max_response_length:
                # Set reward to 0 for this agent's current step
                cur_step = agent.get_current_state()
                if hasattr(cur_step, 'reward'):
                    cur_step.reward = 0.0
            
            return truncated_response_tokens, truncated_response_masks, True
        
        return combined_tokens, combined_masks, False
                
    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Token", **kwargs):
        """Generate trajectories for all environments using workflow execution"""
        if timing_raw is None:
            timing_raw = {}
        
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"
        
        max_concurrency = len(self.envs)
        
        # Add logging to track environment count
        # print(f"\n{'='*60}")
        # print(f"DEBUG: trajectory_generator called")
        # print(f"  - Environment count: {len(self.envs)}")
        # print(f"  - Max concurrency: {max_concurrency}")
        # print(f"{'='*60}\n")
        
        # Wake up all rollout engines before starting trajectory generation
        for engine in self.role_engines.values():
            engine.rollout_engine.wake_up()
                
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
                traceback.print_exc()
                raise e
        
        tasks = [launch_workflow_trajectory(i) for i in range(len(self.envs))]
        
        # print(f"\n{'='*60}")
        # print(f"DEBUG: Created {len(tasks)} trajectory tasks")
        # print(f"  - Tasks will run concurrently with asyncio.as_completed")
        # print(f"{'='*60}\n")
        
        completed_trajectories = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                completed_trajectories.append(result)
                yield result
            except Exception as e:
                raise e
        
        # Visualize trajectory turns at the end using metrics from completed_trajectories
        if completed_trajectories:
            print(f"\n{'='*60}")
            print("TRAJECTORY TURNS SUMMARY:")
            print(f"{'='*60}")
            for idx, traj in enumerate(completed_trajectories):
                steps = traj["metrics"]["steps"]
                bar = "█" * steps
                print(f"Traj {idx}: {bar} ({steps} turns)")
            print(f"{'='*60}\n")
        
        # Sleep all rollout engines after completing all trajectories
        for engine in self.role_engines.values():
            engine.rollout_engine.sleep()
        # Add delay to ensure clean completion
        import time as ts_imp
        ts_imp.sleep(1)
        
        
    def execute_chain_of_experts_batch(
        self, 
        timing_raw: Dict[str, Any] = None, 
        meta_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute Chain of Experts workflow on a training batch"""
        batch_size = len(self.envs)
        
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
                    batch_timeout = meta_info['batch_execution_timeout']
                    results = fut.result(timeout=batch_timeout)
            else:
                results = loop.run_until_complete(_collect_batch())
        except Exception as e:
            raise RuntimeError(f"Mutli-Agent batch execution failed: {str(e)}") from e
        
        return results
    