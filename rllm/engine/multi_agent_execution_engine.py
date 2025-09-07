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
    
    A phase is a collection of agents that execute together in a specific mode:
    - Eg. Chain of Experts, each phase contains exactly one agent in a linear chain
    - The phase defines how agents execute (sequential/parallel) and which agents participate
    
    """
    phase_id: str
    agent_ids: List[str]  
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
        # Per-environment agents: list indexed by env_idx, each a dict of role_id -> agent
        self.env_agents: List[Dict[str, BaseAgent]] = []
        
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
        trajectory = Trajectory()
        
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
        chat_completions = []
        completed_turns = 0
        
        from rllm.agents.utils import convert_messages_to_tokens_and_masks
        final_agent_id = self.phases[-1].agent_ids[0]
        engine = self.role_engines[final_agent_id]
        
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        termination_reason = None
        
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
            final_action = None
            step_start_response_len = response_token_len  # Track tokens added in this step
            
            for phase_idx, phase in enumerate(self.phases):
                agent_id = phase.agent_ids[0]
                agent = agents[agent_id]
                engine = self.role_engines[agent_id]
                
                self._inject_workflow_context(agent_id, phase_responses, agents)
                
                prompt_msgs = agent.chat_completions
                
                # Calculate remaining tokens for this phase
                max_tokens = engine.max_response_length - response_token_len
                
                response = await engine.get_model_response(
                    prompt_msgs, 
                    application_id, 
                    max_tokens=max_tokens,
                    **engine.sampling_params
                )
                
                action = agent.update_from_model(response)
                phase_responses[agent_id] = response
                # Diagnostics: log generated token length per phase
                try:
                    gen_len = len(engine.tokenizer.encode(response, add_special_tokens=False))
                    print(f"MA_PHASE_TOKENS step={step_idx} phase={phase_idx} agent={agent_id} gen_tokens={gen_len}", flush=True)
                except Exception:
                    pass
                final_action = action
            
            if termination_reason == "TRUNCATION":
                break
            
            if final_action:
                observation, reward, done, info = await loop.run_in_executor(
                    None, env.step, final_action.action
                )
                print(f"DEBUG: env_idx={env_idx}, step={step_idx}, action={final_action.action}, reward={reward}, done={done}")
                total_reward += reward
                
                step = Step(
                    observation=observation,
                    model_response=f"Workflow: {' → '.join(phase_responses.keys())}",
                    action=final_action.action,
                    reward=reward,
                    done=done,
                    info=info.copy()
                )
                trajectory.steps.append(step)
                
                if observation:
                    obs_str = str(observation)

                await self._synchronize_agent_histories_with_final_decision(
                    step_idx, observation, final_action, phase_responses, agents
                )
                
                # POST-STEP TOKENIZATION (like single-agent)
                final_agent_id = self.phases[-1].agent_ids[0]
                final_agent = agents[final_agent_id]
                final_engine = self.role_engines[final_agent_id]
                
                from rllm.agents.utils import get_recent_assistant_user_messages
                chat_completions_messages = final_agent.chat_completions
                assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)
                
                assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
                assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
                assistant_msg_tokens, assistant_msg_masks = [], []
                env_msg_tokens, env_msg_masks = [], []
                if assistant_message:
                    assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=final_engine.tokenizer, parser=final_engine.chat_parser, contains_first_msg=False, contains_generation_msg=False)
                if env_messages:
                    env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=final_engine.tokenizer, parser=final_engine.chat_parser, contains_first_msg=False, contains_generation_msg=True)

                response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
                if response_token_len >= final_engine.max_response_length:
                    # Truncation length 
                    truncation_length = final_engine.max_response_length - response_token_len
                    # Truncate the response and masks
                    if truncation_length < 0:
                        truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                        truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                    else:
                        # Edge case where the response is exactly the max response length (like single-agent)
                        truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                        truncated_response_masks = assistant_msg_masks + env_msg_masks
                    # Update token collections (like single-agent)
                    response_tokens.extend(truncated_response_tokens)
                    response_masks.extend(truncated_response_masks)
                    
                    # Log truncation details (like single-agent)
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"MULTI_AGENT_TRUNCATION: Final agent output truncated. Original length: {response_token_len}, Max allowed: {final_engine.max_response_length}, Truncated to: {len(truncated_response_tokens)}, Assistant msg tokens: {len(assistant_msg_tokens)}, Env msg tokens: {len(env_msg_tokens)}")

                    # Apply reward penalty if assistant response was truncated (like single-agent)
                    if response_token_len - len(env_msg_tokens) > final_engine.max_response_length:
                        total_reward = 0.0
                    termination_reason = "TRUNCATION"
                    # Handle returning
                    break

                # Update the token version of trajectory (like single-agent)
                response_tokens.extend(assistant_msg_tokens)
                response_masks.extend(assistant_msg_masks)

                if hasattr(final_agent, 'chat_completions'):
                    chat_completions = final_agent.chat_completions
                
                if done:
                    break
                
                # Add environment messages after done check
                response_tokens.extend(env_msg_tokens)
                response_masks.extend(env_msg_masks)
            completed_turns = completed_turns + 1
        
        if mode == "Token":
            
            return {
                "idx": env_idx,
                "trajectory_reward": total_reward,
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),     # Initial context (like single-agent)
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),  # All workflow responses
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "chat_completions": chat_completions,
                "metrics": {
                    "workflow_steps": len(trajectory.steps),
                    "phases_executed": len(self.phases),
                    "total_reward": total_reward,
                    "completed_turns": completed_turns,
                    "prompt_tokens_length": len(prompt_tokens),
                    "response_tokens_length": len(response_tokens),
                    "response_masks_length": len(response_masks)
                }
            }
        else:
            return {
                "idx": env_idx,
                "trajectory": trajectory,
                "total_reward": total_reward,
                "chat_completions": chat_completions
            }
    
    def _inject_workflow_context(self, current_agent_id: str, phase_responses: Dict[str, str], agents: Dict[str, BaseAgent]):
        context_parts = []
        
        for conn in self.connections:
            if conn.to_agent == current_agent_id and conn.from_agent in phase_responses:
                previous_response = phase_responses[conn.from_agent]
                context_parts.append(f"Input from {conn.from_agent.upper()}: {previous_response}")
        
        if context_parts:
            chain_context = "\n\n".join(context_parts)
            current_agent = agents[current_agent_id]
            if hasattr(current_agent, 'multi_agent_context'):
                current_agent.multi_agent_context["chain_context"] = chain_context
    
    async def _synchronize_agent_histories_with_final_decision(self, step_idx: int, observation: Any, final_action: Action, phase_responses: Dict[str, str], agents: Dict[str, BaseAgent]):
        """
        Synchronizes all agents to follow the same trajectory based on the final agent's decisions.
        
        Ensures consistency between agents during updates.
        Only manipulating state that won't break agent assumptions
        """
        final_agent_id = self.phases[-1].agent_ids[0]
        final_agent = agents[final_agent_id]
        
        final_agent_messages = final_agent.messages
        if len(final_agent_messages) >= 2 and final_agent_messages[-1]["role"] == "assistant":
            final_reasoning = final_agent_messages[-1]["content"]
            
            for agent_id, agent in agents.items():
                if agent_id != final_agent_id:

                    if len(agent.messages) >= 2 and agent.messages[-1]["role"] == "assistant":
                        old_response = agent.messages[-1]["content"]
                        agent.messages[-1] = {
                            "role": "assistant", 
                            "content": final_reasoning
                        }
                    else:
                        agent.messages.append({
                            "role": "assistant", 
                            "content": final_reasoning
                        })
                    
                    old_step = agent.step
                    agent.step = final_agent.step
            
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
        
        # Wake up the rollout engine before starting trajectory generation
        self.role_engines[list(self.role_engines.keys())[0]].rollout_engine.wake_up()
                
        async def launch_workflow_trajectory(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                print(f"\n{'='*60}")
                print(f"DEBUG: Starting workflow trajectory for env_idx={env_idx}")
                print(f"  - Application ID: {application_id}")
                print(f"{'='*60}\n")
                
                result = await self.run_workflow_trajectory_async(
                    env_idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs
                )
                
                print(f"\n{'='*60}")
                print(f"DEBUG: Completed workflow trajectory for env_idx={env_idx}")
                print(f"  - Application ID: {application_id}")
                print(f"  - Timestamp: {time.time()}")
                print(f"{'='*60}\n")
                
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
                steps = traj.get("metrics", {}).get("workflow_steps", 0)
                bar = "█" * steps
                print(f"Traj {idx}: {bar} ({steps} turns)")
            print(f"{'='*60}\n")
        
        # Sleep the rollout engine after all trajectories are completed
        self.role_engines[list(self.role_engines.keys())[0]].rollout_engine.sleep()
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
                    batch_timeout = meta_info.get('batch_execution_timeout', self.trajectory_timeout) if meta_info else self.trajectory_timeout
                    results = fut.result(timeout=batch_timeout)
            else:
                results = loop.run_until_complete(_collect_batch())
        except Exception as e:
            raise RuntimeError(f"Mutli-Agent batch execution failed: {str(e)}") from e
        
        return self._format_results_for_training(results)
    
    def _format_results_for_training(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            raise RuntimeError("Mutli-Agent execution produced no results")
        
        formatted_results = []
        for i, token_result in enumerate(results):
            if not token_result:
                raise RuntimeError(f"Mutli-Agent result {i} is empty")
            
            required_fields = ["trajectory_reward", "chat_completions"]
            for field in required_fields:
                if field not in token_result:
                    raise ValueError(f"Missing required field '{field}' in trajectory result {i}")
            
            formatted_results.append({
                "workflow_type": self.workflow.workflow_id,
                "batch_idx": token_result.get("idx", 0),
                "agent_trajectories": {
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
                "data_source": token_result.get("data_source", "unknown"),
                "uid": token_result.get("uid", f"unknown_{i}"),
            })
        
        return formatted_results
        