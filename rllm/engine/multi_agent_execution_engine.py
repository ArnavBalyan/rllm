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
        
        total_reward = 0.0
        final_tokens = []
        final_masks = []
        completed_turns = 0
        
        # Agent-level response tokens and masks tracking
        agent_response_tokens = {}
        agent_response_masks = {}
        agent_response_text_logs = {}  # Simplified: empty text logs
        agent_matches = {}  # Track matches with final agent per step
        agent_response_token_len = {}  # Track accumulated tokens per agent
        temp_assistant_messages = {}  # Track assistant messages for pretty printing
        temp_env_messages = {}  # Track env messages for pretty printing
        from rllm.agents.utils import convert_messages_to_tokens_and_masks
        final_agent_id = self.phases[-1].agent_id
        engine = self.role_engines[final_agent_id]
        phase_prompt_tokens = {}
        
        for phase in self.phases:
            agent_response_tokens[phase.agent_id] = []
            agent_response_masks[phase.agent_id] = []
            agent_response_text_logs[phase.agent_id] = []
            agent_matches[phase.agent_id] = 0
            agent_response_token_len[phase.agent_id] = 0
            temp_assistant_messages[phase.agent_id] = []
            temp_env_messages[phase.agent_id] = []
            
            # Reset and initialize agent
            agent_id = phase.agent_id
            agent = agents[agent_id]
            phase_engine = self.role_engines[agent_id]
            agent.reset()
            # colorful_print(f"🔄 Reset {agent_id}", "yellow")  # Removed for performance
            
            agent.update_from_env(observation, 0.0, False, info)
            
            initial_messages = agent.chat_completions
            # if agent_id == final_agent_id:
            #     print(f"Printing the initial messages for agent {agent_id}: ", initial_messages)
            
            phase_prompt_tokens[agent_id], _ = convert_messages_to_tokens_and_masks(
                initial_messages,
                tokenizer=phase_engine.tokenizer,
                parser=phase_engine.chat_parser,
                contains_first_msg=True,
                contains_generation_msg=True
            )
            
            temp_assistant_messages[agent_id].append(initial_messages.copy())
        
        termination_reason = None
        response_token_len = 0

        for step_idx in range(self.max_steps):
            
            phase_responses = {}
            phase_actions = {} 
            phase_upstream_contexts = {}
            final_action = None
            
            for phase_idx, phase in enumerate(self.phases):
                agent_id = phase.agent_id
                agent = agents[agent_id]
                engine = self.role_engines[agent_id]
                
                agent.prepare_new_step()
                
                upstream_messages = self._get_upstream_phase_response(agent_id, phase_responses)
                phase_upstream_contexts[agent_id] = upstream_messages
                
                if upstream_messages:
                    for msg in upstream_messages:
                        upstream_agent_id = msg["agent_id"]
                        upstream_response = phase_responses[upstream_agent_id]
                        agent.add_upstream_context(upstream_agent_id, upstream_response)
                
                prompt_msgs = agent.chat_completions.copy()
                max_tokens = engine.max_response_length - response_token_len
        
                replay_data = kwargs.get("meta_info", {}).get("replay_data")
                if replay_data and env_idx < len(replay_data):
                    saved_chat = replay_data[env_idx]["chat_completions"]
                    asst_index = 2 + step_idx * 2
                    if asst_index < len(saved_chat) and saved_chat[asst_index]["role"] == "assistant":
                        response = saved_chat[asst_index]["content"]
                        if step_idx == 0 and env_idx == 0:
                            print(f"🔁 MULTI REPLAY: Using saved responses for agent {agent_id}")
                    else:
                        response = ""
                else:
                    response = ""
                
                action = agent.complete_step_with_model_response(response)
                action_str = action.action 
                
                phase_responses[agent_id] = response
                phase_actions[agent_id] = action_str
                final_action = action_str
            
            final_action_str = phase_actions[final_agent_id]
                
            for agent_id in phase_actions:
                if agent_id != final_agent_id:
                    agent_action_str = phase_actions[agent_id]
                    if agent_action_str == final_action_str:
                        agent_matches[agent_id] += 1
            

            observation, reward, done, info = await loop.run_in_executor(
                None, env.step, final_action
            )
            total_reward = reward
            
            for agent_id, agent in agents.items():
                is_final_agent = (agent_id == final_agent_id)
                agent.update_from_env(
                    observation=observation,
                    reward=reward,
                    done=done,
                    info=info,
                    is_final_agent=is_final_agent,
                )
            
            any_agent_truncated = False
            for phase_idx, phase in enumerate(self.phases):
                agent_id = phase.agent_id
                agent = agents[agent_id]
                agent_tokens, agent_masks, agent_truncated, _ = self._process_agent_tokenization(
                    agent_id, agent, agents, mode, step_idx, phase_idx, temp_assistant_messages[agent_id], temp_env_messages[agent_id], response_token_len
                )
                
                agent_response_token_len[agent_id] += len(agent_tokens)
                response_token_len += len(agent_tokens)
                
                agent_response_tokens[agent_id].extend(agent_tokens)
                agent_response_masks[agent_id].extend(agent_masks)

                if agent_truncated:
                    any_agent_truncated = True
            
            if any_agent_truncated:
                total_reward = 0.0
                termination_reason = "TRUNCATION"
                
            if termination_reason == "TRUNCATION":
                break
                
            if done:
                break
            completed_turns = completed_turns + 1
        
        if mode == "Token":
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
                        agent_reward = total_reward
                        # if agent_id == final_agent_id:
                        #     agent_reward = total_reward
                        # else:
                        #     agent_reward = agent_matches[agent_id] / completed_turns
                    else:
                        agent_reward = total_reward
                    
                    phase_data[agent_id] = {
                        "response_tokens": torch.tensor(agent_response_tokens[agent_id], dtype=torch.long),
                        "response_masks": torch.tensor(agent_response_masks[agent_id], dtype=torch.long),
                        "prompt_tokens": torch.tensor(phase_prompt_tokens[agent_id], dtype=torch.long),  # Agent-specific initial context
                        "trajectory_reward": agent_reward,
                        "phase_id": phase.phase_id,
                        "agent_role": phase.agent_id,
                        "chat_completions": agents[agent_id].chat_completions,
                        "response_text_log": agent_response_text_logs[agent_id]
                    }
            
            all_response_tokens = []
            all_response_masks = []
            for phase in self.phases:
                agent_id = phase.agent_id
                if agent_id in agent_response_tokens:
                    all_response_tokens.extend(agent_response_tokens[agent_id])
                    all_response_masks.extend(agent_response_masks[agent_id])
            
            # Print trajectory summary with final token counts
            bar = "█" * (completed_turns)
            token_info = ""
            agent_tokens = []
            for agent_id in agents.keys():
                if agent_id in agent_response_token_len:
                    response_tokens = agent_response_token_len[agent_id]
                    prompt_tokens = len(phase_prompt_tokens[agent_id])
                    total_tokens = response_tokens + prompt_tokens
                    agent_tokens.append(f"{agent_id}:{total_tokens}t")
            token_info = f" [{', '.join(agent_tokens)}]"
            print(f"Trajectory {env_idx} Complete: {bar} ({completed_turns} turns){token_info}")
            
            # Stream collected tokenization messages for this multi-agent trajectory
            import json
            import os
            tokenization_log_entry = {
                "traj_id": env_idx,
                "application_id": application_id,
                "agent_tokenization_messages": {
                    agent_id: {
                        "temp_assistant_messages": temp_assistant_messages[agent_id],
                        "temp_env_messages": temp_env_messages[agent_id]
                    } for agent_id in temp_assistant_messages
                },
                "trajectory_reward": total_reward,
                "total_steps": completed_turns,
                "phases_executed": len(self.phases),
                "timestamp": time.time()
            }
            
            # Create logs directory if it doesn't exist
            # log_dir = "/workspace/model_response_logs"
            # os.makedirs(log_dir, exist_ok=True)
            # log_file = os.path.join(log_dir, "multi_agent_tokenization_messages.jsonl")
            
            # # Append to JSONL file
            # with open(log_file, "a") as f:
            #     f.write(json.dumps(tokenization_log_entry) + "\n")

            return {
                "idx": env_idx,
                "trajectory_reward": total_reward,
                "prompt_tokens": torch.tensor(phase_prompt_tokens[final_agent_id], dtype=torch.long),     # Final agent's initial context (like single-agent)
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
                    "prompt_tokens_length": len(phase_prompt_tokens[final_agent_id]),
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
                "chat_completions": agents[self.phases[-1].agent_id].chat_completions
            }
    
    def _get_upstream_phase_response(self, current_agent_id: str, phase_responses: Dict[str, str]) -> List[Dict[str, str]]:
        """Get upstream phase responses as a list of agent messages for injection into prompt"""
        upstream_messages = []
        
        for conn in self.connections:
            if conn.to_agent == current_agent_id and conn.from_agent in phase_responses:
                previous_response = phase_responses[conn.from_agent]
                upstream_messages.append({
                    "agent_id": conn.from_agent,
                    "content": previous_response  # Raw response, let agent format it
                })
        
        return upstream_messages
    
    def _process_agent_tokenization(self, agent_id: str, agent: BaseAgent, agents: Dict[str, BaseAgent], mode: str, step_idx: int = 0, phase_idx: int = 0, temp_assistant_messages: List = None, temp_env_messages: List = None, current_response_token_len: int = 0) -> Tuple[List[int], List[int], bool, List[dict]]:
        """
        Process tokenization for a specific agent and return tokens, masks, truncation status, and text logs.
        
        Args:
            agent_id: ID of the agent to process
            agent: The agent instance
            agents: Dictionary of all agents
            mode: Execution mode ("Token" or other)
            step_idx: Current step index for logging
            
        Returns:
            Tuple of (tokens, masks, is_truncated, text_logs)
        """
        from rllm.agents.utils import get_recent_assistant_messages_list, convert_messages_to_tokens_and_masks
        
        engine = self.role_engines[agent_id]
        chat_completions_messages = agent.chat_completions.copy()
        # Get recent assistant messages based on agent's position in chain (phase_idx + 1)
        agent_position = phase_idx + 1
        assistant_messages, env_messages = get_recent_assistant_messages_list(chat_completions_messages, agent_position)
        
        # Track messages for pretty printing (matching single agent pattern)
        if temp_assistant_messages is not None and assistant_messages:
            temp_assistant_messages.append("Custom Log representing message break between steps")
            temp_assistant_messages.append(assistant_messages)
        if temp_env_messages is not None and env_messages:
            temp_env_messages.extend(env_messages)
        assert assistant_messages or mode != "Token", f"Assistant messages is empty for agent {agent_id} when accumulating token trajectories which should be conversations. This should not happen."
        assert env_messages is not None or mode != "Token", f"Environment messages is none for agent {agent_id} when accumulating token trajectories which should be conversations. This should not happen."
        
        assistant_msg_tokens, assistant_msg_masks = [], []
        env_msg_tokens, env_msg_masks = [], []
        
        if assistant_messages:
            assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
                assistant_messages, 
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
        
        combined_tokens = assistant_msg_tokens + env_msg_tokens
        combined_masks = assistant_msg_masks + env_msg_masks
        
        # DIAGNOSTIC: Check mask composition
        if env_msg_masks:
            print(f"🔍 MULTI[{agent_id},step={step_idx}]: asst_tokens={len(assistant_msg_tokens)} env_tokens={len(env_msg_tokens)} env_mask_mean={sum(env_msg_masks)/len(env_msg_masks):.3f} | combined_mask_mean={sum(combined_masks)/len(combined_masks):.6f}")
        
        text_logs = []
        
        updated_response_token_len = current_response_token_len + len(combined_tokens)
        if updated_response_token_len >= engine.max_response_length:
            truncation_length = engine.max_response_length - updated_response_token_len
            if truncation_length < 0:
                truncated_response_tokens = combined_tokens[:truncation_length]
                truncated_response_masks = combined_masks[:truncation_length]
            else:
                truncated_response_tokens = combined_tokens
                truncated_response_masks = combined_masks
            
            cur_step = agent.get_current_state()
            if updated_response_token_len - len(env_msg_tokens) > engine.max_response_length:
                cur_step.reward = 0.0
            cur_step.done = True
            return truncated_response_tokens, truncated_response_masks, True, text_logs
        
        return combined_tokens, combined_masks, False, text_logs
                
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
        
        # Simple trajectory completion summary (token details shown per step above)
        if completed_trajectories:
            print(f"\n{'='*60}")
            print(f"COMPLETED {len(completed_trajectories)} TRAJECTORIES")
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
    