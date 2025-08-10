import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from typing import Dict, List, Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.multi_agent_execution_engine import (
    MultiAgentExecutionEngine,
    BaseWorkflow,
    ChainOfExpertsWorkflow,
    MixtureOfExpertsWorkflow,
    DebateWorkflow,
    AgentConfig,
    AgentRole
)
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)


class MultiAgentPPOTrainer(AgentPPOTrainer):
    """Enhanced PPO trainer for multi-agent workflows"""
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: Dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
        workflow: BaseWorkflow = None,
        multi_agent_config: Dict[str, Any] = None,
    ):
        # Initialize base trainer
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            env_class=env_class,
            agent_class=agent_class,
            env_args=env_args,
            agent_args=agent_args,
        )
        
        # Multi-agent specific configuration
        self.workflow = workflow
        self.multi_agent_config = multi_agent_config or {}
        self.multi_agent_engine = None
        
        # Training modes
        self.training_mode = self.multi_agent_config.get("training_mode", "unified")  # "unified", "individual", "joint"
        self.reward_aggregation = self.multi_agent_config.get("reward_aggregation", "final_agent")  # "final_agent", "average", "weighted"
    
    def init_workers(self):
        """Initialize workers including multi-agent execution engine"""
        super().init_workers()
        
        if self.workflow is not None:
            # Initialize multi-agent execution engine
            if self.hybrid_engine:
                agent_rollout_wg = self.actor_rollout_wg
            else:
                agent_rollout_wg = self.rollout_wg

            if self.config.actor_rollout_ref.rollout.mode == "async":
                rollout_engine = self.async_rollout_manager
            else:
                rollout_engine = agent_rollout_wg

            self.multi_agent_engine = MultiAgentExecutionEngine(
                workflow=self.workflow,
                env_class=self.env_class,
                env_args=self.env_args,
                engine_name="verl",
                tokenizer=self.tokenizer,
                rollout_engine=rollout_engine,
                n_parallel_workflows=self.config.actor_rollout_ref.rollout.n,
                max_response_length=self.config.data.max_response_length,
                max_prompt_length=self.config.data.max_prompt_length,
                trajectory_timeout=self.config.agent.trajectory_timeout,
                **self.config.agent.get("engine_args", {}),
            )
    
    def init_envs_and_agents(self, batch):
        """Initialize environments and agents for multi-agent training"""
        if self.workflow is None:
            # Fall back to single-agent mode
            return super().init_envs_and_agents(batch)
        
        env_args = batch.non_tensor_batch["extra_info"].tolist()
        
        # Create environments for multi-agent workflows
        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **self.env_args})

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Update multi-agent engine with environments
        self.multi_agent_engine.update_envs_and_agents(envs)
        return envs
    
    def generate_multi_agent_trajectory(self, timing_raw=None, meta_info=None):
        """Generate multi-agent trajectories"""
        if timing_raw is None:
            timing_raw = {}
        
        with _timer("collect_multi_agent_trajectory", timing_raw):
            # Create tasks from environments
            tasks = []
            for i, env in enumerate(self.multi_agent_engine.envs):
                task = {
                    "idx": i,
                    "seed": meta_info.get("seed", 0) + i if meta_info else i
                }
                # Add environment-specific task data if available
                if hasattr(env, 'task_data'):
                    task.update(env.task_data)
                tasks.append(task)
            
            # Execute multi-agent workflows
            if self.config.agent.async_engine:
                workflow_results = asyncio.run(
                    self.multi_agent_engine.execute_multi_agent_workflows(tasks)
                )
            else:
                # Synchronous execution (simplified)
                workflow_results = []
                for task in tasks:
                    result = asyncio.run(
                        self.multi_agent_engine.execute_multi_agent_trajectory(
                            workflow_idx=task["idx"],
                            application_id=str(uuid.uuid4()),
                            seed=task.get("seed", 0)
                        )
                    )
                    workflow_results.append(result)
        
        with _timer("transform_multi_agent_trajectory", timing_raw):
            # Transform multi-agent results into training format
            final_gen_batch_output, metrics = self._transform_multi_agent_trajectories(workflow_results)
        
        return final_gen_batch_output, metrics
    
    def _transform_multi_agent_trajectories(self, workflow_results: List[Dict[str, Any]]):
        """Transform multi-agent workflow results into DataProto format"""
        from verl.utils.torch_functional import pad_sequence_to_length
        
        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        metrics = {}
        
        for workflow_result in workflow_results:
            # Extract trajectory data based on training mode
            if self.training_mode == "unified":
                # Create a unified trajectory from all agents
                unified_trajectory = self._create_unified_trajectory(workflow_result)
                prompt_tokens, response_tokens, response_masks, score = unified_trajectory
                
            elif self.training_mode == "final_agent":
                # Use only the final agent's trajectory
                final_trajectory = self._extract_final_agent_trajectory(workflow_result)
                prompt_tokens, response_tokens, response_masks, score = final_trajectory
                
            elif self.training_mode == "individual":
                # Train each agent separately (would need multiple passes)
                individual_trajectories = self._extract_individual_trajectories(workflow_result)
                # For now, use the first trajectory
                prompt_tokens, response_tokens, response_masks, score = individual_trajectories[0]
                
            else:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
            
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(response_masks)
            traj_scores.append(score)
            
            # Create chat completion format
            chat_completion = self._create_chat_completion(workflow_result)
            chat_completions.append(chat_completion)
            
            # Extract metrics
            workflow_metrics = workflow_result.get("metrics", {})
            traj_metrics.append(workflow_metrics)
        
        # Process metrics
        if traj_metrics:
            # Flatten traj_metrics into a dict of lists
            traj_metrics = {k: [d.get(k, 0) for d in traj_metrics] for k in traj_metrics[0]}
            # Aggregate metrics
            for k, v_list in traj_metrics.items():
                v_list = [v for v in v_list if v is not None and v >= 0]
                if v_list:
                    v_list = np.array(v_list)
                    metrics.update({
                        f"multi_agent/{k}_mean": v_list.mean(),
                        f"multi_agent/{k}_min": v_list.min(),
                        f"multi_agent/{k}_max": v_list.max(),
                    })
        
        # Add multi-agent specific metrics
        metrics.update({
            "multi_agent/workflow_type": self.workflow.workflow_id,
            "multi_agent/agent_count": len(self.workflow.agent_configs),
            "multi_agent/training_mode": self.training_mode,
        })
        
        # Save chat completions
        save_dir = os.path.join(self.config.trainer.default_local_dir, "multi_agent_completions")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")
        
        # Create batched tensors (same as single-agent trainer)
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        
        prompts_batch = pad_sequence_to_length(
            prompts_batch, self.config.data.max_prompt_length, 
            self.tokenizer.pad_token_id, left_pad=True
        )
        
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(
            response_batch, max_response_length, 
            self.tokenizer.pad_token_id, left_pad=False
        )
        
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        
        # Place rewards at last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)
        
        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
        
        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
        }
        
        return DataProto.from_dict(tensors=tensor_batch), metrics
    
    def _create_unified_trajectory(self, workflow_result: Dict[str, Any]):
        """Create a unified trajectory from multiple agent interactions"""
        agent_trajectories = workflow_result.get("agent_trajectories", {})
        
        # Combine all agent responses into a single conversation
        combined_prompt = ""
        combined_response = ""
        
        # Start with initial problem/context
        if "task" in workflow_result:
            task_data = workflow_result["task"]
            if isinstance(task_data, dict) and "problem" in task_data:
                combined_prompt = f"Problem: {task_data['problem']}\n\n"
        
        # Add each agent's contribution
        for agent_id, trajectory in agent_trajectories.items():
            if "prompt_tokens" in trajectory and "response_tokens" in trajectory:
                agent_prompt = self.tokenizer.decode(trajectory["prompt_tokens"])
                agent_response = self.tokenizer.decode(trajectory["response_tokens"])
                
                combined_prompt += f"Agent {agent_id} context:\n{agent_prompt}\n\n"
                combined_response += f"Agent {agent_id} response:\n{agent_response}\n\n"
        
        # Tokenize the combined conversation
        prompt_tokens = torch.tensor(
            self.tokenizer.encode(combined_prompt, add_special_tokens=False), 
            dtype=torch.long
        )
        response_tokens = torch.tensor(
            self.tokenizer.encode(combined_response, add_special_tokens=False), 
            dtype=torch.long
        )
        response_masks = torch.ones_like(response_tokens)
        
        # Aggregate scores
        score = self._aggregate_workflow_score(workflow_result)
        
        return prompt_tokens, response_tokens, response_masks, score
    
    def _extract_final_agent_trajectory(self, workflow_result: Dict[str, Any]):
        """Extract trajectory from the final agent in the workflow"""
        agent_trajectories = workflow_result.get("agent_trajectories", {})
        
        if not agent_trajectories:
            # Fallback to empty trajectory
            prompt_tokens = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            response_tokens = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            response_masks = torch.ones_like(response_tokens)
            return prompt_tokens, response_tokens, response_masks, 0.0
        
        # Get the last agent's trajectory (assuming workflow order matters)
        final_agent_id = list(agent_trajectories.keys())[-1]
        final_trajectory = agent_trajectories[final_agent_id]
        
        prompt_tokens = final_trajectory.get("prompt_tokens", torch.tensor([self.tokenizer.eos_token_id]))
        response_tokens = final_trajectory.get("response_tokens", torch.tensor([self.tokenizer.eos_token_id]))
        response_masks = final_trajectory.get("response_masks", torch.ones_like(response_tokens))
        
        score = self._aggregate_workflow_score(workflow_result)
        
        return prompt_tokens, response_tokens, response_masks, score
    
    def _extract_individual_trajectories(self, workflow_result: Dict[str, Any]):
        """Extract individual trajectories for separate training"""
        agent_trajectories = workflow_result.get("agent_trajectories", {})
        individual_trajectories = []
        
        workflow_score = self._aggregate_workflow_score(workflow_result)
        
        for agent_id, trajectory in agent_trajectories.items():
            prompt_tokens = trajectory.get("prompt_tokens", torch.tensor([self.tokenizer.eos_token_id]))
            response_tokens = trajectory.get("response_tokens", torch.tensor([self.tokenizer.eos_token_id]))
            response_masks = trajectory.get("response_masks", torch.ones_like(response_tokens))
            
            individual_trajectories.append((prompt_tokens, response_tokens, response_masks, workflow_score))
        
        return individual_trajectories
    
    def _aggregate_workflow_score(self, workflow_result: Dict[str, Any]) -> float:
        """Aggregate scores from multi-agent workflow"""
        if self.reward_aggregation == "final_agent":
            # Use score from final agent
            agent_trajectories = workflow_result.get("agent_trajectories", {})
            if agent_trajectories:
                final_agent_id = list(agent_trajectories.keys())[-1]
                return agent_trajectories[final_agent_id].get("trajectory_reward", 0.0)
            return 0.0
        
        elif self.reward_aggregation == "average":
            # Average scores across all agents
            agent_trajectories = workflow_result.get("agent_trajectories", {})
            if agent_trajectories:
                scores = [traj.get("trajectory_reward", 0.0) for traj in agent_trajectories.values()]
                return sum(scores) / len(scores)
            return 0.0
        
        elif self.reward_aggregation == "weighted":
            # Weighted average based on agent roles (could be configured)
            agent_trajectories = workflow_result.get("agent_trajectories", {})
            if agent_trajectories:
                # Simple implementation: equal weights
                return self._aggregate_workflow_score(workflow_result.copy())  # Fall back to average
            return 0.0
        
        else:
            return 0.0
    
    def _create_chat_completion(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create chat completion record for logging"""
        return {
            "workflow_id": workflow_result.get("workflow_id", "unknown"),
            "workflow_type": self.workflow.workflow_id,
            "agent_count": len(workflow_result.get("agent_trajectories", {})),
            "total_time": workflow_result.get("total_time", 0.0),
            "final_outputs": workflow_result.get("final_outputs", {}),
            "training_mode": self.training_mode,
            "reward_aggregation": self.reward_aggregation,
        }
    
    def fit_multi_agent(self):
        """Enhanced training loop for multi-agent workflows"""
        if self.workflow is None:
            # Fall back to single-agent training
            return self.fit_agent()
        
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=f"{self.config.trainer.experiment_name}_multi_agent",
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # Load checkpoint before doing anything
        self._load_checkpoint()
        
        # Perform validation before training
        import time
        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_multi_agent()
            pprint(f"Initial multi-agent validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate multi-agent system: {time.time() - start_time}")
        
        # Start from step 1
        self.global_steps += 1
        
        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"Multi-agent epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                
                metrics = {}
                timing_raw = {}
                
                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "multi_agent_rollout": True,
                    "workflow_type": self.workflow.workflow_id,
                }
                
                with _timer("multi_agent_step", timing_raw):
                    self.init_envs_and_agents(batch)
                    
                    # Generate multi-agent trajectories
                    final_gen_batch_output, generate_metrics = self.generate_multi_agent_trajectory(
                        timing_raw=timing_raw, 
                        meta_info=batch.meta_info
                    )
                    batch = batch.union(final_gen_batch_output)
                    metrics.update(generate_metrics)
                    
                    # Continue with standard PPO training pipeline
                    # (compute values, advantages, etc.)
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    with _timer("adv", timing_raw):
                        # Compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]
                        
                        # Apply rejection sampling and other processing as in base trainer
                        # (simplified here for brevity)
                        
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        
                        # Compute advantages
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )
                
                batch = self._pad_dataproto_to_world_size(batch=batch)
                self._balance_batch(batch, metrics=metrics)
                
                # Update critic and actor
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                
                # Validation and checkpointing
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate_multi_agent()
                    metrics.update(val_metrics)
                
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                
                # Collect and log metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1
                
                if self.global_steps >= self.total_training_steps:
                    # Perform final validation
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_multi_agent()
                        pprint(f"Final multi-agent validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _validate_multi_agent(self):
        """Validation for multi-agent workflows"""
        if self.workflow is None:
            return self._validate_agent()
        
        # Implement multi-agent specific validation
        # This is a simplified version - could be more sophisticated
        return self._validate_agent()  # For now, fall back to single-agent validation


# Factory functions for creating common multi-agent configurations
def create_chain_of_experts_trainer(
    base_trainer: AgentPPOTrainer,
    agent_configs: List[AgentConfig],
    **kwargs
) -> MultiAgentPPOTrainer:
    """Create a trainer for chain of experts workflow"""
    workflow = ChainOfExpertsWorkflow(agent_configs)
    return MultiAgentPPOTrainer(
        config=base_trainer.config,
        tokenizer=base_trainer.tokenizer,
        role_worker_mapping=base_trainer.role_worker_mapping,
        resource_pool_manager=base_trainer.resource_pool_manager,
        ray_worker_group_cls=base_trainer.ray_worker_group_cls,
        reward_fn=base_trainer.reward_fn,
        val_reward_fn=base_trainer.val_reward_fn,
        env_class=base_trainer.env_class,
        agent_class=base_trainer.agent_class,
        env_args=base_trainer.env_args,
        agent_args=base_trainer.agent_args,
        workflow=workflow,
        **kwargs
    )


def create_mixture_of_experts_trainer(
    base_trainer: AgentPPOTrainer,
    expert_configs: List[AgentConfig],
    aggregator_config: AgentConfig,
    **kwargs
) -> MultiAgentPPOTrainer:
    """Create a trainer for mixture of experts workflow"""
    workflow = MixtureOfExpertsWorkflow(expert_configs, aggregator_config)
    return MultiAgentPPOTrainer(
        config=base_trainer.config,
        tokenizer=base_trainer.tokenizer,
        role_worker_mapping=base_trainer.role_worker_mapping,
        resource_pool_manager=base_trainer.resource_pool_manager,
        ray_worker_group_cls=base_trainer.ray_worker_group_cls,
        reward_fn=base_trainer.reward_fn,
        val_reward_fn=base_trainer.val_reward_fn,
        env_class=base_trainer.env_class,
        agent_class=base_trainer.agent_class,
        env_args=base_trainer.env_args,
        agent_args=base_trainer.agent_args,
        workflow=workflow,
        **kwargs
    )


def create_debate_trainer(
    base_trainer: AgentPPOTrainer,
    debater_configs: List[AgentConfig],
    judge_config: AgentConfig,
    max_rounds: int = 3,
    **kwargs
) -> MultiAgentPPOTrainer:
    """Create a trainer for debate workflow"""
    workflow = DebateWorkflow(debater_configs, judge_config, max_rounds)
    return MultiAgentPPOTrainer(
        config=base_trainer.config,
        tokenizer=base_trainer.tokenizer,
        role_worker_mapping=base_trainer.role_worker_mapping,
        resource_pool_manager=base_trainer.resource_pool_manager,
        ray_worker_group_cls=base_trainer.ray_worker_group_cls,
        reward_fn=base_trainer.reward_fn,
        val_reward_fn=base_trainer.val_reward_fn,
        env_class=base_trainer.env_class,
        agent_class=base_trainer.agent_class,
        env_args=base_trainer.env_args,
        agent_args=base_trainer.agent_args,
        workflow=workflow,
        multi_agent_config={"max_rounds": max_rounds, **kwargs}
    ) 