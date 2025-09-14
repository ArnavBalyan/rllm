import asyncio
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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
from verl.single_controller.ray import RayClassWithInitArgs
from rllm.misc import colorful_print


class MultiAgentPPOTrainer(AgentPPOTrainer):
    """

    Multi-Agent PPO Trainer    
    Note: "phase" refers to one agent's execution in the chain,
          "step" refers to one conversation turn/action (preserved from base rLLM)
    """
    
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
        
        self.workflow = workflow
        self.multi_agent_config = multi_agent_config or {}
        self.multi_agent_engine = None
        
        self.training_mode = self.config.multi_agent.training_mode
        self.reward_aggregation = self.config.multi_agent.reward_aggregation
        self.agent_rollout_engines = {}
    
    def _init_multiple_agent_workers(self):
        """Initialize separate ActorRolloutRefWorker instances for each agent"""
        from verl.single_controller.ray.base import create_colocated_worker_cls
        
        # Arrays to store pools and worker groups outside the loop
        self.agent_resource_pools = []
        self.agent_worker_groups_array = []
        
        for agent_config in self.workflow.agent_configs_list:
            agent_id = agent_config.agent_id
            
            # Create separate resource pool for this agent
            agent_resource_pool_spec = {f"{agent_id}_pool": [2]}  # 2 GPUs per agent
            agent_mapping = {Role.ActorRollout: f"{agent_id}_pool"}
            agent_rpm = ResourcePoolManager(agent_resource_pool_spec, agent_mapping)
            agent_rpm.create_resource_pool()
            
            # Store resource pool in array
            agent_resource_pool = agent_rpm.get_resource_pool(Role.ActorRollout)
            self.agent_resource_pools.append(agent_resource_pool)
            
            # Create agent-specific config with different model path
            agent_config_dict = self.config.copy()
            if agent_config.model_path:
                agent_config_dict.actor_rollout_ref.model.path = agent_config.model_path
            
            # Create RayClassWithInitArgs (following RayPPOTrainer pattern)
            agent_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=agent_config_dict.actor_rollout_ref,
                role="actor_rollout",
            )
            
            # Create worker group (following lines 841-845 from RayPPOTrainer)
            resource_pool_to_cls = {agent_resource_pool: {"actor_rollout": agent_rollout_cls}}
            
            for resource_pool, class_dict in resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                
                # Get the actor_rollout worker group and init model
                agent_wg = spawn_wg['actor_rollout']
                agent_wg.init_model()  # Each agent loads its own model
                
                # Create AsyncLLMServerManager for proper multi-agent isolation
                from verl.workers.rollout.async_server import AsyncLLMServerManager
                agent_async_manager = AsyncLLMServerManager(
                    config=agent_config_dict.actor_rollout_ref,
                    worker_group=agent_wg,
                    scheduler_kwargs={"agent_id": agent_id} 
                )
                self.agent_rollout_engines[agent_id] = agent_async_manager
                self.agent_worker_groups_array.append(agent_wg)  # Store in array
                print(f"Created AsyncLLMServerManager for agent '{agent_id}' with model: {agent_config.model_path or 'default'}")
        
        if self.use_critic and self.agent_resource_pools:
            first_agent_pool = self.agent_resource_pools[0]
            
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.critic
            )
            
            resource_pool_to_cls = {first_agent_pool: {"critic": critic_cls}}
            
            for resource_pool, class_dict in resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                
                self.critic_wg = spawn_wg['critic']
                self.critic_wg.init_model()
    
    def init_workers(self):
        self._init_multiple_agent_workers()
        
        if self.workflow is not None:
            rollout_engine = self.agent_rollout_engines

            self.multi_agent_engine = MultiAgentExecutionEngine(
                workflow=self.workflow,
                env_class=self.env_class,
                env_args=self.env_args,
                engine_name="verl",
                tokenizer=self.tokenizer,
                rollout_engine=rollout_engine,
                config=self.config,
                max_response_length=self.config.data.max_response_length,
                max_prompt_length=self.config.data.max_prompt_length,
                trajectory_timeout=self.config.agent.trajectory_timeout,
                **self.config.agent.get("engine_args", {}),
            )
            
            # For compatibility with base class, set these attributes to the final agent's engine
            # but actual training will use agent_rollout_engines for independent training
            final_agent_id = self.workflow.agent_configs_list[-1].agent_id
            final_agent_engine = self.multi_agent_engine.role_engines[final_agent_id]
            
            if self.hybrid_engine:
                self.actor_rollout_wg = final_agent_engine.rollout_engine
            else:
                self.rollout_wg = final_agent_engine.rollout_engine
    
    def init_envs_and_agents(self, batch):
        
        env_args = batch.non_tensor_batch["extra_info"].tolist()
        
        envs = []
        for i, env_arg in enumerate(env_args):
            if isinstance(env_arg, str):
                env_arg = json.loads(env_arg)
            env = self.env_class.from_dict({**env_arg, **self.env_args})
            envs.append(env)

        self.multi_agent_engine.update_envs_and_agents(envs)
        return envs
    
    def generate_chain_of_experts_trajectories(self, timing_raw=None, meta_info=None):
        """Generate Chain of Experts trajectories by processing batch through phases"""
        """Returns a list of agent results"""

        with _timer("collect_chain_of_experts_trajectories", timing_raw):
            workflow_results = self.multi_agent_engine.execute_chain_of_experts_batch(
                timing_raw=timing_raw,
                meta_info=meta_info
            )
        
        agent_batches = {}
        metrics = {}
        with _timer("transform_chain_of_experts_trajectories", timing_raw):
            agent_ids = [cfg.agent_id for cfg in self.workflow.agent_configs_list]
            
            for agent_id in agent_ids:
                agent_batch = self._transform_agent_trajectories_for_agent(
                    agent_id, workflow_results
                )
                agent_batches[agent_id] = agent_batch


        traj_metrics = []
        for traj in workflow_results:
            traj_metrics.append(traj["metrics"])

        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        return agent_batches, metrics
    
    def _transform_agent_trajectories_for_agent(self, agent_id: str, workflow_results: List[Dict[str, Any]]):
        """Transform trajectories for a specific agent into tokenized DataProto format."""
        from verl.utils.torch_functional import pad_sequence_to_length        

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        
        for workflow_result in workflow_results:
            traj = workflow_result["phase_data"][agent_id]

            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            
            
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])

        save_dir = os.path.join(self.config.trainer.default_local_dir, f"chat_completions/{agent_id}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # Pad and create tensors (same logic as base class)
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        
        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        
        # Place all rewards to last response token
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
        
        return DataProto.from_dict(tensors=tensor_batch)
    
    def fit_multi_agent(self):
        """Enhanced training loop for Multi-Agent workflows"""
        if self.workflow is None:
            return self.fit_agent()
        
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=f"{self.config.trainer.experiment_name}_chain_of_experts",
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        self._load_checkpoint()
        
        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_multi_agent()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate Chain of Experts system: {time.time() - start_time}")
        
        self.global_steps += 1
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch_global: DataProto = DataProto.from_single_dict(batch_dict)
                batch_global.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_global.batch))], dtype=object)
                batch_global = batch_global.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                print("batch_global dict starting update")
                
                metrics_global = {}
                timing_raw = {}
                
                batch_global.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch_global.meta_info = {
                    "chain_of_experts_rollout": True,
                    "workflow_type": self.workflow.workflow_id,
                    "temperature": self.config.actor_rollout_ref.rollout.temperature,
                }
                print("Batch dict 364")
                
                with _timer("chain_of_experts_batch", timing_raw):
                    print("Batch dict 367")
                    
                    self.init_envs_and_agents(batch_global)
                    print("Batch dict 370")

                    # at this point the system ahs multiple boards, and coordinator agent assigned
                    # to each board with some basic metadata and initialized empty lists etc.
                    
                    final_gen_batch_output, generate_metrics = self.generate_chain_of_experts_trajectories(
                        timing_raw=timing_raw, 
                        meta_info=batch_global.meta_info
                    )
                    print("CHAIN OF EXPERTS COMPLETE FOR THE CURRENT STEP")

                    for agent_id, agent_batch in final_gen_batch_output.items():
                        if agent_id not in metrics_global:
                            metrics_global[agent_id] = {}

                        print(f"Processing agent: {agent_id}")
                        
                        local_batch = final_gen_batch_output[agent_id]
                        
                        batch = type(batch_global)(
                            batch=batch_global.batch.clone(),
                            non_tensor_batch=deepcopy(batch_global.non_tensor_batch),
                            meta_info=deepcopy(batch_global.meta_info)
                        )
                        batch = batch.union(local_batch)
                        # generate_metrics is a single dict of trajectory metrics, not per-agent
                        metrics_global[agent_id].update(generate_metrics)
                    
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                raise Exception("This should not be called")
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)
                    
                        with _timer("adv", timing_raw):
                            if self.use_rm:
                                raise Exception("This should not be called")
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)
                        
                            if "token_level_scores" not in batch.batch:
                                reward_tensor = self.reward_fn(batch)
                                batch.batch["token_level_scores"] = reward_tensor
                            else:
                                reward_tensor = batch.batch["token_level_scores"]
                                                
                            uids = batch.non_tensor_batch["uid"]
                            unique_uids = np.unique(uids)
                            valid_mask = torch.ones(len(uids), dtype=torch.bool)
                            solve_none = 0
                            solve_all = 0
                            solve_none = 0
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                uid_rewards = reward_tensor[uid_mask].sum(-1) 
                                if (uid_rewards <= 0).all():
                                    valid_mask[uid_mask] = False
                                    solve_none += 1
                                elif (uid_rewards >= 1).all():
                                    valid_mask[uid_mask] = False
                                    solve_all += 1

                            metrics_global[agent_id]["batch/solve_none"] = solve_none
                            metrics_global[agent_id]["batch/solve_all"] = solve_all
                            metrics_global[agent_id]["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all
                            
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            with _timer("old_log_prob", timing_raw):
                                agent_worker_group = self.agent_rollout_engines[agent_id].worker_group
                                old_log_prob = agent_worker_group.compute_log_prob(batch)
                                batch = batch.union(old_log_prob)

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                                clip_advantages=self.config.algorithm.clip_advantages,
                            )
                
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                        agent_metrics = {}
                        self._balance_batch(batch, metrics=agent_metrics)
                        metrics_global[agent_id].update(agent_metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics_global[agent_id].update(critic_output_metrics)
                        print("critic update complete")
                
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with _timer("update_actor", timing_raw):
                                agent_worker_group = self.agent_rollout_engines[agent_id].worker_group
                                actor_output = agent_worker_group.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics_global[agent_id].update(actor_output_metrics)
                        print("actor update complete")
                
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            print("validation started")
                            val_metrics: dict = self._validate_multi_agent()
                            print("validation complete")
                        metrics_global["validation"] = {}
                        metrics_global["validation"].update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    print("Checkpoint saved")
                # Collect and log metrics
                metrics_global.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics_global.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                logger.log(data=metrics_global, step=self.global_steps)
                self.global_steps += 1
                
                if self.global_steps >= self.total_training_steps:
                    if self.val_reward_fn is not None:
                        print("Validating the Mutli agent loop")
                        val_metrics = self._validate_multi_agent()
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _write_latest_audit(self, batch: DataProto, step_tag: str = "train"):
        import os
        from datetime import datetime
        sample_k = 300
        max_str = 2000

        # Select indices (front of batch for determinism)
        bs = batch.batch["prompts"].shape[0]
        if bs == 0:
            return
        idxs = list(range(min(sample_k, bs)))

        # Decode helpers
        def _decode_tokens(t):
            mask = t != self.tokenizer.pad_token_id
            return self.tokenizer.decode(t[mask])[:max_str]

        # Build records
        records = []
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        token_level_scores = batch.batch["token_level_scores"] if "token_level_scores" in batch.batch else None
        advantages = batch.batch["advantages"] if "advantages" in batch.batch else None
        old_log_probs = batch.batch["old_log_probs"] if "old_log_probs" in batch.batch else None
        response_mask = batch.batch["response_mask"] if "response_mask" in batch.batch else None
        uid_arr = batch.non_tensor_batch["uid"] if "uid" in batch.non_tensor_batch else [f"unknown_{i}" for i in range(bs)]

        for i in idxs:
            prompt_text = _decode_tokens(prompts[i]) if prompts is not None else ""
            resp_text = _decode_tokens(responses[i]) if responses is not None else ""
            # last valid response token position
            prompt_len = prompts[i].ne(self.tokenizer.pad_token_id).sum().item() if prompts is not None else 0
            valid_resp = attention_mask[i, prompt_len:].sum().item() if attention_mask is not None else 0
            last_pos = int(valid_resp - 1) if valid_resp > 0 else -1

            rec = {
                "step": int(self.global_steps),
                "tag": step_tag,
                "idx": i,
                "uid": uid_arr[i] if isinstance(uid_arr, (list, tuple, np.ndarray)) else uid_arr,
                "prompt_text_tail": prompt_text[-max_str:],
                "response_text_tail": resp_text[-max_str:],
                "last_reward_pos": last_pos,
            }

            if token_level_scores is not None:
                try:
                    rec["reward_last"] = float(token_level_scores[i, last_pos].item()) if last_pos >= 0 else 0.0
                except Exception:
                    rec["reward_last"] = None
            if advantages is not None:
                try:
                    # if response_mask exists, average only unmasked tokens
                    if response_mask is not None:
                        m = response_mask[i].bool()
                        masked_adv = advantages[i][m]
                        rec["adv_mean_unmasked"] = float(masked_adv.mean().item()) if masked_adv.numel() else None
                    else:
                        rec["adv_mean"] = float(advantages[i].mean().item())
                except Exception:
                    rec["adv_mean_unmasked"] = None
            if old_log_probs is not None and last_pos >= 0:
                try:
                    rec["old_log_prob_last"] = float(old_log_probs[i, last_pos].item())
                    rec["old_prob_last"] = float(torch.exp(old_log_probs[i, last_pos]).item())
                except Exception:
                    rec["old_log_prob_last"] = None

            records.append(rec)

        out_path = "/home/ubuntu/rllm/latest.json"
        payload = {
            "time": datetime.utcnow().isoformat() + "Z",
            "global_step": int(self.global_steps),
            "records": records,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f)
    
    def _validate_multi_agent(self):
        if self.workflow is None:
            return self._validate_agent()
        
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            print(f"[DEBUG VAL] n_val_samples={n_val_samples} | batch_size_after_repeat={len(test_batch.batch)}")
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])
            
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "chain_of_experts_rollout": True,
                "workflow_type": self.workflow.workflow_id,
                "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
            }
            
            self.init_envs_and_agents(test_batch)
            
            test_output_gen_batch, _ = self.generate_chain_of_experts_trajectories(
                timing_raw={},  # Add empty dict for timing_raw
                meta_info=test_batch.meta_info
            )
            # Get final agent's batch for inference
            final_agent_id = self.workflow.agent_configs_list[-1].agent_id
            final_agent_batch = test_output_gen_batch[final_agent_id]
            test_batch = test_batch.union(final_agent_batch)
            reward_tensor = test_batch.batch["token_level_scores"]
            rewards_lst.append(reward_tensor.sum(-1).cpu().numpy())
            
            data_source_lst.extend(test_batch.non_tensor_batch["data_source"] if "data_source" in test_batch.non_tensor_batch else ["unknown"] * len(test_batch.batch))
            uid_lst.extend(test_batch.non_tensor_batch["uid"])
        
        all_rewards = np.concatenate(rewards_lst, axis=0)
        data_sources = np.array(data_source_lst)
        uid_tensor = np.array(uid_lst)
        
        unique_uids = np.unique(uid_tensor)
        solve_none = solve_all = solve_partial = 0
        for uid in unique_uids:
            uid_mask = uid_tensor == uid
            uid_rewards = all_rewards[uid_mask]
            if (uid_rewards <= 0).all():
                solve_none += 1
            elif (uid_rewards >= 1).all():
                solve_all += 1
            else:
                solve_partial += 1
        
        data_source_reward = {}
        data_source_uid_pass_rates = {}
        for i in range(all_rewards.shape[0]):
            data_source = data_sources[i]
            uid = uid_tensor[i]
            
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_uid_pass_rates[data_source] = {}
            data_source_reward[data_source].append(all_rewards[i])
            
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], all_rewards[i])
        
        val_metrics = {
            "chain_of_experts/val_reward_mean": np.mean(all_rewards),
            "chain_of_experts/val_reward_max": np.max(all_rewards),
            "chain_of_experts/val_reward_min": np.min(all_rewards),
            "chain_of_experts/val_reward_std": np.std(all_rewards),
            "chain_of_experts/solve_none": solve_none,
            "chain_of_experts/solve_all": solve_all,
            "chain_of_experts/solve_partial": solve_partial,
        }
        
        for data_source, rewards in data_source_reward.items():
            rewards_array = np.clip(np.array(rewards), 0, 1)
            val_metrics[f"chain_of_experts/val/test_score/{data_source}"] = np.mean(rewards_array)
            
            pass_k_lst = [pass_score >= 1 for pass_score in data_source_uid_pass_rates[data_source].values()]
            val_metrics[f"chain_of_experts/val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)
        
        return val_metrics
            


def create_chain_of_experts_trainer(
    config,
    tokenizer,
    role_worker_mapping,
    resource_pool_manager,
    agent_configs: List[AgentConfig],
    ray_worker_group_cls=RayWorkerGroup,
    reward_fn=None,
    val_reward_fn=None,
    env_class=None,
    agent_class=None,
    env_args=None,
    agent_args=None,
    multi_agent_config: Dict[str, Any] = None,
    **kwargs
) -> MultiAgentPPOTrainer:
    """
    Create a trainer for Chain of Experts workflow.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer instance
        role_worker_mapping: Mapping of roles to worker types
        resource_pool_manager: Resource pool manager
        agent_configs: List of agent configurations in chain order
        ray_worker_group_cls: Ray worker group class
        reward_fn: Reward function
        val_reward_fn: Validation reward function
        env_class: Environment class
        agent_class: Agent class
        env_args: Environment arguments
        agent_args: Agent arguments
        multi_agent_config: Multi-agent configuration options
        **kwargs: Additional configuration options
        
    Returns:
        MultiAgentPPOTrainer configured for Chain of Experts
    """
    workflow = ChainOfExpertsWorkflow(agent_configs)
    return MultiAgentPPOTrainer(
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
        workflow=workflow,
        multi_agent_config=multi_agent_config,
        **kwargs
    ) 