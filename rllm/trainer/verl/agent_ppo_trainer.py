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

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from copy import deepcopy
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    marked_timer,
    reduce_metrics,
)
from rllm.trainer.verl.worker_group_manager import WorkerGroupManager


class AgentPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer,
        worker_group_managers: list[WorkerGroupManager],
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.worker_group_managers = worker_group_managers
        self.reward_fn = worker_group_managers[0].reward_fn
        self.val_reward_fn = worker_group_managers[0].val_reward_fn
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}
        self.global_steps = 0

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"

        if self.config.rllm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        # Initialize all managers and collect their rollout engines
        rollout_engines = []
        for manager in self.worker_group_managers:
            manager.init_and_get_worker_groups()
            rollout_engines.append(manager.async_rollout_manager)
        
        # Create single execution engine with all rollout engines
        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engines=rollout_engines,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            **self.config.rllm.agent.get("engine_args", {}),
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents for each environment and rollout engine (num_envs x num_agents)
        # agents[env_idx][agent_id] = agent for environment env_idx and rollout engine agent_id
        num_agents = len(self.worker_group_managers)
        agents = [[None] * num_agents for _ in range(len(envs))]
        
        def _create_agent_for_env_and_agent_id(env_idx, agent_id):
            return env_idx, agent_id, self.agent_class(**full_agent_args)
        
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for env_idx in range(len(envs)):
                for agent_id in range(num_agents):
                    futures.append(executor.submit(_create_agent_for_env_and_agent_id, env_idx, agent_id))
            
            for future in as_completed(futures):
                env_idx, agent_id, agent = future.result()
                agents[env_idx][agent_id] = agent
        
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        manager = self.worker_group_managers[0]

        # load checkpoint before doing anything
        manager._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in manager.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    # Generate trajectories for all agents
                    if self.config.rllm.stepwise_advantage.enable:
                        # For stepwise advantage mode, generate steps for all agents
                        # TODO: Make generate_agent_steps return per-agent outputs
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # For now, all agents share the same steps output
                        agent_outputs = {agent_id: (final_gen_batch_output, {}) for agent_id in range(len(self.worker_group_managers))}
                    else:
                        # Generate trajectories for all agents (returns dict of per-agent outputs)
                        agent_outputs = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                    
                    final_batch = None 
                    # Process each agent separately
                    for agent_id, agent_manager in enumerate(self.worker_group_managers):
                        # Skip if agent has no training data (all samples truncated/invalidated)
                        if agent_id not in agent_outputs:
                            print(f"⚠️  Agent {agent_id} has no training data for this batch (all samples truncated), skipping...")
                            metrics[f"agent_{agent_id}_batch/skipped_all_truncated"] = 1
                            continue
                        
                        # Extract this agent's trajectory output and union with prompts
                        final_gen_batch_output, generate_metrics = agent_outputs[agent_id]

                        # Filter local_batch to only include indices that are present in final_gen_batch_output
                        # This handles the case where some trajectories were truncated/invalidated
                        if "idxs" in final_gen_batch_output.non_tensor_batch:
                            valid_idxs = final_gen_batch_output.non_tensor_batch["idxs"]
                            local_batch = type(batch)(
                                batch=batch.batch.clone(),
                                non_tensor_batch=deepcopy(batch.non_tensor_batch),
                                meta_info=deepcopy(batch.meta_info)
                            )
                            # Select only the rows corresponding to valid trajectories (with idx alignment)
                            local_batch = local_batch.select_idxs(valid_idxs)
                            
                            # Track truncation for metrics
                            num_truncated = len(batch) - len(valid_idxs)
                            if num_truncated > 0:
                                metrics[f"agent_{agent_id}_batch/num_truncated"] = num_truncated
                        else:
                            # Fallback for stepwise mode or other cases where idxs not provided
                            local_batch = type(batch)(
                                batch=batch.batch.clone(),
                                non_tensor_batch=deepcopy(batch.non_tensor_batch),
                                meta_info=deepcopy(batch.meta_info)
                            )

                        agent_batch = local_batch.union(final_gen_batch_output) #todo this may need a deep copy of batch
                        agent_metrics = {f"agent_{agent_id}_{k}": v for k, v in generate_metrics.items()}
                        metrics.update(agent_metrics)

                        # compute values
                        if agent_manager.use_critic:
                            with marked_timer(f"values_agent_{agent_id}", timing_raw):
                                values = agent_manager.critic_wg.compute_values(agent_batch)
                                agent_batch = agent_batch.union(values)

                        with marked_timer(f"adv_agent_{agent_id}", timing_raw):
                            if agent_manager.use_rm:
                                reward_tensor = agent_manager.rm_wg.compute_rm_score(agent_batch)
                                agent_batch = agent_batch.union(reward_tensor)

                            # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                            if "token_level_scores" not in agent_batch.batch:
                                reward_tensor = self.reward_fn(agent_batch)
                                agent_batch.batch["token_level_scores"] = reward_tensor
                            else:
                                reward_tensor = agent_batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                            # Rejection sampling based on rewards
                            # Group rewards by uid
                            uids = agent_batch.non_tensor_batch["uid"]
                            unique_uids = np.unique(uids)
                            valid_mask = torch.ones(len(uids), dtype=torch.bool)
                            solve_none = 0
                            solve_all = 0
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                                # Check if all rewards are <= 0 or all are 1 >= for this uid
                                if (uid_rewards <= 0).all():
                                    valid_mask[uid_mask] = False
                                    solve_none += 1
                                elif (uid_rewards >= 1).all():
                                    valid_mask[uid_mask] = False
                                    solve_all += 1

                            # Log to metrics
                            metrics[f"agent_{agent_id}_batch/solve_none"] = solve_none
                            metrics[f"agent_{agent_id}_batch/solve_all"] = solve_all
                            metrics[f"agent_{agent_id}_batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                            if self.config.rllm.rejection_sample.enable:
                                # log the actual complete training rewards before rejection sampling
                                token_level_rewards = None  # for metrics calculation
                                if self.config.rllm.stepwise_advantage.enable:
                                    is_pad_step = agent_batch.non_tensor_batch["is_pad_step"]
                                    non_pad_step_indices = np.where(is_pad_step == False)[0]
                                    non_pad_steps = agent_batch.select_idxs(non_pad_step_indices)
                                    is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                    valid_last_step_indices = np.where(is_last_step == True)[0]
                                    last_step_batch = agent_batch.select_idxs(valid_last_step_indices)
                                    token_level_rewards = last_step_batch.batch["token_level_scores"]
                                else:
                                    token_level_rewards = agent_batch.batch["token_level_scores"]
                                full_sequence_score = token_level_rewards.sum(-1)
                                metrics[f"agent_{agent_id}_critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                                metrics[f"agent_{agent_id}_critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                                metrics[f"agent_{agent_id}_critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                                # If no valid samples remain, skip this agent
                                if not valid_mask.any():
                                    continue

                                # Filter batch to keep only valid samples
                                agent_batch = agent_batch[valid_mask]

                                if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                    # agent_batch now only contains steps with valid uids
                                    # filter out padding steps
                                    is_pad_step = agent_batch.non_tensor_batch["is_pad_step"]
                                    non_pad_step_indices = np.where(is_pad_step == False)[0]
                                    agent_batch = agent_batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                    # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                    # separate out last step and intermediate steps
                                    is_last_step = agent_batch.non_tensor_batch["is_last_step"]
                                    valid_last_step_indices = np.where(is_last_step == True)[0]
                                    not_last_step_indices = np.where(is_last_step == False)[0]
                                    last_step_batch = agent_batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                    non_last_step_batch = agent_batch.select_idxs(not_last_step_indices)

                                    # filter last_step_batch to make sure its multiple of world size
                                    num_trainer_replicas = agent_manager.actor_rollout_wg.world_size
                                    max_batch_size = (
                                        last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                        // num_trainer_replicas
                                    ) * num_trainer_replicas
                                    if not max_batch_size:
                                        # give up, you got everything either all wrong or right.
                                        continue

                                    size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                    size_mask[:max_batch_size] = True
                                    last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                    # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                    valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                    non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                    non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                    non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                    # concatenate then pad
                                    agent_batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                    agent_batch = self._pad_dataproto_to_world_size(agent_batch)
                                else:
                                    # Round down to the nearest multiple of world size
                                    num_trainer_replicas = agent_manager.actor_rollout_wg.world_size
                                    max_batch_size = (agent_batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                    if not max_batch_size:
                                        # give up, you got everything either all wrong or right.
                                        continue

                                    size_mask = torch.zeros(agent_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                    size_mask[:max_batch_size] = True
                                    agent_batch = agent_batch[size_mask]

                            # recompute old_log_probs
                            with marked_timer(f"old_log_prob_agent_{agent_id}", timing_raw):
                                old_log_prob = agent_manager.actor_rollout_wg.compute_log_prob(agent_batch)
                                agent_batch = agent_batch.union(old_log_prob)

                            # recompute old_log_probs
                            with marked_timer(f"old_log_prob_agent_{agent_id}_2", timing_raw, color="blue"):
                                old_log_prob = agent_manager.actor_rollout_wg.compute_log_prob(agent_batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = agent_batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {f"agent_{agent_id}_actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                agent_batch = agent_batch.union(old_log_prob)

                                if "rollout_log_probs" in agent_batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    rollout_old_log_probs = agent_batch.batch["rollout_log_probs"]
                                    actor_old_log_probs = agent_batch.batch["old_log_probs"]
                                    attention_mask = agent_batch.batch["attention_mask"]
                                    responses = agent_batch.batch["responses"]
                                    response_length = responses.size(1)
                                    response_mask = attention_mask[:, -response_length:]

                                    rollout_probs = torch.exp(rollout_old_log_probs)
                                    actor_probs = torch.exp(actor_old_log_probs)
                                    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                    rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                    rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                    rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                    metrics.update(
                                        {
                                            f"agent_{agent_id}_training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                            f"agent_{agent_id}_training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                            f"agent_{agent_id}_training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                        }
                                    )

                            if agent_manager.use_reference_policy:
                                # compute reference log_prob
                                with marked_timer(f"ref_agent_{agent_id}", timing_raw):
                                    ref_log_prob = agent_manager.ref_policy_wg.compute_ref_log_prob(agent_batch)
                                    agent_batch = agent_batch.union(ref_log_prob)

                            # compute rewards with KL penalty if needed

                            # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                            # where it is subtracted directly from the policy loss

                            # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            #     agent_batch, kl_metrics = apply_kl_penalty(agent_batch,
                            #                                        kl_ctrl=self.kl_ctrl,
                            #                                        kl_penalty=self.config.algorithm.kl_penalty)
                            #     metrics.update(kl_metrics)
                            # else:
                            #     agent_batch.batch['token_level_rewards'] = agent_batch.batch['token_level_scores']

                            agent_batch.batch["token_level_rewards"] = agent_batch.batch["token_level_scores"]

                            if self.config.rllm.stepwise_advantage.enable:
                                if self.config.rllm.stepwise_advantage.mode == "per_step":
                                    agent_batch.batch["token_level_rewards"] = agent_batch.batch["mc_returns"]
                                    agent_batch.non_tensor_batch["uid"] = agent_batch.non_tensor_batch["step_ids"]

                                    is_pad_step = agent_batch.non_tensor_batch["is_pad_step"]
                                    non_pad_step_indices = np.where(is_pad_step == False)[0]
                                    agent_batch = agent_batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                                elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                                    # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                    is_last_step = agent_batch.non_tensor_batch["is_last_step"]
                                    last_step_indices = np.where(is_last_step == True)[0]
                                    other_step_indices = np.where(is_last_step == False)[0]
                                    other_step_batch = agent_batch.select_idxs(other_step_indices)
                                    agent_batch = agent_batch.select_idxs(last_step_indices)  # This batch only has last steps
                                else:
                                    raise ValueError(f"Stepwise advantage mode {self.config.rllm.stepwise_advantage.mode} not supported")

                            # compute advantages, executed on the driver process
                            agent_batch = compute_advantage(
                                agent_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # remove the padded last steps
                                # Merging the separated out steps using the advantage from last steps
                                self._stepwise_advantage_broadcast(agent_batch, other_step_batch=other_step_batch)
                                # agent_batch = agent_batch.merge(other_step_batch)
                                agent_batch = DataProto.concat([agent_batch, other_step_batch])

                            if self.config.rllm.mask_truncated_samples:
                                mask = agent_batch.batch["attention_mask"][:, -1] == 1
                                agent_batch = agent_batch[~mask]

                            agent_batch = self._pad_dataproto_to_world_size(agent_batch)
                            # balance the number of valid tokens on each dp rank.
                            # Note that this breaks the order of data inside the batch.
                            # Please take care when you implement group based adv computation such as GRPO and rloo
                            agent_manager._balance_batch(agent_batch, metrics=agent_metrics)

                            # compute global_valid tokens
                            agent_batch.meta_info["global_token_num"] = torch.sum(agent_batch.batch["attention_mask"], dim=-1).tolist()

                            # update critic
                            if agent_manager.use_critic:
                                with marked_timer(f"update_critic_agent_{agent_id}", timing_raw):
                                    critic_output = agent_manager.critic_wg.update_critic(agent_batch)
                                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                agent_metrics = {f"agent_{agent_id}_{k}": v for k, v in critic_output_metrics.items()}
                                metrics.update(agent_metrics)

                            # implement critic warmup
                            if self.config.trainer.critic_warmup <= self.global_steps:
                                # update actor
                                with marked_timer(f"update_actor_agent_{agent_id}", timing_raw):
                                    actor_output = agent_manager.actor_rollout_wg.update_actor(agent_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                agent_metrics = {f"agent_{agent_id}_{k}": v for k, v in actor_output_metrics.items()}
                                metrics.update(agent_metrics)
                        final_batch = agent_batch

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw):
                            manager._save_checkpoint()

                # collect metrics (using first agent's batch for overall metrics)
                # Note: In multi-agent setup, agent_batch is local to the agent loop
                # For now, we skip batch-level metrics or compute them differently
                metrics.update(compute_timing_metrics(batch=final_batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= manager.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        manager = self.worker_group_managers[0]
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        val_metrics = {}  # Track validation filtering metrics
        for test_data in manager.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)
                test_batch_union = test_batch.union(test_output_gen_batch)
                reward_tensor = test_batch_union.batch["token_level_scores"]
                rewards_lst.append(reward_tensor.sum(-1).cpu())
                data_source_lst.append(test_batch_union.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
                uid_lst.append(test_batch_union.non_tensor_batch["uid"])
            else:
                agent_outputs = self.generate_agent_trajectory(meta_info=test_batch.meta_info)
                for agent_id, (test_output_gen_batch, _) in agent_outputs.items():
                    if "idxs" in test_output_gen_batch.non_tensor_batch:
                        valid_idxs = test_output_gen_batch.non_tensor_batch["idxs"]
                        local_test_batch = test_batch.select_idxs(valid_idxs)
                        # Track truncated samples
                        num_truncated = len(test_batch) - len(valid_idxs)
                        val_metrics[f"val/agent_{agent_id}_num_truncated"] = val_metrics.get(f"val/agent_{agent_id}_num_truncated", 0) + num_truncated
                    else:
                        local_test_batch = test_batch
                    test_batch_union = local_test_batch.union(test_output_gen_batch)
                    reward_tensor = test_batch_union.batch["token_level_scores"]
                    rewards_lst.append(reward_tensor.sum(-1).cpu())
                    # Tag data source with agent_id to separate metrics per agent
                    base_sources = test_batch_union.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
                    agent_sources = [f"agent_{agent_id}_{src}" for src in base_sources]
                    data_source_lst.append(agent_sources)
                    uid_lst.append(test_batch_union.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: max score}
        data_source_uid_all_rewards = {}  # data source to {uid: [all rewards]} for solve categorization

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}
                data_source_uid_all_rewards[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
                data_source_uid_all_rewards[data_source][uid] = []
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())
            data_source_uid_all_rewards[data_source][uid].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            solve_none = 0
            solve_all = 0
            solve_partial = 0
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
                # Categorize based on ALL attempts for this uid
                uid_rewards = data_source_uid_all_rewards[data_source][uid]
                if all(r <= 0 for r in uid_rewards):
                    solve_none += 1
                elif all(r >= 1 for r in uid_rewards):
                    solve_all += 1
                else:
                    solve_partial += 1
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)
            # Add solve category metrics
            metric_dict[f"val/{data_source}/solve_none"] = solve_none
            metric_dict[f"val/{data_source}/solve_all"] = solve_all
            metric_dict[f"val/{data_source}/solve_partial"] = solve_partial

        # Merge validation filtering metrics
        metric_dict.update(val_metrics)
        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if True:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])
        
        with marked_timer("transform_trajectory", timing_raw):
            # Extract agent-level trajectories and transform them
            agent_level_batch_outputs = {}
            agent_level_metrics = {}
            
            for agent_id in range(len(self.worker_group_managers)):
                # Extract this agent's trajectories from all results
                agent_trajectories = []
                for result in trajectories:
                    agent_level_result = result["agent_level_result"]
                    if agent_id in agent_level_result:
                        agent_traj = agent_level_result[agent_id]
                        # Add idx back for compatibility with transform function
                        agent_traj["idx"] = result["idx"]
                        agent_trajectories.append(agent_traj)
                
                # Skip if no valid trajectories (all samples truncated/invalidated for this agent)
                if len(agent_trajectories) == 0:
                    print(f"⚠️  Agent {agent_id} has no valid trajectories (all samples truncated), skipping agent for this batch...")
                    continue
                
                # Transform this agent's trajectories
                agent_batch_output, agent_metrics = self._transform_agent_trajectories(agent_trajectories, agent_id)
                agent_level_batch_outputs[agent_id] = (agent_batch_output, agent_metrics)
        
        # Return agent-level outputs as dictionary
        return agent_level_batch_outputs

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with marked_timer("collect_trajectory", timing_raw):
            step_results = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, step_result in enumerate(gen_seq_generator):
                step_results.append(step_result)
        # Sort by idx to ensure they are in order
        step_results.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Extract agent 0's steps for backward compatibility
            agent_0_steps = []
            for result in step_results:
                agent_level_result = result["agent_level_result"]
                if 0 in agent_level_result:
                    agent_step = agent_level_result[0]
                    # Add idx back for compatibility
                    agent_step["idx"] = result["idx"]
                    agent_0_steps.append(agent_step)
            
            # Transform the raw trajectories into DataProto format
            final_gen_batch_output = self._transform_agent_steps(agent_0_steps, agent_id=0, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict], agent_id: int):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.
            agent_id (int): Agent ID to determine max_response_length.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        traj_idxs = []  # Preserve idx for matching with original batch
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])
            traj_idxs.append(traj["idx"])  # Extract and preserve idx

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
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

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses (use agent-specific max_response_length)
        max_response_length = self.config.data.max_response_length[agent_id]
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token (e.g., eos token)
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, score in enumerate(traj_scores):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= score_batch.shape[1]:
                score_batch[i, resp_len - 1] = score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "response_mask": traj_mask,
        }
        
        non_tensor_batch = {
            "idxs": np.array(traj_idxs, dtype=np.int64),  # Preserve idx for alignment with original batch
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from rllm.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        # Full attention mask (covers prompt + response); split it into prompt and response parts
        full_attn_mask = tensor_batch.batch["attention_mask"]
        prompt_len = prompts.shape[1]
        resp_len = responses.shape[1]
        prompt_attn_mask = full_attn_mask[:, :prompt_len]
        response_attn_mask = full_attn_mask[:, -resp_len:]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            colorful_print("\n" + "=" * 60, fg="cyan", bold=True)
            colorful_print(f"Sample {i}", fg="cyan", bold=True)

            # Legend before the example
            legend = " ".join(
                [
                    "\x1b[37mwhite=masked\x1b[0m",
                    "\x1b[34mblue=unmasked\x1b[0m",
                    "\x1b[42m green bg=reward>0 \x1b[0m",
                    "\x1b[41m red bg=reward<=0 \x1b[0m",
                ]
            )
            print(f"[{legend}]")

            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_valid_mask = prompt_attn_mask[i].bool()
            # Build one-line colored prompt (prompt is always masked-from-loss => white)
            prompt_parts = []
            for tok_id, is_valid in zip(prompt_tokens.tolist(), prompt_valid_mask.tolist(), strict=False):
                if not is_valid:
                    continue
                tok = self.tokenizer.decode([tok_id]).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                prompt_parts.append(f"\x1b[37m{tok}\x1b[0m")  # white
            print("".join(prompt_parts))

            # Separator line between prompt and response for readability
            print("----------------")

            # Detokenize response with token-level highlighting
            resp_tokens = responses[i]
            resp_valid_mask = response_attn_mask[i].bool()
            loss_mask = traj_mask[i]
            rewards = token_level_scores[i]

            # Pre-compute reward positions (typically only the last valid resp token has nonzero reward)
            reward_idx = None
            reward_value = 0.0
            if rewards is not None:
                # consider only valid response positions
                for j, is_valid in enumerate(resp_valid_mask.tolist()):
                    if not is_valid:
                        continue
                    val = float(rewards[j].item()) if hasattr(rewards[j], "item") else float(rewards[j])
                    if abs(val) > 1e-9:
                        reward_idx = j
                        reward_value = val

            # Fallback: if no nonzero reward found, use the last valid response token
            if reward_idx is None:
                valid_indices = [idx for idx, v in enumerate(resp_valid_mask.tolist()) if v]
                if valid_indices:
                    reward_idx = valid_indices[-1]
                    if rewards is not None:
                        val = float(rewards[reward_idx].item()) if hasattr(rewards[reward_idx], "item") else float(rewards[reward_idx])
                        reward_value = val

            # Colors: white for masked-from-loss; blue for contributes-to-loss; overlay background red/green if reward token
            response_parts = []
            for j, tok_id in enumerate(resp_tokens.tolist()):
                if not bool(resp_valid_mask[j].item() if hasattr(resp_valid_mask[j], "item") else resp_valid_mask[j]):
                    continue
                tok = self.tokenizer.decode([tok_id]).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

                contributes = bool(loss_mask[j].item()) if hasattr(loss_mask[j], "item") else bool(loss_mask[j])
                fg = "\x1b[34m" if contributes else "\x1b[37m"  # blue if in loss, else white

                bg = ""
                if reward_idx is not None and j == reward_idx:
                    bg = "\x1b[42m" if reward_value > 0 else "\x1b[41m"  # green background for positive, red for negative/zero

                response_parts.append(f"{bg}{fg}{tok}\x1b[0m")

            print("".join(response_parts))

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], agent_id: int, uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses (use agent-specific max_response_length)
        max_response_length = self.config.data.max_response_length[agent_id]
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = attention_mask[:, max_prompt_length:]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.worker_group_managers[0].use_critic and self.worker_group_managers[0].critic_wg.world_size != 0:
            world_sizes.append(self.worker_group_managers[0].critic_wg.world_size)
        if self.worker_group_managers[0].use_reference_policy and self.worker_group_managers[0].ref_policy_wg.world_size != 0:
            world_sizes.append(self.worker_group_managers[0].ref_policy_wg.world_size)
        if self.worker_group_managers[0].use_rm and self.worker_group_managers[0].rm_wg.world_size != 0:
            world_sizes.append(self.worker_group_managers[0].rm_wg.world_size)
        if self.worker_group_managers[0].hybrid_engine:
            if self.worker_group_managers[0].actor_rollout_wg.world_size != 0:
                world_sizes.append(self.worker_group_managers[0].actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
