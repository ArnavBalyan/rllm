import asyncio
import concurrent.futures
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.misc import colorful_print
from rllm.parser import ChatTemplateParser

logger = logging.getLogger(__name__)


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engines=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter
        self.rollout_engines = rollout_engines
        self.n_rollout_engines = len(self.rollout_engines)

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = [2048, 3072]
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [[None for _ in range(self.n_rollout_engines)] for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engines = [
                OpenAIEngine(
                    **rollout_engine_args,
                    api_retries=api_retries,
                    tokenizer=self.tokenizer,
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length[i],
                    disable_thinking=kwargs.get("disable_thinking", False),
                )
                for i, _ in enumerate(rollout_engines)
            ]
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engines = [
                VerlEngine(
                    config=self.config,
                    rollout_manager=manager,
                    tokenizer=self.tokenizer,
                    disable_thinking=self.config.rllm.disable_thinking,
                )
                for manager in rollout_engines
            ]
        
        self.n_rollout_engines = len(self.rollout_engines)

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, prompt, application_id, agent_id=0, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            agent_id: Index of the agent/rollout engine to use (default: 0)
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        
        # Select the correct rollout engine
        rollout_engine = self.rollout_engines[agent_id]

        if self.engine_name == "openai":
            output = await rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output.text
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: 2D list of agents - agents[env_idx][agent_id] = agent for environment env_idx and rollout engine agent_id
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        env = self.envs[idx]
        loop = asyncio.get_event_loop()
        
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps
        
        termination_reason = {}
        prompt_token_len = {}
        prompt_tokens = {}
        response_token_len = {}
        response_tokens = {}
        response_masks = {}
        reward_time = {}
        llm_time = {}
        env_time = {}
        reward = {}
        episode_steps = {}
        total_time = 0.0
        
        for engine_id in range(self.n_rollout_engines):
            agent = self.agents[idx][engine_id]
            agent.reset()
            agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)
            
            messages = agent.chat_completions
            tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
            
            termination_reason[engine_id] = None
            prompt_token_len[engine_id] = len(tokens)
            prompt_tokens[engine_id] = tokens
            response_token_len[engine_id] = 0
            response_tokens[engine_id] = []
            response_masks[engine_id] = []
            reward_time[engine_id] = None
            llm_time[engine_id] = 0.0
            env_time[engine_id] = 0.0
            reward[engine_id] = 0.0
            episode_steps[engine_id] = []
            
            if prompt_token_len[engine_id] > self.max_prompt_length:
                raise Exception(f"Trajectory {idx} Engine {engine_id}: initial prompt length {prompt_token_len[engine_id]} already exceeded max_prompt_length {self.max_prompt_length}")

        for step_idx in range(self.max_steps):
            actions = {}
            upstream_response = None
            
            # Run each engine in sequence
            for engine_id in range(self.n_rollout_engines):
                # Skip engines that have already terminated in a previous step
                if termination_reason.get(engine_id) is not None:
                    continue

                # If this engine has already exhausted its token budget in a previous step,
                # mark truncation and skip calling the model (mirrors single-engine step boundary)
                if (not self.enforce_max_prompt_length) and response_token_len[engine_id] >= self.max_response_length[engine_id]:
                    termination_reason[engine_id] = "TRUNCATION"
                    continue

                agent = self.agents[idx][engine_id]
                
                # Inject upstream recommendation BEFORE calling model for downstream engines
                if engine_id > 0 and upstream_response is not None:
                    agent.chat_completions.append({
                        "role": "user",
                        "content": f"Recommendation from the proposer: {upstream_response}",
                        "_is_upstream_recommendation": True  # Signal for tokenization
                    })
                
                # Remove signal field before sending to model
                prompt_messages = [
                    {k: v for k, v in msg.items() if k != '_is_upstream_recommendation'}
                    for msg in agent.chat_completions
                ]
                # print(f"\n[Engine {engine_id}] [Step {step_idx}] Prompt Messages ({prompt_messages} ")
                
                if not self.enforce_max_prompt_length:
                    max_tokens = self.max_response_length[engine_id] - response_token_len[engine_id]
                    print(
                        f"[DEBUG] Step {step_idx} Engine {engine_id} computing max_tokens: "
                        f"{self.max_response_length[engine_id]} - {response_token_len[engine_id]} = {max_tokens}"
                    )
                else:
                    max_tokens = self.max_response_length[engine_id]
                    # since max prompt is enforced, we filter out too long prompts.
                    prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                    prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                    if prompt_len > self.max_prompt_length:
                        termination_reason[engine_id] = "PROMPT_TRUNCATION"
                        should_break = True
                        break

                print(f"[DEBUG] Step {step_idx} Engine {engine_id} calling model with max_tokens={max_tokens}")
                kwargs["max_tokens"] = max_tokens

                start_time = time.time()
                response = await self.get_model_response(prompt_messages, application_id, agent_id=engine_id, **kwargs)
                delta_time = time.time() - start_time
                llm_time[engine_id] += delta_time
                total_time += delta_time
                
                prompt_response_pair = {
                    "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                    "response": response,
                }
                episode_steps[engine_id].append(prompt_response_pair)

                action_obj: Action = agent.update_from_model(response)
                actions[engine_id] = action_obj.action
                
                if engine_id == 0:
                    upstream_response = response

            # Use last engine's action for environment step
            action = actions[self.n_rollout_engines - 1]
            
            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, env_reward, global_done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                env_reward = 0
                global_done = True
                break

            info["max_steps"] = self.max_steps
            should_break = False
            # Update each agent with environment result
            for engine_id in range(self.n_rollout_engines):
                agent = self.agents[idx][engine_id]
                info["cur_tokens"] = response_token_len[engine_id]
                
                reward[engine_id] = env_reward

                # Update agent internal state.
                agent.update_from_env(
                    observation=next_observation,
                    reward=env_reward,
                    done=global_done,
                    info=info,
                )

                cur_step = agent.get_current_state()
                cur_step.reward = env_reward
                cur_step.done = global_done
                cur_step.info.update(info)
     
                chat_completions_messages = agent.chat_completions
                assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)
     
                # print(f"\n[Engine {engine_id}] [Step {step_idx}] Recent Messages ({assistant_message}")
                # print(f"\n[Engine {engine_id}] [Step {step_idx}] Environment Messages ({env_messages})")

                # Check and convert to tokens if necessary
                assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
                assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
                assistant_msg_tokens, assistant_msg_masks = [], []
                env_msg_tokens, env_msg_masks = [], []
                if assistant_message:
                    assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
                if env_messages:
                    env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

                # Update repsonse token length
                response_token_len[engine_id] += len(assistant_msg_tokens) + len(env_msg_tokens)
                # Reached maximum number of tokens for the trajectory
                if not self.enforce_max_prompt_length and response_token_len[engine_id] >= self.max_response_length[engine_id]:
                    # Truncation length
                    truncation_length = self.max_response_length[engine_id] - response_token_len[engine_id]
                    # Truncate the response and masks
                    if truncation_length < 0:
                        truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                        truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                    else:
                        # Edge case where the response is exactly the max response length.
                        truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                        truncated_response_masks = assistant_msg_masks + env_msg_masks
                    # Update token collections
                    response_tokens[engine_id].extend(truncated_response_tokens)
                    response_masks[engine_id].extend(truncated_response_masks)

                    cur_step = agent.get_current_state()
                    if response_token_len[engine_id] - len(env_msg_tokens) > self.max_response_length[engine_id]:
                        cur_step.reward = 0.0
                    cur_step.done = True
                    termination_reason[engine_id] = "TRUNCATION"
                    should_break = True
                    # handle returning
                    break

                # Update the token version of trajectory
                response_tokens[engine_id].extend(assistant_msg_tokens)
                response_masks[engine_id].extend(assistant_msg_masks)
                observation = next_observation

                # Check if episode is done
                if global_done:
                    termination_reason[engine_id] = "ENV_DONE"
                    should_break = True
                    continue
                
                response_tokens[engine_id].extend(env_msg_tokens)
                response_masks[engine_id].extend(env_msg_masks)
                
                if step_idx == self.max_steps - 1:
                    termination_reason[engine_id] = "MAX_STEPS"
            
            # Break outer loop if environment is done
            if should_break:
                break

        # Process each engine's results
        for engine_id in range(self.n_rollout_engines):
            agent = self.agents[idx][engine_id]
            
            masked_out = False
            if self.overlong_filter:
                if termination_reason[engine_id] in ["TRUNCATION", "MAX_STEPS", "TIMEOUT"]:
                    # Mask out the entire response for overlong trajectories if the reward is 0.
                    response_masks[engine_id] = [0] * len(response_masks[engine_id])
                    masked_out = True

            if hasattr(env, "compute_final_reward") and not masked_out:
                start_time = time.time()
                _final_reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
                reward_time[engine_id] = time.time() - start_time
                cur_step = agent.get_current_state()
                cur_step.reward = _final_reward
                reward[engine_id] = _final_reward
                
            if termination_reason[engine_id]:
                if reward[engine_id] > 0:
                    color = "green"
                else:
                    color = "yellow"
                colorful_print(
                    f"Trajectory {idx} Engine {engine_id} completed due to: {termination_reason[engine_id]}. Reward is {reward[engine_id]}. \n",
                    color,
                )
                if masked_out:
                    colorful_print(f"Trajectory {idx} Engine {engine_id} is masked out due to overlong filter.", "red")

            trajectory = agent.trajectory
            compute_trajectory_reward(trajectory)
            compute_mc_return(trajectory, gamma=self.gamma)
        
        await loop.run_in_executor(self.executor, env.close)

        if mode == "Text":
            return self.agents[idx][0].trajectory
        elif mode == "Token":
            # Determine which engines to include based on truncation
            # If any engine truncated, only include that engine
            # If no truncation, include all engines
            truncated_engine = None
            for engine_id in range(self.n_rollout_engines):
                if termination_reason[engine_id] == "TRUNCATION":
                    truncated_engine = engine_id
                    break
            
            engines_result = {}
            for engine_id in range(self.n_rollout_engines):
                # Skip this engine if another engine truncated
                if truncated_engine is not None and engine_id != truncated_engine:
                    continue
                    
                agent = self.agents[idx][engine_id]
                engines_result[engine_id] = {
                    "prompt_tokens": torch.tensor(prompt_tokens[engine_id], dtype=torch.long),
                    "response_tokens": torch.tensor(response_tokens[engine_id], dtype=torch.long),
                    "response_masks": torch.tensor(response_masks[engine_id], dtype=torch.long),
                    "trajectory_reward": agent.trajectory.reward,
                    "chat_completions": agent.chat_completions,
                    "metrics": {
                        "steps": len(agent.trajectory.steps),
                        "reward_time": reward_time[engine_id],
                        "env_time": env_time[engine_id],
                        "llm_time": llm_time[engine_id],
                        "total_time": total_time,
                    },
                }
            token_result = {
                "idx": env.idx,
                "agent_level_result": engines_result,
            }
            return token_result
        elif mode == "Conversation":
            return self.agents[idx][0].chat_completions
        elif mode == "Step":
            engines_step_result = {}
            for engine_id in range(self.n_rollout_engines):
                agent = self.agents[idx][engine_id]
                engines_step_result[engine_id] = {
                    "steps": episode_steps[engine_id],
                    "trajectory_reward": agent.trajectory.reward,
                    "mc_returns": [step.mc_return for step in agent.trajectory.steps][: len(episode_steps[engine_id])],
                }
            steps_result = {
                "idx": env.idx,
                "agent_level_result": engines_step_result,
            }
            return steps_result

    async def run_agent_trajectory_with_retry(self, idx, application_id, seed=0, mode="Text", **kwargs):
        for _ in range(self.retry_limit):
            try:
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception:
                traceback.print_exc()
                continue
        traceback.print_exc()
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message")

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            for rollout_engine in self.rollout_engines:
                rollout_engine.wake_up()

        async def launch_one_trajectory_task(env_idx: int):
            try:
                application_id = str(uuid.uuid4())
                result = await self.run_agent_trajectory_with_retry(
                    idx=env_idx,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    **kwargs,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                raise e
            return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            for rollout_engine in self.rollout_engines:
                rollout_engine.sleep()

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
