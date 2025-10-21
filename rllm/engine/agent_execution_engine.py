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

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        if rollout_engines is None or len(rollout_engines) == 0:
            raise ValueError("rollout_engines must be a non-empty list")
        self.num_agents = len(rollout_engines)
        # agents[env_idx][agent_id] = agent for environment env_idx and rollout engine agent_id
        self.agents = [[None for _ in range(self.num_agents)] for _ in range(n_parallel_agents)]
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
                    max_response_length=self.max_response_length,
                    disable_thinking=kwargs.get("disable_thinking", False),
                )
                for _ in rollout_engines
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
        
        # Get the specific rollout engine for this agent
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
        # agents is now 2D: agents[env_idx][agent_id]
        assert len(agents) == len(envs), f"Number of agent rows must equal to number of environments but received {len(agents)} and {len(envs)}"
        assert all(len(agents[i]) == self.num_agents for i in range(len(agents))), f"Each environment must have {self.num_agents} agents"
        
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run trajectories for all agents asynchronously and return structured result
        
        Returns:
            dict: {
                "idx": env.idx,
                "agent_level_result": {
                    agent_id: {
                        "prompt_tokens": torch.tensor,
                        "response_tokens": torch.tensor,
                        "response_masks": torch.tensor,
                        "trajectory_reward": float,
                        "chat_completions": list,
                        "metrics": dict
                    }
                }
            }
        """
        
        # Per-env state
        agent_level_result = {}
        proposer_responses_by_step: list[str] = []
        shared_response_token_len = 0
        env = self.envs[idx]
        agents_row = self.agents[idx]

        # for step return (log follower steps if needed)
        episode_steps = []
        # Reset environment ONCE
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps
        # Init ALL agents with same observation and compute their initial prompt tokens
        prompt_tokens_per_agent = {aid: [] for aid in range(self.num_agents)}
        response_tokens_per_agent = {aid: [] for aid in range(self.num_agents)}
        response_masks_per_agent  = {aid: [] for aid in range(self.num_agents)}
        response_token_len_per_agent = {aid: 0 for aid in range(self.num_agents)}
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        termination_reason = None
        reward = 0.0
        for aid in range(self.num_agents):
            ag = agents_row[aid]
            ag.reset()
            ag.update_from_env(observation=observation, reward=0.0, done=False, info=dict(info))
            msgs = ag.chat_completions
            ptoks, _ = convert_messages_to_tokens_and_masks(msgs, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
            prompt_tokens_per_agent[aid] = ptoks
            if len(ptoks) > self.max_prompt_length:
                raise Exception(f"Trajectory {idx}: initial prompt length {len(ptoks)} exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Get action from agent
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - shared_response_token_len
            else:
                max_tokens = self.max_response_length

            # 1) Agent 0 (proposer) generates
            a0 = agents_row[0]
            prompt0 = a0.chat_completions.copy()
            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0
                done = True
                break

            info["max_steps"] = self.max_steps
            # 4) Update ALL agents with the same transition
            for aid in range(self.num_agents):
                agents_row[aid].update_from_env(
                    observation=next_observation,
                    reward=reward,
                    done=done,
                    info=dict(info),
                )

            # 5) Commit ENV messages for both agents; filter synthetic hint
            for aid in range(self.num_agents):
                chat_msgs = agents_row[aid].chat_completions
                _, env_messages = get_recent_assistant_user_messages(chat_msgs)
                if env_messages:
                    env_messages = [
                        m for m in env_messages
                        if not m.get("_synthetic") and not m.get("content", "").startswith("Recommendation from proposer:")
                    ]
                env_msg_tokens, env_msg_masks = [], []
                if env_messages:
                    env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)
                add_env = len(env_msg_tokens)
                response_tokens_per_agent[aid].extend(env_msg_tokens)
                response_masks_per_agent[aid].extend(env_msg_masks)
                response_token_len_per_agent[aid] += add_env
                shared_response_token_len += add_env
            
            # Check truncation AFTER all agents' env messages collected
            if not self.enforce_max_prompt_length and shared_response_token_len >= self.max_response_length:
                for aid in range(self.num_agents):
                    cs = agents_row[aid].get_current_state()
                    if cs:
                        cs.reward = 0.0
                        cs.done = True
                termination_reason = "TRUNCATION"
                break
            
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                for aid in range(self.num_agents):
                    cs = agents_row[aid].get_current_state()
                    if cs:
                        cs.done = True
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break
            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                for aid in range(self.num_agents):
                    response_masks_per_agent[aid] = [0] * len(response_masks_per_agent[aid])
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            for aid in range(self.num_agents):
                cs = agents_row[aid].get_current_state()
                if cs:
                    cs.reward = reward

        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        # Aggregate per-agent outputs
        for aid in range(self.num_agents):
            agent = agents_row[aid]
            trajectory: Trajectory = agent.trajectory
            compute_trajectory_reward(trajectory)
            compute_mc_return(trajectory, gamma=self.gamma)
            if mode == "Text":
                agent_level_result[aid] = trajectory
            elif mode == "Token":
                token_result = {
                    "prompt_tokens": torch.tensor(prompt_tokens_per_agent[aid], dtype=torch.long),
                    "response_tokens": torch.tensor(response_tokens_per_agent[aid], dtype=torch.long),
                    "response_masks": torch.tensor(response_masks_per_agent[aid], dtype=torch.long),
                    "trajectory_reward": trajectory.reward,
                    "chat_completions": agent.chat_completions,
                    "metrics": {
                        "steps": len(trajectory.steps),
                        "reward_time": reward_time,
                        "env_time": env_time,
                        "llm_time": llm_time,
                        "total_time": total_time,
                    },
                }
                agent_level_result[aid] = token_result
            elif mode == "Conversation":
                agent_level_result[aid] = agent.chat_completions
            elif mode == "Step":
                steps_result = {
                    "steps": episode_steps,
                    "trajectory_reward": trajectory.reward,
                    "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
                }
                agent_level_result[aid] = steps_result
        
        # Return structured result with idx and agent_level_result
        return {
            "idx": idx,
            "agent_level_result": agent_level_result
        }

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
            for engine in self.rollout_engines:
                engine.wake_up()

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
            for engine in self.rollout_engines:
                engine.sleep()

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
                    # Build a row of agents for this env
                    self.agents[index] = [self.agent_class(**self.agent_args) for _ in range(self.num_agents)]
                    for ag in self.agents[index]:
                        assert isinstance(ag, BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                        ag.trajectory.task = task  # type: ignore
                    result_dict = await self.run_agent_trajectory_async(index, application_id=task_id)
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, result_dict
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
