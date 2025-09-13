from typing import Any, Dict, List, Optional

from rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from rllm.engine.multi_agent_execution_engine import AgentRole


class MultiAgentBase(BaseAgent):
    """Base class for Multi-Agent workflows"""
    
    def __init__(
        self, 
        agent_id: str, 
        role: AgentRole = AgentRole.SPECIALIST,
        system_prompt: str = "",
        **kwargs
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self._trajectory = Trajectory()
        self.multi_agent_context: Dict[str, Any] = {}
        
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """
        current_obs_str = str(observation)
        user_prompt_content = f"Current Observation ({self.step}): \n" + current_obs_str + "\n" + "You have not achieved the goal, P has not reached G yet. Please give the next action."

        if self._trajectory.steps and self._trajectory.steps[-1].action is not None:  # Check if the last step has an action (meaning it's a completed step)
            last_step_obs_str = self._trajectory.steps[-1].observation
            if last_step_obs_str == current_obs_str:
                user_prompt_content += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."

        if self.max_steps is not None and self.max_steps - self.step > 0:
            user_prompt_content += f"\nThe maximum number of steps remaining is {self.max_steps - self.step}."

        self.messages.append({"role": "user", "content": user_prompt_content})
        self.current_observation = current_obs_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        content = response

        if not self.accumulate_thinking:
            _, sep, after = content.partition("</think>")
            if sep:
                content = after

        thought, action_str = self._parse_model_response(content)

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), thought=thought, action=action_str, model_response=content, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        self.messages.append({"role": "assistant", "content": content})

        self.step += 1

        return Action(action=action_str)

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

        thought = response
        action_str = str(FrozenLakeEnv.INVALID_ACTION)

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = response.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = response[:last_match_index].strip()

            extracted_text = last_match_content.lower()

            if extracted_text in DIRECTION_MAP:
                action_str = str(DIRECTION_MAP[extracted_text])
            elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
                action_str = str(int(extracted_text))

        return thought, action_str

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT if not self.multistep_prompt else self.MULTI_SHOT_SYSTEM_PROMPT,
            }
        ]
        self.step = 0
