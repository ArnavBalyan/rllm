import hydra
from omegaconf import DictConfig

from rllm.agents.frozenlake_multi_agent import (
    FrozenLakeProposerAgent,
    FrozenLakeExpertAgent, 
    FrozenLakeJudgeAgent,
    FrozenLakeMultiAgentEnv
)
from rllm.data import DatasetRegistry
from rllm.engine.multi_agent_execution_engine import (
    AgentConfig,
    AgentRole,
    ChainOfExpertsWorkflow
)
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.environments.multi_agent_env import MultiAgentEnv
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.trainer.verl.multi_agent_ppo_trainer import (
    MultiAgentPPOTrainer,
    create_chain_of_experts_trainer
)


class FrozenLakeChainOfExpertsEnv(MultiAgentEnv):
    """
    FrozenLake environment adapted for Chain of Experts workflow.
    
    Inherits from MultiAgentEnv to handle context passing between agents.
    """
    
    def __init__(self, **kwargs):
        # Initialize base FrozenLake environment
        self.base_env = FrozenLakeEnv(**kwargs)
        super().__init__()
    
    def reset(self):
        """Reset the environment for a new episode"""
        observation, info = self.base_env.reset()
        self.multi_agent_context = {}
        self.agent_history = []
        return observation, info
    
    def reset_with_input(self, agent_input):
        """Reset with input from previous agents in the chain"""
        super().reset_with_input(agent_input)
        observation, info = self.base_env.reset()
        return observation, info
    
    def step(self, action):
        """Take a step in the environment"""
        return self.base_env.step(action)
    
    def render(self, *args, **kwargs):
        """Render the environment"""
        return self.base_env.render(*args, **kwargs)
    
    def finished(self):
        """Check if episode is finished"""
        return self.base_env.finished()
    
    def success(self):
        """Check if goal was reached"""
        return self.base_env.success()
    
    @classmethod
    def from_dict(cls, env_dict):
        """Create environment from dictionary (required for dataset loading)"""
        return cls(**env_dict)
    
    def __getattr__(self, name):
        """Forward all other attributes to base environment"""
        return getattr(self.base_env, name)


def create_frozenlake_chain_of_experts_agents():
    """Create the Chain of Experts agent configuration for FrozenLake"""
    
    agents = [
        AgentConfig(
            agent_id="proposer",
            agent_class=FrozenLakeProposerAgent,
            agent_args={"max_steps": 10},
            role=AgentRole.PROPOSER,
            temperature=0.7,
            top_p=0.9
        ),
        AgentConfig(
            agent_id="expert", 
            agent_class=FrozenLakeExpertAgent,
            agent_args={"max_steps": 10},
            role=AgentRole.SPECIALIST,
            temperature=0.7,
            top_p=0.9
        ),
        AgentConfig(
            agent_id="judge",
            agent_class=FrozenLakeJudgeAgent, 
            agent_args={"max_steps": 10},
            role=AgentRole.JUDGE,
            temperature=0.7,
            top_p=0.9
        )
    ]
    
    return agents


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    """Main training function for FrozenLake Chain of Experts"""
    
    print("Starting FrozenLake Chain of Experts Training")
    print("=" * 60)
    
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("frozenlake", "train")
    val_dataset = DatasetRegistry.load_dataset("frozenlake", "test")
    
    print(f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    agent_configs = create_frozenlake_chain_of_experts_agents()
    
    print("Created Chain of Experts agents:")
    for agent_config in agent_configs:
        print(f"  - {agent_config.agent_id} ({agent_config.role.value}): {agent_config.agent_class.__name__}")
    
    base_trainer = AgentPPOTrainer(
        agent_class=FrozenLakeProposerAgent,
        env_class=FrozenLakeChainOfExpertsEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    print("Created base trainer")
    
    multi_agent_config = {
        "training_mode": "final_agent",  # Train on final judge's decision
        "reward_aggregation": "final_agent",  # Use final judge's reward
    }
    
    trainer = create_chain_of_experts_trainer(
        base_trainer=base_trainer,
        agent_configs=agent_configs,
        multi_agent_config=multi_agent_config
    )
    
    print(f"   Training mode: {multi_agent_config['training_mode']}")
    print(f"   Reward aggregation: {multi_agent_config['reward_aggregation']}")
    print("=" * 60)
    
    print("Starting Chain of Experts training...")
    trainer.fit_multi_agent()
    
    print("Training completed!")


if __name__ == "__main__":
    main() 