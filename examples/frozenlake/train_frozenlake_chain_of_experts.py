import hydra
import ray
from omegaconf import DictConfig

from rllm.agents.frozenlake_multi_agent import (
    FrozenLakeProposerAgent,
    FrozenLakeExpertAgent, 
    FrozenLakeJudgeAgent,
)
from rllm.data import DatasetRegistry
from rllm.engine.multi_agent_execution_engine import (
    AgentConfig,
    AgentRole,
    ChainOfExpertsWorkflow
)
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.trainer.verl.multi_agent_ppo_trainer import (
    MultiAgentPPOTrainer,
    create_chain_of_experts_trainer
)


def create_frozenlake_chain_of_experts_agents(config):
    from rllm.engine.multi_agent_execution_engine import AgentConfig, AgentRole
    from rllm.agents.frozenlake_multi_agent import (
        FrozenLakeProposerAgent,
        FrozenLakeExpertAgent, 
        FrozenLakeJudgeAgent
    )
    
    multi_agent_section = config.get("multi_agent", {})
    
    agent_configs = [
        AgentConfig(
            agent_id="proposer",
            agent_class=FrozenLakeProposerAgent,
            agent_args={"max_steps": config.agent.max_steps},
            role=AgentRole.PROPOSER,
            model_path=config.actor_rollout_ref.model.path,  # Can be different model
            temperature=0.7,
            top_p=0.9,
        ),
        # AgentConfig(
        #     agent_id="expert", 
        #     agent_class=FrozenLakeExpertAgent,
        #     agent_args={"max_steps": config.agent.max_steps},
        #     role=AgentRole.SPECIALIST,
        #     temperature=0.5,
        #     top_p=0.8,
        # ),
        AgentConfig(
            agent_id="judge",
            agent_class=FrozenLakeJudgeAgent,
            agent_args={"max_steps": config.agent.max_steps},
            role=AgentRole.JUDGE,
            model_path=config.actor_rollout_ref.model.path,  # Can be different model
            temperature=0.3,
            top_p=0.7,
        )
    ]
    
    return agent_configs


@ray.remote(num_cpus=1)
def train_frozenlake_chain_of_experts(config, agent_class=None, env_class=None, agent_args=None, env_args=None):
    """
    Multi-agent training function that sets up all required infrastructure.
    """
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        
        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    # Resource pools are now managed internally by MultiAgentPPOTrainer
    # when train_all_agents=true is set
    resource_pool_spec = {
        "global_pool_id": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: "global_pool_id",
        Role.Critic: "global_pool_id",
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if env_class is None:
        env_class = FrozenLakeEnv
    if agent_class is None:
        agent_class = FrozenLakeProposerAgent

    env_args = env_args or {}
    agent_args = agent_args or {}
    if config.env.get("env_args") is not None:
        env_args.update(config.env.get("env_args"))
    if config.agent.get("agent_args") is not None:
        agent_args.update(config.agent.get("agent_args"))

    agent_configs = create_frozenlake_chain_of_experts_agents(config)
    
    for agent_config in agent_configs:
        print(f"  - {agent_config.agent_id} ({agent_config.role.value}): {agent_config.agent_class.__name__}")

    # Add multi_agent config to main config
    config.multi_agent = {
        "training_mode": "final_agent",  
        "reward_aggregation": "final_agent",
        "train_all_agents": True,  # Enable independent agent training
        "reward_mode": config.multi_agent.reward_mode
    }
    
    multi_agent_config = config.multi_agent

    # Create Chain of Experts workflow directly
    from rllm.engine.multi_agent_execution_engine import ChainOfExpertsWorkflow
    workflow = ChainOfExpertsWorkflow(agent_configs)
    
    # Create MultiAgentPPOTrainer directly without base trainer
    trainer = MultiAgentPPOTrainer(
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
    )
    
    trainer.init_workers()
    trainer.fit_multi_agent()
    
    print("Training completed!")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    """Main function that starts the Ray-based training"""
    
    print("Starting FrozenLake Chain of Experts Training")
    print("=" * 60)
    
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(train_frozenlake_chain_of_experts.remote(
        config, 
        agent_class=FrozenLakeProposerAgent, 
        env_class=FrozenLakeEnv,
        agent_args={},
        env_args={}
    ))


if __name__ == "__main__":
    main() 